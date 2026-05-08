"""Long-audio performance bench for aistack ASR.

Hits /v1/audio/transcriptions with bench/audio/perf-{12,25,50,97}min.mp3
(or files passed via --file) and reports wall time, RTF, segment
count, VRAM peak (from access log), and a word-LCS quality ratio vs
the lemonfox reference json.

Usage:
    # Default 4 files, no label (outputs to bench/output/)
    python -m bench.perf_eval --model parakeet

    # Tagged variant — outputs to bench/output/<label>/
    python -m bench.perf_eval --model parakeet --label ctx-256 \\
        --file bench/audio/perf-25min.mp3 --file bench/audio/perf-50min.mp3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx


DEFAULT_FILES = [
    "bench/audio/perf-12min.mp3",
    "bench/audio/perf-25min.mp3",
    "bench/audio/perf-50min.mp3",
    "bench/audio/perf-97min.mp3",
]

OUTPUT_ROOT = Path("bench/output")
LOG_DIR = Path("logs")

_NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm(w: str) -> str:
    return _NORM_RE.sub("", w.lower())


def load_reference(audio_path: Path) -> dict | None:
    ref_path = audio_path.with_name(audio_path.stem + "_en.json")
    if not ref_path.exists():
        return None
    return json.loads(ref_path.read_text(encoding="utf-8"))


def _flatten_words(d: dict) -> list[str]:
    if d.get("words"):
        return [w.get("word", "") for w in d["words"]]
    return [w.get("word", "") for s in d.get("segments", []) for w in s.get("words", [])]


def lcs_length(a: list[str], b: list[str]) -> int:
    """Length-only LCS over normalized word strings (memory-efficient)."""
    a = [_norm(w) for w in a if _norm(w)]
    b = [_norm(w) for w in b if _norm(w)]
    n, m = len(a), len(b)
    if not n or not m:
        return 0
    # rolling 1D dp
    prev = [0] * (m + 1)
    for i in range(n):
        cur = [0] * (m + 1)
        ai = a[i]
        for j in range(m):
            if ai == b[j]:
                cur[j + 1] = prev[j] + 1
            else:
                cur[j + 1] = max(cur[j], prev[j + 1])
        prev = cur
    return prev[m]


def quality_pct(ours: dict, ref: dict) -> dict:
    """Word-LCS ratio. Returns {lcs, ours_words, ref_words, recall, precision}."""
    a = _flatten_words(ours)
    b = _flatten_words(ref)
    lcs = lcs_length(a, b)
    a_norm = [w for w in a if _norm(w)]
    b_norm = [w for w in b if _norm(w)]
    return {
        "lcs": lcs,
        "ours_words": len(a_norm),
        "ref_words": len(b_norm),
        "recall_pct": 100.0 * lcs / len(b_norm) if b_norm else None,
        "precision_pct": 100.0 * lcs / len(a_norm) if a_norm else None,
    }


def harvest_access_log(request_id: str, log_dir: Path = LOG_DIR) -> dict | None:
    """Pull the last access-log entry matching request_id."""
    if not log_dir.exists():
        return None
    found = None
    for p in sorted(log_dir.glob("access-*.jsonl")):
        try:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    if request_id not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("request_id") == request_id:
                        found = rec  # keep last match across files
        except OSError:
            continue
    return found


def run_one(client: httpx.Client, base_url: str, model: str,
            language: str, audio_path: Path, output_dir: Path,
            label: str) -> dict:
    ref = load_reference(audio_path)
    # Embed label + epoch so repeated runs do not collide in the log.
    rid = f"perf-{label or 'nolabel'}-{audio_path.stem}-{int(time.time())}"
    t0 = time.perf_counter()
    with open(audio_path, "rb") as fh:
        r = client.post(
            f"{base_url}/v1/audio/transcriptions",
            files={"file": (audio_path.name, fh, "audio/mpeg")},
            data={"model": model, "language": language,
                  "response_format": "verbose_json"},
            headers={"X-Request-ID": rid},
            timeout=3600,
        )
    wall = time.perf_counter() - t0
    if r.status_code != 200:
        return {"file": audio_path.name, "wall": wall,
                "error": f"HTTP {r.status_code}: {r.text[:200]}",
                "request_id": rid}
    body = r.json()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / f"{audio_path.stem}_{model}.json"
    out_json.write_text(json.dumps(body, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    n_segs = len(body.get("segments", []))
    n_words = len(body.get("words") or [])
    if not n_words:
        n_words = sum(len(s.get("words", [])) for s in body.get("segments", []))
    text = body.get("text", "")
    duration = float(body.get("duration", 0.0))

    out = {
        "file": audio_path.name,
        "output_json": str(out_json),
        "duration_sec": duration,
        "wall": wall,
        "rtf": wall / duration if duration > 0 else None,
        "segments": n_segs,
        "words": n_words,
        "text_chars": len(text),
        "request_id": rid,
    }
    # Pull VRAM/observability data from access log (logged after response).
    log_rec = harvest_access_log(rid)
    if log_rec:
        ex = log_rec.get("extra") or {}
        out["vram_peak_mb"] = ex.get("vram_peak_mb")
        out["vram_reserved_peak_mb"] = ex.get("vram_reserved_peak_mb")
        out["server_rtf"] = ex.get("rtf")
        out["server_latency_ms"] = log_rec.get("latency_ms")

    if ref:
        q = quality_pct(body, ref)
        out["quality"] = q
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="parakeet")
    p.add_argument("--language", default="en")
    p.add_argument("--file", action="append",
                   help="audio path; repeatable. Defaults to 4 perf-*min.mp3")
    p.add_argument("--label", default="",
                   help="experiment tag — outputs go to bench/output/<label>/")
    p.add_argument("--base-url", default="http://127.0.0.1:11500")
    args = p.parse_args()

    files = [Path(f) for f in (args.file or DEFAULT_FILES)]
    for f in files:
        if not f.exists():
            print(f"missing: {f}", file=sys.stderr)
            return 2

    out_dir = OUTPUT_ROOT / args.label if args.label else OUTPUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        h = httpx.get(f"{args.base_url}/health", timeout=15).json()
        print(f"aistack: {h}")
    except Exception as e:
        print(f"aistack /health unreachable: {e}", file=sys.stderr)
        return 2

    client = httpx.Client()
    results = []
    for f in files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"\n→ {f.name}  ({size_mb:.1f} MB)")
        # Give access log time to flush before harvest (it batches).
        time.sleep(0.1)
        res = run_one(client, args.base_url, args.model, args.language,
                      f, out_dir, args.label)
        # Log writes shortly after response; small grace period then harvest.
        time.sleep(1.5)
        if "vram_peak_mb" not in res and "error" not in res:
            log_rec = harvest_access_log(res["request_id"])
            if log_rec:
                ex = log_rec.get("extra") or {}
                res["vram_peak_mb"] = ex.get("vram_peak_mb")
                res["vram_reserved_peak_mb"] = ex.get("vram_reserved_peak_mb")
                res["server_rtf"] = ex.get("rtf")
                res["server_latency_ms"] = log_rec.get("latency_ms")
        results.append(res)
        if "error" in res:
            print(f"  ERROR: {res['error']}")
            continue
        line = (
            f"  wall={res['wall']:.1f}s  RTF={res['rtf']:.3f}  "
            f"segs={res['segments']}  words={res['words']}"
        )
        if res.get("vram_peak_mb"):
            line += (f"  vram={res['vram_peak_mb']}MB "
                     f"(rsv {res.get('vram_reserved_peak_mb')}MB)")
        print(line)
        q = res.get("quality")
        if q and q.get("recall_pct") is not None:
            print(f"  quality vs ref: recall {q['recall_pct']:.2f}%  "
                  f"precision {q['precision_pct']:.2f}%  "
                  f"({q['lcs']}/{q['ref_words']} ref words matched)")

    summary = {
        "label": args.label,
        "model": args.model,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "files": [str(f) for f in files],
        "results": results,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    print()
    print("=" * 76)
    print(f"{args.label or '(no label)'} × {args.model} × {len(results)} files")
    print("=" * 76)
    print(f"{'file':<22} {'dur':>6} {'wall':>6} {'RTF':>6} {'segs':>5} "
          f"{'vram':>6} {'recall':>7}")
    for r in results:
        if "error" in r:
            print(f"{r['file']:<22} ERROR")
            continue
        rec = (r.get('quality') or {}).get('recall_pct')
        print(f"{r['file']:<22} {r['duration_sec']:>5.0f}s "
              f"{r['wall']:>5.1f}s {r['rtf']:>6.3f} {r['segments']:>5} "
              f"{r.get('vram_peak_mb') or 0:>5}M "
              f"{(rec if rec is not None else 0):>6.2f}%")
    print(f"\nsummary → {summary_path}")
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
