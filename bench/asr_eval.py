"""ASR evaluation against LibriSpeech dev-clean.

按 ASR 行业通行方法做评测：拿 LibriSpeech（CC-BY-4.0，公开领域朗读类英文）跑一遍，
报告 WER（质量）+ RTF（性能）+ HTTP 异常计数（aistack 自身正确性）。

使用：
    # 启动 aistack
    dev.bat

    # 跑评测（默认前 10 条样本）
    python -m bench.asr_eval --model whisper-small

    # 全量（dev-clean 共 2703 条 ≈ 5.4 小时）
    python -m bench.asr_eval --model whisper-small --limit 0

    # 输出 JSON 归档
    python -m bench.asr_eval --model whisper-small --json results.json

数据自动下载到 ~/.cache/aistack-bench/，340 MB 一次性，后续复用。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import httpx

CACHE = Path.home() / ".cache" / "aistack-bench"
DATASETS = {
    "librispeech-dev-clean": {
        "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "size_mb": 337,
    },
}


# ─── data plumbing ──────────────────────────────────────────────────────────

def _download_with_progress(url: str, dest: Path) -> None:
    print(f"  → {url}")
    last_pct = -1
    def hook(blocks: int, block_size: int, total: int) -> None:
        nonlocal last_pct
        downloaded = blocks * block_size
        pct = int(downloaded * 100 / total) if total > 0 else 0
        if pct != last_pct and pct % 5 == 0:
            mb = downloaded / 1024 / 1024
            total_mb = total / 1024 / 1024
            sys.stdout.write(f"\r    {pct:3d}%  ({mb:6.1f} / {total_mb:.0f} MB)")
            sys.stdout.flush()
            last_pct = pct
    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print()


def fetch_librispeech_dev_clean() -> Path:
    cache_dir = CACHE / "librispeech-dev-clean"
    raw_dir = cache_dir / "raw"
    if raw_dir.exists():
        return raw_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "dev-clean.tar.gz"
    if not archive.exists():
        print(f"downloading LibriSpeech dev-clean (~337 MB)...")
        _download_with_progress(DATASETS["librispeech-dev-clean"]["url"], archive)
    print(f"extracting...")
    with tarfile.open(archive) as tf:
        tf.extractall(raw_dir)
    return raw_dir


def index_librispeech(raw_dir: Path) -> list[dict]:
    """LibriSpeech format: speaker/chapter/<utt>.flac + speaker-chapter.trans.txt."""
    clips: list[dict] = []
    base = raw_dir / "LibriSpeech" / "dev-clean"
    if not base.exists():
        raise RuntimeError(f"unexpected layout: {base} not found")
    for trans in base.rglob("*.trans.txt"):
        for line in trans.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            utt_id, _, text = line.partition(" ")
            audio = trans.parent / f"{utt_id}.flac"
            if audio.exists():
                clips.append({
                    "id": utt_id,
                    "audio": audio,
                    "ground_truth": text,
                })
    clips.sort(key=lambda c: c["id"])
    return clips


# ─── WER (Levenshtein over words) ───────────────────────────────────────────

def _normalize(text: str) -> list[str]:
    text = text.upper()
    text = re.sub(r"[^A-Z' ]+", " ", text)
    return text.split()


def wer(reference: str, hypothesis: str) -> float:
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    # classic edit distance dp
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    return dp[n][m] / n


# ─── eval loop ──────────────────────────────────────────────────────────────

def evaluate_one(client: httpx.Client, base_url: str, model: str,
                 language: str, clip: dict) -> dict:
    rid = f"bench-{clip['id']}"
    t0 = time.perf_counter()
    try:
        with open(clip["audio"], "rb") as fh:
            r = client.post(
                f"{base_url}/v1/audio/transcriptions",
                files={"file": (clip["audio"].name, fh, "audio/flac")},
                data={"model": model, "language": language,
                      "response_format": "json"},
                headers={"X-Request-ID": rid},
                timeout=120,
            )
    except Exception as e:
        return {"id": clip["id"], "error": f"{type(e).__name__}: {e}"}

    wall = time.perf_counter() - t0
    if r.status_code != 200:
        return {"id": clip["id"], "error": f"HTTP {r.status_code}",
                "body": r.text[:200], "wall": wall}

    try:
        body = r.json()
        pred = body.get("text", "")
    except Exception as e:
        return {"id": clip["id"], "error": f"json parse: {e}", "wall": wall}

    return {
        "id": clip["id"],
        "wall": wall,
        "ground_truth": clip["ground_truth"],
        "predicted": pred,
        "wer": wer(clip["ground_truth"], pred),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="whisper-small",
                   help="aistack model id (default: whisper-small)")
    p.add_argument("--language", default="en")
    p.add_argument("--limit", type=int, default=10,
                   help="number of clips (0 = all 2703)")
    p.add_argument("--base-url", default="http://127.0.0.1:11500")
    p.add_argument("--json", help="write detailed results to this JSON file")
    args = p.parse_args()

    raw_dir = fetch_librispeech_dev_clean()
    clips = index_librispeech(raw_dir)
    print(f"indexed {len(clips)} clips from LibriSpeech dev-clean")

    if args.limit > 0:
        clips = clips[: args.limit]
        print(f"limit={args.limit} → evaluating first {len(clips)}")

    # quick health check
    try:
        h = httpx.get(f"{args.base_url}/health", timeout=3).json()
        print(f"aistack: {h}")
    except Exception as e:
        print(f"aistack /health unreachable: {e}", file=sys.stderr)
        return 2

    client = httpx.Client()
    results: list[dict] = []
    t_start = time.perf_counter()
    for i, clip in enumerate(clips, 1):
        res = evaluate_one(client, args.base_url, args.model, args.language, clip)
        results.append(res)
        if "error" in res:
            print(f"  [{i:3d}/{len(clips)}] {res['id']:24s} ERROR: {res['error']}")
        else:
            print(f"  [{i:3d}/{len(clips)}] {res['id']:24s} "
                  f"wall={res['wall']:5.2f}s WER={res['wer']*100:5.2f}%")
    total_wall = time.perf_counter() - t_start

    # ── summary ─────────────────────────────────────────────────────────
    ok = [r for r in results if "wer" in r]
    errors = [r for r in results if "error" in r]
    walls = [r["wall"] for r in ok]
    wers = [r["wer"] for r in ok]
    audio_sec = sum(0.0 for _ in ok)  # WER samples; LibriSpeech utterances ~7s avg
    # We don't have per-clip duration here (would need ffprobe); RTF estimated from
    # known dev-clean total ≈ 19438 s spread over 2703 clips ≈ 7.2 s avg.
    avg_clip_dur = 7.2

    print()
    print("=" * 60)
    print(f"LibriSpeech dev-clean × {args.model} × {len(results)} clips")
    print("=" * 60)
    print(f"correctness:  {len(ok)} ok  /  {len(errors)} errored")
    if errors:
        print("  errored ids:")
        for r in errors[:10]:
            print(f"    {r['id']:24s} {r['error']}")
    if ok:
        print(f"quality:      WER avg={sum(wers)/len(wers)*100:.2f}%   "
              f"median={sorted(wers)[len(wers)//2]*100:.2f}%   "
              f"max={max(wers)*100:.2f}%")
        worst = sorted(ok, key=lambda r: r["wer"], reverse=True)[:3]
        print("  worst 3:")
        for r in worst:
            print(f"    {r['id']:24s} WER={r['wer']*100:.1f}%")
            print(f"      GT:   {r['ground_truth'][:90]}")
            print(f"      pred: {r['predicted'][:90]}")
        print(f"performance:  total wall={total_wall:.1f}s   "
              f"avg per clip={sum(walls)/len(walls):.2f}s   "
              f"approx RTF={(sum(walls)/len(walls))/avg_clip_dur:.3f}")

    if args.json:
        Path(args.json).write_text(
            json.dumps({
                "model": args.model,
                "dataset": "librispeech-dev-clean",
                "n_clips": len(results),
                "n_errors": len(errors),
                "wer_avg": sum(wers) / len(wers) if wers else None,
                "total_wall_s": total_wall,
                "results": results,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nresults JSON → {args.json}")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
