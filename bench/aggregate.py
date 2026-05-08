"""Print a comparison table across labeled perf_eval runs.

Reads bench/output/<label>/summary.json for every <label> subdir,
groups results by audio file, and prints one row per (file, metric) ×
column per label. Metric rows: wall, RTF, vram_peak_mb, recall_pct.

Usage:
    python -m bench.aggregate
    python -m bench.aggregate --metric wall  # single metric only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUTPUT_ROOT = Path("bench/output")


def load_summaries() -> dict[str, dict]:
    out = {}
    for d in sorted(OUTPUT_ROOT.iterdir()) if OUTPUT_ROOT.exists() else []:
        if not d.is_dir():
            continue
        sp = d / "summary.json"
        if not sp.exists():
            continue
        try:
            out[d.name] = json.loads(sp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  skip {sp}: {e}")
    return out


def fmt_cell(metric: str, val) -> str:
    if val is None:
        return "  -- "
    if metric == "wall":
        return f"{val:5.1f}s"
    if metric == "rtf":
        return f"{val:6.3f}"
    if metric == "vram_peak_mb":
        return f"{int(val):>5}M"
    if metric == "recall_pct":
        return f"{val:5.2f}%"
    return str(val)


def cell_from(result: dict, metric: str):
    if metric == "wall":
        return result.get("wall")
    if metric == "rtf":
        return result.get("rtf")
    if metric == "vram_peak_mb":
        return result.get("vram_peak_mb")
    if metric == "recall_pct":
        q = result.get("quality") or {}
        return q.get("recall_pct")
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--metric", choices=["wall", "rtf", "vram_peak_mb",
                                        "recall_pct", "all"],
                   default="all")
    args = p.parse_args()

    summaries = load_summaries()
    if not summaries:
        print(f"no labeled runs in {OUTPUT_ROOT}/")
        return 1

    labels = sorted(summaries.keys())
    # Collect all distinct files seen across labels.
    files: list[str] = []
    for s in summaries.values():
        for r in s.get("results", []):
            if r.get("file") not in files:
                files.append(r["file"])

    metrics = [args.metric] if args.metric != "all" else \
        ["wall", "rtf", "vram_peak_mb", "recall_pct"]

    col_w = max(8, max((len(l) for l in labels), default=8))
    print()
    for metric in metrics:
        print(f"── {metric} " + "─" * 60)
        head = f"{'file':<22} " + " ".join(f"{l:>{col_w}}" for l in labels)
        print(head)
        for f in files:
            row = f"{f:<22} "
            for label in labels:
                results = summaries[label].get("results", [])
                match = next((r for r in results if r.get("file") == f), None)
                cell = cell_from(match, metric) if match else None
                row += f"{fmt_cell(metric, cell):>{col_w}} "
            print(row)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
