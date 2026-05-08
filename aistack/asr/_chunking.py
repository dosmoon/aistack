"""Audio chunking + word-list stitching for long-audio ASR.

Used by the Parakeet provider to keep long inputs inside its short-audio
"safe" memory regime (~4 GB VRAM). The provider transcodes audio to a
single 16 kHz mono WAV, asks plan_chunks() for absolute (start, end)
windows, runs each window independently, shifts each window's
timestamps to absolute time, then stitches adjacent windows together
via stitch_words().

plan_chunks rule (one rule, no extra threshold):
    * Default window 12 min, overlap 2 min on each seam (stride 10 min).
    * If audio fits in one window, return a single (0, T).
    * Otherwise lay down windows at i*stride, length = window. The very
      last window absorbs any tail < min_last (default 5 min) — rather
      than producing a tiny isolated chunk, we extend the previous
      chunk to T. Worst case: last chunk = stride + min_last - eps
      ≈ 15 min, still inside Parakeet's safe range.

stitch rule:
    Given two adjacent chunks A and B, both with absolute timestamps
    and a known overlap window [B.start, A.end]:
    * Restrict A's words to those starting before B.start and B's
      words to those starting after A.end → those are committed
      (outside the overlap zone).
    * In the overlap zone, find the longest common word subsequence
      (LCS over normalized text). Cut A at the LCS midpoint and keep
      B from the matching midpoint onward. This guarantees the seam
      lands on a word both passes agree on, so segments don't get
      sliced mid-clause.
    * If the overlap zone has no words (silence) or LCS is empty,
      fall back to "cut at midpoint of overlap by time".
"""
from __future__ import annotations


def plan_chunks(
    total_sec: float,
    *,
    window_sec: float = 720.0,
    overlap_sec: float = 120.0,
    min_last_sec: float = 300.0,
) -> list[tuple[float, float]]:
    """Return [(start, end)] in seconds covering [0, total_sec]."""
    if total_sec <= window_sec:
        return [(0.0, total_sec)]
    stride = window_sec - overlap_sec
    chunks: list[tuple[float, float]] = []
    start = 0.0
    while start + window_sec < total_sec:
        chunks.append((start, start + window_sec))
        start += stride
    tail_len = total_sec - start
    if tail_len < min_last_sec and chunks:
        s, _ = chunks[-1]
        chunks[-1] = (s, total_sec)
    else:
        chunks.append((start, total_sec))
    return chunks


# ─── Stitching ──────────────────────────────────────────────────────────────

import re

_WORD_NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm(w: str) -> str:
    return _WORD_NORM_RE.sub("", w.lower())


def _lcs_indices(a: list[str], b: list[str]) -> list[tuple[int, int]]:
    """Return list of (i, j) index pairs forming an LCS of a and b."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    out: list[tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            out.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def stitch_words(
    a_words: list[dict],
    b_words: list[dict],
    *,
    seam_start_sec: float,
    seam_end_sec: float,
) -> list[dict]:
    """Merge word lists from two adjacent chunks into one.

    a_words / b_words: dicts with at least {start, end, word}, in absolute
                      time (already shifted by chunk offsets).
    seam_start_sec  : absolute time where chunk B began.
    seam_end_sec    : absolute time where chunk A ended.

    Words from A starting before seam_start_sec and from B starting
    after seam_end_sec are kept verbatim. The overlap zone gets a
    single seam chosen by LCS midpoint. Returns a new list (input
    not mutated).
    """
    a_pre = [w for w in a_words if w["start"] < seam_start_sec]
    a_overlap = [w for w in a_words if seam_start_sec <= w["start"] < seam_end_sec]
    b_overlap = [w for w in b_words if seam_start_sec <= w["start"] < seam_end_sec]
    b_post = [w for w in b_words if w["start"] >= seam_end_sec]

    # Find LCS over normalized text in overlap zone.
    a_norm = [_norm(w["word"]) for w in a_overlap]
    b_norm = [_norm(w["word"]) for w in b_overlap]
    lcs = _lcs_indices(a_norm, b_norm)

    if lcs:
        # Cut at LCS midpoint — the seam word itself goes to A, B picks
        # up immediately after.
        mid = len(lcs) // 2
        ai, bj = lcs[mid]
        return a_pre + a_overlap[: ai + 1] + b_overlap[bj + 1 :] + b_post

    # No LCS (silence or wildly-divergent transcripts). Fallback: cut by
    # time at overlap midpoint.
    mid_t = (seam_start_sec + seam_end_sec) / 2.0
    a_keep = [w for w in a_overlap if w["start"] < mid_t]
    b_keep = [w for w in b_overlap if w["start"] >= mid_t]
    return a_pre + a_keep + b_keep + b_post


def shift_words(words: list[dict], offset_sec: float) -> list[dict]:
    """Return a new word list with start/end shifted by offset_sec."""
    out = []
    for w in words:
        nw = dict(w)
        nw["start"] = w["start"] + offset_sec
        nw["end"] = w["end"] + offset_sec
        out.append(nw)
    return out
