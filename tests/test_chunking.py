"""Unit tests for aistack.asr._chunking."""
from __future__ import annotations

from aistack.asr._chunking import plan_chunks, stitch_words, shift_words


# ─── plan_chunks ────────────────────────────────────────────────────────────

# Existing edge-case tests pass overlap_sec=60 explicitly so they don't
# depend on the production default (which is 120 — see the
# test_default_overlap_is_120 test below).

def test_plan_short_audio_returns_single_block():
    # 12.5 min — below window+min_last threshold; tail merged into single chunk
    out = plan_chunks(12.5 * 60, overlap_sec=60)
    assert out == [(0.0, 750.0)]


def test_plan_exactly_window_single_block():
    out = plan_chunks(720.0, overlap_sec=60)
    assert out == [(0.0, 720.0)]


def test_plan_15min_single_block():
    # tail = 4min < min_last (5min) → merged
    out = plan_chunks(15 * 60, overlap_sec=60)
    assert out == [(0.0, 900.0)]


def test_plan_16min_two_chunks():
    # tail exactly 5min → kept
    out = plan_chunks(16 * 60, overlap_sec=60)
    assert len(out) == 2
    assert out[0] == (0.0, 720.0)
    assert out[1] == (660.0, 960.0)
    assert out[1][1] - out[1][0] == 300.0  # last chunk 5min


def test_plan_20min_two_chunks():
    out = plan_chunks(20 * 60, overlap_sec=60)
    assert out == [(0.0, 720.0), (660.0, 1200.0)]


def test_plan_23p5min_merge_tail():
    # tail = 1.5min < 5min → merged into prev. Last chunk 12.5min.
    out = plan_chunks(23.5 * 60, overlap_sec=60)
    assert len(out) == 2
    assert out[0] == (0.0, 720.0)
    assert out[1] == (660.0, 1410.0)


def test_plan_25min_merge_tail_to_14min():
    # tail = 3min < 5min → merged. Last chunk = stride + tail = 14min.
    out = plan_chunks(25 * 60, overlap_sec=60)
    assert len(out) == 2
    assert out[1] == (660.0, 1500.0)
    assert out[1][1] - out[1][0] == 840.0


def test_plan_27min_three_chunks_with_5min_tail():
    out = plan_chunks(27 * 60, overlap_sec=60)
    assert len(out) == 3
    assert out == [(0.0, 720.0), (660.0, 1380.0), (1320.0, 1620.0)]


def test_plan_50min_coverage_and_overlaps():
    out = plan_chunks(50 * 60, overlap_sec=60)
    # Sanity: covers full audio and consecutive chunks overlap.
    assert out[0][0] == 0.0
    assert out[-1][1] == 50 * 60
    for prev, nxt in zip(out, out[1:]):
        assert nxt[0] < prev[1], "consecutive chunks must overlap"


def test_plan_97min_coverage_and_overlaps():
    out = plan_chunks(97 * 60, overlap_sec=60)
    assert out[0][0] == 0.0
    assert out[-1][1] == 97 * 60
    for prev, nxt in zip(out, out[1:]):
        overlap = prev[1] - nxt[0]
        assert overlap >= 60.0 - 1e-6, f"overlap shrunk to {overlap}"


def test_plan_last_chunk_never_below_min_last():
    # Sweep durations 16..60 min and verify last chunk >= 5 min.
    for t_min in range(16, 61):
        out = plan_chunks(t_min * 60, overlap_sec=60)
        last_len = out[-1][1] - out[-1][0]
        assert last_len >= 300.0 - 1e-6, (
            f"T={t_min}min: last chunk only {last_len}s"
        )


def test_default_overlap_is_120():
    # Production default — 50min audio → 5 chunks with 2-min overlap.
    out = plan_chunks(50 * 60)
    assert out[0] == (0.0, 720.0)
    assert out[1] == (600.0, 1320.0)  # stride 600 = window 720 - overlap 120
    assert out[-1][1] == 50 * 60
    for prev, nxt in zip(out, out[1:]):
        assert prev[1] - nxt[0] >= 120.0 - 1e-6


def test_default_25min_three_balanced_chunks():
    # 25min @ overlap=120 produces 3 chunks instead of 2; last chunk
    # 5min instead of 14min — the win behind making 120 the default.
    out = plan_chunks(25 * 60)
    assert len(out) == 3
    last_len = out[-1][1] - out[-1][0]
    assert 290.0 < last_len < 310.0, f"expected ~5min last chunk, got {last_len}s"


# ─── shift_words ────────────────────────────────────────────────────────────

def test_shift_words_preserves_input():
    words = [{"start": 1.0, "end": 2.0, "word": "hi"}]
    out = shift_words(words, 100.0)
    assert words[0]["start"] == 1.0  # original untouched
    assert out[0]["start"] == 101.0
    assert out[0]["end"] == 102.0
    assert out[0]["word"] == "hi"


# ─── stitch_words ───────────────────────────────────────────────────────────

def _w(start, end, word):
    return {"start": float(start), "end": float(end), "word": word}


def test_stitch_clean_overlap_lcs_cuts_at_midpoint():
    # Both chunks heard "alpha bravo charlie delta echo" in [10, 20] zone.
    a = [_w(5, 6, "begin"),
         _w(11, 12, "alpha"), _w(13, 14, "bravo"), _w(15, 16, "charlie"),
         _w(17, 18, "delta"), _w(19, 20, "echo")]
    b = [_w(11, 12, "alpha"), _w(13, 14, "bravo"), _w(15, 16, "charlie"),
         _w(17, 18, "delta"), _w(19, 20, "echo"),
         _w(25, 26, "end")]
    merged = stitch_words(a, b, seam_start_sec=10.0, seam_end_sec=20.0)
    texts = [w["word"] for w in merged]
    assert texts[0] == "begin"
    assert texts[-1] == "end"
    # No duplicates of any LCS word.
    assert texts.count("charlie") == 1
    assert texts.count("alpha") == 1
    assert texts.count("echo") == 1


def test_stitch_disjoint_overlap_falls_back_to_time_split():
    # Overlap region: A heard "x y", B heard "p q". No LCS.
    a = [_w(5, 6, "before"), _w(11, 12, "x"), _w(13, 14, "y")]
    b = [_w(11, 12, "p"), _w(13, 14, "q"), _w(25, 26, "after")]
    merged = stitch_words(a, b, seam_start_sec=10.0, seam_end_sec=20.0)
    # Time-midpoint split: a keeps words < 15s, b keeps >= 15s.
    # A overlap words are at 11 and 13 (both < 15) → kept.
    # B overlap words at 11 and 13 (both < 15) → dropped.
    texts = [w["word"] for w in merged]
    assert texts == ["before", "x", "y", "after"]


def test_stitch_no_overlap_words_pass_through():
    a = [_w(1, 2, "a"), _w(3, 4, "b")]
    b = [_w(15, 16, "c"), _w(17, 18, "d")]
    merged = stitch_words(a, b, seam_start_sec=10.0, seam_end_sec=14.0)
    texts = [w["word"] for w in merged]
    assert texts == ["a", "b", "c", "d"]


def test_stitch_normalization_handles_punctuation_case():
    # A's overlap: "Alpha," "Bravo."   B's overlap: "alpha" "bravo"
    # Should still align via LCS.
    a = [_w(11, 12, "Alpha,"), _w(13, 14, "Bravo."),
         _w(15, 16, "extra-from-A")]
    b = [_w(11, 12, "alpha"), _w(13, 14, "bravo"),
         _w(15, 16, "Bridge"), _w(20, 21, "after")]
    merged = stitch_words(a, b, seam_start_sec=10.0, seam_end_sec=18.0)
    texts = [w["word"] for w in merged]
    # LCS = [alpha, bravo], mid index 1 → seam word = bravo (kept on A side).
    # B picks up after bravo → "Bridge", "after"
    assert texts == ["Alpha,", "Bravo.", "Bridge", "after"]
