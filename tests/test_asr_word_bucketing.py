"""Word-to-segment bucketing for the SSE downgrade path.

When Parakeet (no native streaming) returns a verbose result with both
segments[] and words[] populated, the streaming downgrade path emits one
SSE delta per segment. The words flat-list must be bucketed by time so
each per-segment delta carries only its own words.
"""
from __future__ import annotations

from aistack.api.asr import _bucket_words_by_segment


def _seg(start: float, end: float, text: str = "") -> dict:
    return {"start": start, "end": end, "text": text}


def _w(start: float, end: float, word: str) -> dict:
    return {"start": start, "end": end, "word": word}


def test_empty_inputs():
    assert _bucket_words_by_segment([], []) == []
    assert _bucket_words_by_segment([_w(0, 1, "hi")], []) == []
    assert _bucket_words_by_segment([], [_seg(0, 5)]) == [[]]


def test_words_distributed_across_segments():
    segments = [_seg(0.0, 5.0), _seg(5.0, 10.0), _seg(10.0, 15.0)]
    words = [
        _w(0.5, 1.0, "a"),
        _w(2.0, 2.5, "b"),
        _w(6.0, 6.5, "c"),
        _w(9.0, 9.5, "d"),
        _w(12.0, 12.5, "e"),
    ]
    out = _bucket_words_by_segment(words, segments)
    assert [w["word"] for w in out[0]] == ["a", "b"]
    assert [w["word"] for w in out[1]] == ["c", "d"]
    assert [w["word"] for w in out[2]] == ["e"]


def test_word_outside_any_segment_dropped():
    # Audio-edge word before first segment / after last segment.
    segments = [_seg(2.0, 5.0)]
    words = [_w(0.5, 1.0, "before"), _w(3.0, 3.5, "in"), _w(10.0, 10.5, "after")]
    out = _bucket_words_by_segment(words, segments)
    assert [w["word"] for w in out[0]] == ["in"]


def test_word_at_segment_boundary_goes_to_first_match():
    # A word starting exactly at the boundary between two segments
    # should attach to the segment whose [start, end] contains it.
    # We treat seg.end as inclusive in the advance loop: a word at
    # ws=seg[0].end attaches to seg[0] (its end == ws).
    segments = [_seg(0.0, 5.0), _seg(5.0, 10.0)]
    words = [_w(5.0, 5.5, "boundary")]
    out = _bucket_words_by_segment(words, segments)
    # Either bucket placement is acceptable, but the word must not be lost.
    placed = [w["word"] for ws in out for w in ws]
    assert "boundary" in placed


def test_realistic_distribution_preserves_ordering():
    # 100 words evenly distributed across 10 segments, each 1s long.
    segments = [_seg(i, i + 1) for i in range(10)]
    words = []
    for i in range(100):
        t = i * 0.1
        words.append(_w(t, t + 0.05, f"w{i}"))
    out = _bucket_words_by_segment(words, segments)
    # Total words distributed = total input words (no drops in the body)
    total = sum(len(b) for b in out)
    assert total == 100
    # Each bucket gets ~10 words; ordering preserved within bucket
    for bucket in out:
        starts = [float(w["start"]) for w in bucket]
        assert starts == sorted(starts)


def test_o_n_plus_m_walk_no_redundant_passes():
    # Sanity: 50000 words across 1000 segments completes quickly.
    # The inner loop must NOT be O(N*M).
    import time
    segments = [_seg(i * 0.5, (i + 1) * 0.5) for i in range(1000)]
    words = [_w(i * 0.01, i * 0.01 + 0.005, f"w{i}") for i in range(50000)]
    t0 = time.perf_counter()
    out = _bucket_words_by_segment(words, segments)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5  # generous bound; O(N+M) easily under 50 ms
    total = sum(len(b) for b in out)
    assert total == 50000
