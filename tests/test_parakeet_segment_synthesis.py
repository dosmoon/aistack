"""Word→segment synthesis for Parakeet outputs that lack segment timestamps.

NeMo sometimes returns word timestamps but empty segment timestamps on
long-form audio under local attention. The pre-fix fallback collapsed
the entire transcription into a single 0..duration cue, which made SRT
output unusable. _segments_from_words() rebuilds subtitle-friendly
segments from the word-level timing.
"""
from __future__ import annotations

from aistack.asr.parakeet import _segments_from_words


def _w(start: float, end: float, word: str) -> dict:
    return {"start": start, "end": end, "word": word}


def test_empty_words_returns_empty():
    assert _segments_from_words([]) == []


def test_single_short_utterance_one_segment():
    words = [_w(0.0, 0.5, "Hello."), _w(0.6, 1.0, "World")]
    segs = _segments_from_words(words)
    assert len(segs) == 1
    assert segs[0]["text"] == "Hello. World"
    assert segs[0]["start"] == 0.0
    assert abs(segs[0]["end"] - 1.0) < 1e-6


def test_silence_gap_breaks_segment():
    # 1.0s gap > 0.6s threshold -> split
    words = [
        _w(0.0, 0.4, "First"), _w(0.4, 0.8, "sentence."),
        _w(2.0, 2.4, "Second"), _w(2.4, 2.8, "one."),
    ]
    segs = _segments_from_words(words)
    assert len(segs) == 2
    assert segs[0]["text"] == "First sentence."
    assert segs[1]["text"] == "Second one."


def test_sentence_ender_breaks_only_after_min_duration():
    # Many short words ending in '.' but cumulative duration < 5s should
    # NOT split — prevents over-segmentation of fast speech.
    words = [_w(i * 0.3, i * 0.3 + 0.25, "word.") for i in range(5)]  # ~1.5s total
    segs = _segments_from_words(words)
    # Either one segment, or at most a couple — never one per word.
    assert len(segs) <= 2


def test_long_run_on_speech_force_flushes():
    # 50 words, ~10s each at 0.2s/word with no punctuation, no silence.
    # The hard_max (1.5 * 5 = 7.5s) must force splits even without
    # sentence enders or gaps.
    words = [_w(i * 0.2, i * 0.2 + 0.18, f"w{i}") for i in range(50)]
    segs = _segments_from_words(words)
    assert len(segs) >= 2
    for s in segs:
        assert s["end"] - s["start"] <= 7.6  # hard_max with float slack


def test_segments_have_monotonic_times_and_ids():
    words = [_w(i * 0.5, i * 0.5 + 0.4, f"w{i}.") for i in range(20)]
    segs = _segments_from_words(words)
    last_end = -1.0
    for idx, s in enumerate(segs):
        assert s["id"] == idx
        assert s["start"] >= last_end
        assert s["end"] >= s["start"]
        last_end = s["end"]


def test_realistic_paragraph_splits_at_sentence_boundaries():
    # Simulates words at typical English speech rate (~150 wpm = 0.4s/word)
    # with sentence-final punctuation every ~10 words and a small silence.
    words = []
    t = 0.0
    sents = [
        "Hello there how are you doing today my friend",
        "I am doing well thanks for asking how about you",
        "The weather is beautiful and sunny outside today",
    ]
    for i, s in enumerate(sents):
        toks = s.split()
        for j, tok in enumerate(toks):
            end_punct = "." if j == len(toks) - 1 else ""
            words.append(_w(t, t + 0.35, tok + end_punct))
            t += 0.4
        t += 0.8  # sentence-end silence (> 0.6 gap threshold)

    segs = _segments_from_words(words)
    # Each sentence should end up in its own segment (or close).
    assert len(segs) >= len(sents)
    # First segment should start near 0 and end before the second sentence.
    assert segs[0]["start"] < 0.5
    # Combined transcription preserves all words.
    rebuilt = " ".join(s["text"] for s in segs)
    for tok in ["Hello", "doing", "weather", "outside"]:
        assert tok in rebuilt
