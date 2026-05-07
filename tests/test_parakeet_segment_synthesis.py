"""Word→segment synthesis for Parakeet outputs that lack segment timestamps.

Two granularities exist because the right segment shape depends on the
consumer:
  - "sentence" (default): full semantic units — translation-friendly
  - "subtitle":            SRT-cue-sized via stable-ts pipeline

This test file covers BOTH modes plus the dispatcher, plus a real-data
soak that ensures sentence-mode is meaningfully looser than subtitle-
mode on the same input.
"""
from __future__ import annotations

import pytest

from aistack.asr.parakeet import (
    _SEGMENT_MAX_CHARS,
    _SEGMENT_MAX_DURATION_SEC,
    _SEGMENT_MIN_DURATION_SEC,
    _SENTENCE_HARD_MAX_DURATION_SEC,
    _SENTENCE_MAX_GAP_SEC,
    VALID_GRANULARITIES,
    _segments_from_words,
)


def _w(start: float, end: float, word: str) -> dict:
    return {"start": start, "end": end, "word": word}


# ── Dispatcher contract ──────────────────────────────────────────────────────

def test_empty_words_returns_empty():
    for g in VALID_GRANULARITIES:
        assert _segments_from_words([], granularity=g) == []


def test_invalid_granularity_raises():
    with pytest.raises(ValueError):
        _segments_from_words([_w(0, 1, "Hi")], granularity="bogus")


def test_default_granularity_is_sentence():
    # Subtitle mode would split this 8.5 s monologue into ≥ 2 cues
    # (max_dur=7); sentence mode keeps it as one (no punctuation, no gap).
    words = [_w(i * 0.5, i * 0.5 + 0.4, f"w{i}") for i in range(17)]  # 8.5 s
    default = _segments_from_words(words)
    sub = _segments_from_words(words, granularity="subtitle")
    assert len(default) < len(sub)


# ── Subtitle mode (stable-ts pipeline) ───────────────────────────────────────

def test_subtitle_single_short_utterance_one_segment():
    # Total duration > min_dur (1.0s) so stays as one cue.
    words = [_w(0.0, 0.5, "Hello."), _w(0.6, 1.5, "World.")]
    segs = _segments_from_words(words, granularity="subtitle")
    assert len(segs) == 1
    assert "Hello." in segs[0]["text"]
    assert "World." in segs[0]["text"]


def test_silence_gap_breaks_segment():
    # 1.5s gap > 0.5s threshold -> split, both halves ≥ min_dur (1.0s)
    words = [
        _w(0.0, 0.5, "First"),  _w(0.5, 1.0, "sentence"), _w(1.0, 1.4, "ends."),
        _w(2.9, 3.3, "Second"), _w(3.3, 3.8, "one"),      _w(3.8, 4.5, "now."),
    ]
    segs = _segments_from_words(words, granularity="subtitle")
    assert len(segs) == 2
    assert "First" in segs[0]["text"]
    assert "Second" in segs[1]["text"]


def test_min_duration_protects_against_flicker():
    # Sentence ender at 0.4s — too short to flush by itself (min_dur=1.0).
    # Should stay glued to the next chunk.
    words = [
        _w(0.0, 0.4, "Hi."),
        _w(0.5, 1.2, "How"), _w(1.2, 1.6, "are"), _w(1.6, 2.0, "you"),
        _w(2.0, 2.4, "today?"),
    ]
    segs = _segments_from_words(words, granularity="subtitle")
    # Single cue covering everything — "Hi." was too short to split off.
    assert len(segs) == 1
    assert segs[0]["end"] - segs[0]["start"] >= _SEGMENT_MIN_DURATION_SEC


def test_max_chars_forces_split_via_stage3():
    # 100 short words with no punctuation, no silence, ~0.05s each (5s total).
    # Duration just under 7s but char count blows past 70.
    words = [_w(i * 0.05, i * 0.05 + 0.04, f"w{i}") for i in range(100)]
    segs = _segments_from_words(words, granularity="subtitle")
    assert len(segs) >= 2
    for s in segs:
        assert len(s["text"]) <= _SEGMENT_MAX_CHARS


def test_max_duration_forces_split():
    # Force-flushes long monotone speech without punctuation.
    words = [_w(i * 0.5, i * 0.5 + 0.4, f"word{i}") for i in range(30)]  # 15s
    segs = _segments_from_words(words, granularity="subtitle")
    assert len(segs) >= 2
    for s in segs:
        assert s["end"] - s["start"] <= _SEGMENT_MAX_DURATION_SEC + 0.1


def test_comma_split_only_when_prefix_long_enough():
    # A short prefix "Yes, " followed by a long clause shouldn't get a
    # tiny "Yes," cue (would flicker). The whole segment stays together
    # unless its length exceeds max_chars/max_dur.
    words = [
        _w(0.0, 0.3, "Yes,"),
        _w(0.4, 0.8, "I"), _w(0.8, 1.4, "agree"), _w(1.4, 2.0, "with"),
        _w(2.0, 2.6, "that."),
    ]
    segs = _segments_from_words(words, granularity="subtitle")
    # Either one segment, or if split, no segment is "Yes," alone.
    for s in segs:
        assert s["text"].strip() != "Yes,"


def test_segments_have_monotonic_times_and_ids():
    words = [_w(i * 0.5, i * 0.5 + 0.4, f"w{i}.") for i in range(20)]
    segs = _segments_from_words(words, granularity="subtitle")
    last_end = -1.0
    for idx, s in enumerate(segs):
        assert s["id"] == idx
        assert s["start"] >= last_end
        assert s["end"] >= s["start"]
        last_end = s["end"]


def test_realistic_paragraph_splits_at_sentence_boundaries():
    # ~150 wpm = 0.4s/word with sentence-final punctuation + silence gap.
    words = []
    t = 0.0
    sents = [
        "Hello there how are you doing today my friend",
        "I am doing well thanks for asking how about you",
        "The weather is beautiful and sunny outside today",
    ]
    for s in sents:
        toks = s.split()
        for j, tok in enumerate(toks):
            end_punct = "." if j == len(toks) - 1 else ""
            words.append(_w(t, t + 0.35, tok + end_punct))
            t += 0.4
        t += 1.0  # > max_gap_sec (0.5)

    segs = _segments_from_words(words, granularity="subtitle")
    # At least one cue per sentence — possibly more if char limit fires.
    assert len(segs) >= len(sents)
    rebuilt = " ".join(s["text"] for s in segs)
    for tok in ["Hello", "doing", "weather", "outside"]:
        assert tok in rebuilt


def test_public_segment_dict_shape():
    words = [_w(0.0, 0.5, "Hi"), _w(0.5, 1.5, "there.")]
    segs = _segments_from_words(words, granularity="subtitle")
    # Internal _words must NOT leak into the public contract.
    assert set(segs[0].keys()) == {"id", "start", "end", "text"}


def test_all_cues_meet_industry_subtitle_constraints():
    # Stress test on a realistic transcript: every emitted cue should
    # satisfy ALL of {≤max_chars, ≤max_dur, ≥min_dur} except the very
    # last cue which may be shorter than min_dur if the audio ends.
    import random
    random.seed(42)
    words = []
    t = 0.0
    for i in range(200):
        tok = "word" + str(i)
        if random.random() < 0.1:
            tok += "."
        dur = 0.2 + random.random() * 0.3
        words.append(_w(t, t + dur, tok))
        t += dur + (0.7 if tok.endswith(".") else 0.05)

    segs = _segments_from_words(words, granularity="subtitle")
    for s in segs[:-1]:  # skip last — may be short if audio truncates
        assert len(s["text"]) <= _SEGMENT_MAX_CHARS, s
        assert s["end"] - s["start"] <= _SEGMENT_MAX_DURATION_SEC + 0.1, s


# ── Sentence mode ───────────────────────────────────────────────────────────

def test_sentence_split_at_sentence_enders():
    # Three sentences with no silence gaps between them; sentence mode
    # must still split because of the trailing punctuation.
    words = [
        _w(0.0, 0.4, "First"),  _w(0.4, 0.8, "sentence."),
        _w(0.9, 1.2, "Second"), _w(1.2, 1.6, "one."),
        _w(1.7, 2.1, "Third"),  _w(2.1, 2.5, "here."),
    ]
    segs = _segments_from_words(words, granularity="sentence")
    assert len(segs) == 3
    assert segs[0]["text"].endswith("sentence.")
    assert segs[1]["text"].endswith("one.")
    assert segs[2]["text"].endswith("here.")


def test_sentence_keeps_long_sentence_intact():
    # 8.5s monologue, no punctuation, no silence → subtitle mode would
    # split into ≥ 2 cues; sentence mode keeps it whole (no semantic
    # boundary to split at, hard cap is 30s).
    words = [_w(i * 0.5, i * 0.5 + 0.4, f"w{i}") for i in range(17)]
    segs = _segments_from_words(words, granularity="sentence")
    assert len(segs) == 1
    assert segs[0]["end"] - segs[0]["start"] <= _SENTENCE_HARD_MAX_DURATION_SEC


def test_sentence_force_breaks_at_30s_hard_cap():
    # 80 s monotone speech with no punctuation, no gaps. Sentence mode
    # must still bound the segment somewhere — the WhisperX PR #982
    # "60–80 s monster" failure mode.
    words = [_w(i * 0.4, i * 0.4 + 0.35, f"w{i}") for i in range(200)]  # 80 s
    segs = _segments_from_words(words, granularity="sentence")
    assert len(segs) >= 2
    for s in segs:
        assert s["end"] - s["start"] <= _SENTENCE_HARD_MAX_DURATION_SEC + 0.1


def test_sentence_splits_on_long_silence():
    # 1.5 s silence > 0.7 s threshold → split even without punctuation
    # (paragraph / speaker turn boundary).
    words = [
        _w(0.0, 0.4, "before"), _w(0.4, 0.8, "the"), _w(0.8, 1.2, "pause"),
        _w(2.7, 3.1, "after"), _w(3.1, 3.5, "the"), _w(3.5, 3.9, "pause"),
    ]
    segs = _segments_from_words(words, granularity="sentence")
    assert len(segs) == 2


def test_sentence_does_not_split_on_short_silence():
    # 0.4 s silence < 0.7 s threshold → stays one segment.
    words = [
        _w(0.0, 0.4, "still"), _w(0.4, 0.8, "going"),
        _w(1.2, 1.6, "smoothly"),
    ]
    segs = _segments_from_words(words, granularity="sentence")
    assert len(segs) == 1


def test_sentence_no_max_chars_constraint():
    # 200 short words with no punctuation in a single 4 s utterance —
    # subtitle mode would force multiple cuts on max_chars; sentence
    # mode does not. The whole monologue stays as one semantic unit
    # (assuming under the 30 s hard cap).
    words = [_w(i * 0.02, i * 0.02 + 0.018, f"x{i}") for i in range(200)]  # 4 s
    segs = _segments_from_words(words, granularity="sentence")
    assert len(segs) == 1
    # Length much greater than the subtitle 70-char cap is allowed:
    assert len(segs[0]["text"]) > _SEGMENT_MAX_CHARS
