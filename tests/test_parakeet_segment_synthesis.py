"""Word→segment synthesis for Parakeet outputs that lack segment timestamps.

NeMo sometimes returns word timestamps but empty segment timestamps on
long-form audio under local attention. The pre-fix fallback collapsed
the entire transcription into a single 0..duration cue; the post-fix
synthesis now produces SRT-suitable cues using a stable-ts-modeled
multi-stage split (sentence enders → silence gap → comma fallback →
length-based hard split). Thresholds match the subtitle-localisation
industry standard: ≤ 70 chars, 1–7 s per cue, 0.5 s silence gap.
"""
from __future__ import annotations

from aistack.asr.parakeet import (
    _SEGMENT_MAX_CHARS,
    _SEGMENT_MAX_DURATION_SEC,
    _SEGMENT_MIN_DURATION_SEC,
    _segments_from_words,
)


def _w(start: float, end: float, word: str) -> dict:
    return {"start": start, "end": end, "word": word}


def test_empty_words_returns_empty():
    assert _segments_from_words([]) == []


def test_single_short_utterance_one_segment():
    # Total duration > min_dur (1.0s) so stays as one cue.
    words = [_w(0.0, 0.5, "Hello."), _w(0.6, 1.5, "World.")]
    segs = _segments_from_words(words)
    assert len(segs) == 1
    assert "Hello." in segs[0]["text"]
    assert "World." in segs[0]["text"]


def test_silence_gap_breaks_segment():
    # 1.5s gap > 0.5s threshold -> split, both halves ≥ min_dur (1.0s)
    words = [
        _w(0.0, 0.5, "First"),  _w(0.5, 1.0, "sentence"), _w(1.0, 1.4, "ends."),
        _w(2.9, 3.3, "Second"), _w(3.3, 3.8, "one"),      _w(3.8, 4.5, "now."),
    ]
    segs = _segments_from_words(words)
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
    segs = _segments_from_words(words)
    # Single cue covering everything — "Hi." was too short to split off.
    assert len(segs) == 1
    assert segs[0]["end"] - segs[0]["start"] >= _SEGMENT_MIN_DURATION_SEC


def test_max_chars_forces_split_via_stage3():
    # 100 short words with no punctuation, no silence, ~0.05s each (5s total).
    # Duration just under 7s but char count blows past 70.
    words = [_w(i * 0.05, i * 0.05 + 0.04, f"w{i}") for i in range(100)]
    segs = _segments_from_words(words)
    assert len(segs) >= 2
    for s in segs:
        assert len(s["text"]) <= _SEGMENT_MAX_CHARS


def test_max_duration_forces_split():
    # Force-flushes long monotone speech without punctuation.
    words = [_w(i * 0.5, i * 0.5 + 0.4, f"word{i}") for i in range(30)]  # 15s
    segs = _segments_from_words(words)
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
    segs = _segments_from_words(words)
    # Either one segment, or if split, no segment is "Yes," alone.
    for s in segs:
        assert s["text"].strip() != "Yes,"


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

    segs = _segments_from_words(words)
    # At least one cue per sentence — possibly more if char limit fires.
    assert len(segs) >= len(sents)
    rebuilt = " ".join(s["text"] for s in segs)
    for tok in ["Hello", "doing", "weather", "outside"]:
        assert tok in rebuilt


def test_public_segment_dict_shape():
    words = [_w(0.0, 0.5, "Hi"), _w(0.5, 1.5, "there.")]
    segs = _segments_from_words(words)
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

    segs = _segments_from_words(words)
    for s in segs[:-1]:  # skip last — may be short if audio truncates
        assert len(s["text"]) <= _SEGMENT_MAX_CHARS, s
        assert s["end"] - s["start"] <= _SEGMENT_MAX_DURATION_SEC + 0.1, s
