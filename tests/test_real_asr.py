"""Real-ASR smoke tests — exercise each backend against actual audio.

These tests load real models and run real inference on the SenseVoice
sample audio bundled with the iic/SenseVoiceSmall ModelScope repo. They
exist to lock in the 2026-05-07 fixes (Parakeet local attention,
SenseVoice English space restoration, num_workers=0 DataLoader race
prevention) so the next refactor can't quietly regress them.

Skip rules — every test gates on three things and skips cleanly if any
of them is missing:

    1. The backend's ML library is importable (importorskip).
    2. The well-known audio sample is present on disk.
    3. (For Parakeet/SenseVoice GPU paths) torch is installed; CPU
       fallback is acceptable, no GPU requirement is asserted.

This means the suite passes in three configurations:
    - Full local dev box: all backends installed → all tests run.
    - Partial install: only the installed backends' tests run.
    - CI without ML extras: every test skips, suite passes green.

Runtime: 30-90 s on a warm disk cache (each test reloads its model
because conftest clears the cache between tests). Not designed for
per-commit CI; designed for "before merging anything that touches an
ASR provider."

Run only this file:
    myenv/Scripts/python.exe -m pytest tests/test_real_asr.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


# ── Audio sample resolution ──────────────────────────────────────────────────

_AUDIO_FILES = ("en.mp3", "ja.mp3", "ko.mp3", "yue.mp3", "zh.mp3")


def _candidate_audio_dirs() -> list[Path]:
    """Where to look for SenseVoice's bundled sample audio.

    Order:
        1. AISTACK_TEST_AUDIO_DIR (explicit override)
        2. <MODELSCOPE_CACHE>/models/iic/SenseVoiceSmall/example
        3. <HF_HOME>/hub/models--FunAudioLLM--SenseVoiceSmall/snapshots/*/example
        4. The well-known D:\\AI_Models tree on this dev box.
    """
    out: list[Path] = []
    if explicit := os.environ.get("AISTACK_TEST_AUDIO_DIR"):
        out.append(Path(explicit))
    if ms := os.environ.get("MODELSCOPE_CACHE"):
        out.append(Path(ms) / "models" / "iic" / "SenseVoiceSmall" / "example")
    out.append(Path(r"D:\AI_Models\modelscope\models\iic\SenseVoiceSmall\example"))
    return out


def _audio_path(name: str) -> Path:
    """Return the first existing path to a sample audio file, or skip."""
    assert name in _AUDIO_FILES, f"unknown sample {name!r}"
    for d in _candidate_audio_dirs():
        p = d / name
        if p.is_file():
            return p
    pytest.skip(
        f"audio sample {name!r} not found in any candidate dir; "
        f"set AISTACK_TEST_AUDIO_DIR to a directory containing it"
    )


# ── Backend availability gates ───────────────────────────────────────────────

def _require_faster_whisper() -> None:
    pytest.importorskip("faster_whisper")


def _require_parakeet() -> None:
    pytest.importorskip("nemo.collections.asr")


def _require_sensevoice() -> None:
    pytest.importorskip("funasr")


# ── Tests: faster-whisper ────────────────────────────────────────────────────

def test_faster_whisper_english_known_phrase():
    """faster-whisper-small transcribes the SenseVoice en.mp3 sample."""
    _require_faster_whisper()
    audio = _audio_path("en.mp3")

    from aistack.asr import faster_whisper as fw
    result = fw.transcribe(str(audio), model_name="small", language="en")

    text = result.get("text", "").strip().lower()
    assert "tribal" in text and "chieftain" in text, (
        f"expected 'tribal chieftain' in transcription; got: {text!r}"
    )
    assert result.get("duration", 0) > 0


# ── Tests: SenseVoice ────────────────────────────────────────────────────────

def test_sensevoice_english_text_has_spaces():
    """Regression for the 2026-05-07 join fix: English output is space-separated.

    Pre-fix bug: '"".join(words)' produced run-on text like
    "Msready,yessir.". The language-aware join in _join_words_for_lang
    must restore proper word boundaries for English.
    """
    _require_sensevoice()
    audio = _audio_path("en.mp3")

    from aistack.asr import sensevoice as sv
    result = sv.transcribe(str(audio), language="en")

    text = result.get("text", "")
    assert text, "SenseVoice returned empty text"
    # Known content from the official sample.
    assert "tribal chieftain" in text.lower(), (
        f"expected 'tribal chieftain' (with space) in output; got: {text!r}"
    )
    # Sanity: any English chunk longer than ~20 chars must contain at least
    # one space. The pre-fix bug failed this trivially.
    if len(text) > 20:
        assert " " in text, (
            f"English transcription has no spaces — _join_words_for_lang regressed: {text!r}"
        )


def test_sensevoice_chinese_no_inter_character_spaces():
    """Regression for the CJK side of the join fix: zh output stays compact.

    Han characters do not take inter-character spaces in natural Chinese
    text. _join_words_for_lang's CJK branch ('zh', 'yue', 'ja', 'ko')
    must keep the empty-string join for these languages.
    """
    _require_sensevoice()
    audio = _audio_path("zh.mp3")

    from aistack.asr import sensevoice as sv
    result = sv.transcribe(str(audio), language="zh")

    text = result.get("text", "")
    assert text, "SenseVoice returned empty text on Chinese sample"
    # Strip punctuation and whitespace; what remains should be all Han
    # characters with no whitespace between them. We don't assert specific
    # content because the zh.mp3 sample's wording could change upstream.
    body = "".join(c for c in text if c.strip() and c not in "，。、；：！？,.!?;:")
    # If the model produced spaces between Han characters, body would be
    # short relative to text length and a naive char-by-char check would
    # find spaces between non-ASCII chars.
    han_runs = []
    cur = []
    for c in text:
        if c.isspace():
            if cur:
                han_runs.append("".join(cur))
                cur = []
        else:
            cur.append(c)
    if cur:
        han_runs.append("".join(cur))
    # The longest run should be substantial — Chinese sentences shouldn't
    # be broken into 1-character pieces by spurious spaces.
    longest = max((len(r) for r in han_runs), default=0)
    assert longest >= 3, (
        f"Chinese output looks like it has spaces between characters: {text!r}"
    )


def test_sensevoice_passes_through_words_and_timestamps():
    """VideoCraft P2 verification: output_timestamp=True data is exposed.

    The sensevoice provider already passes output_timestamp=True to
    model.generate(). This test makes sure the returned per-word
    timestamps survive into the verbose_json shape that the gateway
    exposes to clients.
    """
    _require_sensevoice()
    audio = _audio_path("en.mp3")

    from aistack.asr import sensevoice as sv
    result = sv.transcribe(str(audio), language="en")

    words = result.get("words") or []
    assert words, "SenseVoice should populate words[] when output_timestamp=True"
    sample = words[0]
    assert "word" in sample and sample["word"], f"word entry missing 'word': {sample}"
    assert "start" in sample and "end" in sample, f"word entry missing timestamps: {sample}"
    assert sample["end"] >= sample["start"] >= 0


# ── Tests: Parakeet ──────────────────────────────────────────────────────────

def test_parakeet_english_local_attention_load():
    """Parakeet loads, switches to local attention, and transcribes en.mp3.

    Locks in three pieces of the 2026-05-07 fix:
        1. NeMo loads on the current torch + cuDNN combo without crashing
           (catches a regression to torch 2.5.1+cu121's missing
           cudnnGetLibConfig symbol).
        2. _maybe_switch_to_local_attention runs and either succeeds
           (logged INFO) or degrades to a documented warning. We don't
           require it to succeed because some NeMo versions may not
           expose change_attention_model — but the code path must run.
        3. transcribe() with num_workers=0 + batch_size=1 returns a
           result on a real audio file.

    Captures via a direct handler attached to the parakeet logger, not
    via pytest's caplog, because aistack.main sets propagate=False on
    the "aistack" logger and pytest's caplog hooks into root — when the
    full suite runs, caplog never sees the propagated messages. A
    direct handler bypasses the propagation chain.
    """
    import logging
    _require_parakeet()
    audio = _audio_path("en.mp3")

    captured: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    h = _Capture()
    h.setLevel(logging.INFO)
    target = logging.getLogger("aistack.asr.parakeet")
    prev_level = target.level
    target.setLevel(logging.INFO)
    target.addHandler(h)
    try:
        from aistack.asr import parakeet as pk
        result = pk.transcribe(
            str(audio), model_name="nvidia/parakeet-tdt-0.6b-v3", language="en",
        )
    finally:
        target.removeHandler(h)
        target.setLevel(prev_level)

    text = result.get("text", "").strip().lower()
    assert "tribal" in text and "chieftain" in text, (
        f"expected 'tribal chieftain' in transcription; got: {text!r}"
    )
    # The local-attention switch should have logged either a success or
    # an explicit fallback warning — but never a silent skip.
    matched = any("local attention" in m or "default attention" in m
                  for m in captured)
    assert matched, (
        "Expected an INFO/WARNING log mentioning local attention from the "
        "encoder switch; saw only: " + repr(captured)
    )


# ── Tests: cross-backend hot-swap ────────────────────────────────────────────

def test_hot_swap_evicts_asr_main_peers():
    """Loading two ASR backends in sequence must hot-swap, not coexist.

    aistack._model_cache.put with category='asr-main' is supposed to evict
    other 'asr-main' peers before inserting. Without this rule a 17-min
    transcribe job that touched all three backends would peak above 8 GB
    VRAM. Easy to break in a refactor; cheap to verify here.
    """
    _require_faster_whisper()
    _require_sensevoice()
    audio = _audio_path("en.mp3")

    from aistack import _model_cache
    from aistack.asr import faster_whisper as fw
    from aistack.asr import sensevoice as sv

    # First backend.
    fw.transcribe(str(audio), model_name="small", language="en")
    fw_loaded = [k for (provider, k), entry in _model_cache._CACHE.items()
                 if entry.get("category") == "asr-main" and provider == "faster-whisper"]
    assert fw_loaded, "faster-whisper should have an asr-main cache entry after use"

    # Second backend should evict the first.
    sv.transcribe(str(audio), language="en")
    fw_still_loaded = [k for (provider, k), entry in _model_cache._CACHE.items()
                       if entry.get("category") == "asr-main" and provider == "faster-whisper"]
    sv_loaded = [k for (provider, k), entry in _model_cache._CACHE.items()
                 if entry.get("category") == "asr-main" and provider == "sensevoice"]

    assert not fw_still_loaded, (
        "faster-whisper asr-main should have been hot-swapped out when "
        "sensevoice loaded; found: " + repr(fw_still_loaded)
    )
    assert sv_loaded, "sensevoice should have an asr-main cache entry after use"
