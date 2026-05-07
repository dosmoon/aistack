"""SSE downgrade contract for non-streaming-capable backends (Parakeet).

Locks in: when a client sends stream=true to a model whose
supports_streaming=false, the SSE event sequence MUST be:

    1 × {"type": "warning", ...}
    N × {"type": "transcript.text.delta", ...}      where N == len(segments)
    1 × {"type": "transcript.text.done", ...}

The previous implementation collapsed the N deltas into a single
synthetic 0..duration cue containing the full transcription text. That
bug was masked while NeMo returned empty segment_ts (1 segment was the
only thing we had), and surfaced once subsampling chunking restored
real segment timestamps. This test prevents the regression.

Drives the async generator `_stream_transcribe` directly via
asyncio.run instead of going through FastAPI TestClient — the latter
deadlocks on Windows + pytest-asyncio strict mode when the generator
internally creates its disconnect-watcher task. Running the generator
directly is a closer fit anyway: we want to lock in event-shape
behavior, not the FastAPI plumbing around it (covered by other
endpoint tests).
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock

import pytest

from aistack import _gpu_lock
from aistack.api import asr as asr_api
from aistack.asr import parakeet as parakeet_module


def _fake_parakeet_result(*, segments_n: int = 5) -> dict:
    """Fake transcribe() return value with N sentence segments + words."""
    segments = []
    words = []
    text_parts = []
    for i in range(segments_n):
        seg_start = i * 2.0
        seg_end = seg_start + 1.5
        sentence = f"Sentence {i}."
        segments.append({"id": i, "start": seg_start, "end": seg_end, "text": sentence})
        text_parts.append(sentence)
        words.append({"start": seg_start + 0.1, "end": seg_start + 0.5,
                      "word": "Sentence"})
        words.append({"start": seg_start + 0.6, "end": seg_start + 1.0,
                      "word": f"{i}."})
    return {
        "language": "en",
        "duration": (segments_n - 1) * 2.0 + 1.5,
        "text": " ".join(text_parts),
        "segments": segments,
        "words": words,
    }


def _drive_downgrade(*, segments_n: int, monkeypatch) -> list[dict]:
    """Run `_stream_transcribe` (downgrade path) end-to-end and return
    the parsed SSE events as a list of dicts.

    Patches:
      - parakeet_module.transcribe → returns a fake result with N segments
      - aistack._gpu_lock.release → no-op (we acquire/release inline below)
    Sets up:
      - fake tmp_dir + audio_path so the generator's finally cleanup works
      - synthetic Request stub whose is_disconnected() returns False
    """
    fake = _fake_parakeet_result(segments_n=segments_n)
    monkeypatch.setattr(parakeet_module, "transcribe",
                        lambda *a, **kw: fake)

    # Acquire the slot so the generator's finally release() is balanced.
    _gpu_lock.try_acquire_or_503("asr")

    tmp_dir = tempfile.mkdtemp(prefix="aistack_test_downgrade_")
    audio_path = os.path.join(tmp_dir, "fake.wav")
    with open(audio_path, "wb") as fh:
        fh.write(b"fake-audio-bytes")

    # Minimal Request stand-in — generator only calls is_disconnected().
    request = MagicMock()
    async def _is_disconnected():
        return False
    request.is_disconnected = _is_disconnected

    async def collect():
        events: list[dict] = []
        async for chunk in asr_api._stream_transcribe(
            module=parakeet_module,
            audio_path=audio_path,
            language="en",
            translate=False,
            kwargs={"model_name": "nvidia/parakeet-tdt-0.6b-v3"},
            request=request,
            supports_streaming=False,
            canonical_model_id="nvidia/parakeet-tdt-0.6b-v3",
            tmp_dir=tmp_dir,
            obs_state=None,
        ):
            text = chunk.decode("utf-8")
            for line in text.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        return events

    try:
        return asyncio.run(collect())
    finally:
        # Generator's finally clause calls _gpu_lock.release() and
        # rmtree(tmp_dir), so by the time asyncio.run returns the lock
        # should already be free and tmp_dir gone. Defensive cleanup
        # in case of pytest-side error.
        if _gpu_lock.is_busy():
            try:
                _gpu_lock.release()
            except RuntimeError:
                pass
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_emits_one_delta_per_segment(monkeypatch):
    events = _drive_downgrade(segments_n=5, monkeypatch=monkeypatch)
    types = [e["type"] for e in events]

    assert types.count("warning") == 1
    assert types.count("transcript.text.done") == 1
    delta_count = types.count("transcript.text.delta")
    assert delta_count == 5, (
        f"expected 5 deltas (one per segment), got {delta_count}; "
        f"sequence: {types}"
    )
    # Order: warning first, done last.
    assert types[0] == "warning"
    assert types[-1] == "transcript.text.done"


def test_delta_segments_carry_real_timestamps(monkeypatch):
    events = _drive_downgrade(segments_n=3, monkeypatch=monkeypatch)
    deltas = [e for e in events if e["type"] == "transcript.text.delta"]

    assert len(deltas) == 3
    # Pre-fix bug: every delta would carry start=0.0, end=duration.
    # Real per-segment timestamps must vary across deltas.
    starts = [d["segment"]["start"] for d in deltas]
    ends = [d["segment"]["end"] for d in deltas]
    assert starts == [0.0, 2.0, 4.0]
    assert ends == [1.5, 3.5, 5.5]
    assert [d["delta"] for d in deltas] == [
        "Sentence 0.", "Sentence 1.", "Sentence 2.",
    ]


def test_warning_event_first_and_self_describing(monkeypatch):
    events = _drive_downgrade(segments_n=2, monkeypatch=monkeypatch)
    warning = events[0]
    assert warning["type"] == "warning"
    assert warning["code"] == "streaming_not_supported"
    assert "parakeet" in warning["model"].lower()


def test_words_bucketed_into_their_segment(monkeypatch):
    events = _drive_downgrade(segments_n=4, monkeypatch=monkeypatch)
    deltas = [e for e in events if e["type"] == "transcript.text.delta"]

    # Each delta's bucketed words must have timestamps inside that
    # segment's [start, end] — never bleeding across.
    for ev in deltas:
        seg = ev["segment"]
        for w in seg["words"]:
            assert seg["start"] <= w["start"] <= seg["end"], (
                f"word {w!r} leaked outside segment "
                f"{seg['start']}-{seg['end']}"
            )


def test_done_carries_language_and_duration(monkeypatch):
    events = _drive_downgrade(segments_n=2, monkeypatch=monkeypatch)
    done = next(e for e in events if e["type"] == "transcript.text.done")
    # 2 segments × 2.0s spacing + 1.5s segment length = 3.5 (last segment ends at 3.5)
    # but our duration formula is (segments_n - 1) * 2.0 + 1.5 = 3.5. Match it.
    assert done["language"] == "en"
    assert done["duration"] == 3.5
