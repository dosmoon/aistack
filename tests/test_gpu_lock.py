"""GPU lock smoke tests.

Verifies the load-bearing claim of the gateway: at most one capability
holds the GPU slot at a time, regardless of which endpoint requested it.
A second acquirer (whether ASR / LLM / TTS) gets HTTP 503 + Retry-After.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from aistack import _gpu_lock


def test_busy_or_503_releases_on_normal_exit():
    with _gpu_lock.busy_or_503("asr"):
        assert _gpu_lock.is_busy()
        assert _gpu_lock.current_holder() == "asr"
    assert not _gpu_lock.is_busy()
    assert _gpu_lock.current_holder() is None


def test_busy_or_503_releases_on_exception():
    with pytest.raises(ValueError):
        with _gpu_lock.busy_or_503("asr"):
            raise ValueError("boom")
    assert not _gpu_lock.is_busy()


def test_concurrent_busy_or_503_rejects_second():
    with _gpu_lock.busy_or_503("asr"):
        with pytest.raises(HTTPException) as ei:
            with _gpu_lock.busy_or_503("asr"):
                pytest.fail("second acquire should have raised")
        assert ei.value.status_code == 503
        assert ei.value.headers.get("Retry-After") == "5"


def test_try_acquire_release_pair():
    _gpu_lock.try_acquire_or_503("llm")
    assert _gpu_lock.is_busy()
    assert _gpu_lock.current_holder() == "llm"
    _gpu_lock.release()
    assert not _gpu_lock.is_busy()
    assert _gpu_lock.current_holder() is None


def test_lock_is_shared_across_capabilities():
    """ASR holding the slot must block LLM, and vice versa.

    This is the architectural promise of the unified lock. If this test
    fails, the gateway claim of GPU scheduling is broken — concurrent
    ASR + LLM on 8 GB cards would OOM.
    """
    # ASR style acquire (sync context manager).
    with _gpu_lock.busy_or_503("asr"):
        # LLM style acquire (manual) must fail.
        with pytest.raises(HTTPException) as ei:
            _gpu_lock.try_acquire_or_503("llm")
        assert ei.value.status_code == 503
        # And TTS style must also fail.
        with pytest.raises(HTTPException) as ei:
            _gpu_lock.try_acquire_or_503("tts")
        assert ei.value.status_code == 503

    # After ASR releases, LLM can take the slot.
    _gpu_lock.try_acquire_or_503("llm")
    assert _gpu_lock.current_holder() == "llm"
    # And ASR is then blocked.
    with pytest.raises(HTTPException):
        with _gpu_lock.busy_or_503("asr"):
            pytest.fail("ASR should have been blocked while LLM held slot")
    _gpu_lock.release()


def test_503_message_names_current_holder():
    _gpu_lock.try_acquire_or_503("tts")
    try:
        with pytest.raises(HTTPException) as ei:
            _gpu_lock.try_acquire_or_503("asr")
        # detail is the standard envelope dict, not a bare string.
        detail = ei.value.detail
        assert isinstance(detail, dict) and "error" in detail
        assert detail["error"]["kind"] == "network"
        assert detail["error"]["provider"] == "aistack"
        assert "tts" in detail["error"]["message"]
    finally:
        _gpu_lock.release()
