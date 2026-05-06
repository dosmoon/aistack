"""Shared pytest fixtures.

The aistack app pulls in heavy ML imports at module-load time only when
their providers are actually exercised, so plain `from aistack.main import
app` works fine on a CI box with no ML extras installed. Fixtures here
just give tests a clean lock state and a stub for the LLM/TTS upstreams.
"""
from __future__ import annotations

import pytest

from aistack import _gpu_lock, _model_cache


@pytest.fixture(autouse=True)
def _reset_global_state():
    """Each test starts with the GPU slot free and the model cache empty.

    Tests that exercise lock contention or eviction would otherwise see
    state bleed across test functions because both modules are
    process-singletons.
    """
    # Force-release the lock if a prior test left it held (e.g. crashed
    # mid-acquire). RLock semantics: only the holder can release, but
    # threading.Lock.release on an unlocked lock raises RuntimeError —
    # so we probe with locked() first.
    if _gpu_lock.is_busy():
        try:
            _gpu_lock.release()
        except RuntimeError:
            pass
    _model_cache._CACHE.clear()
    yield
    if _gpu_lock.is_busy():
        try:
            _gpu_lock.release()
        except RuntimeError:
            pass
    _model_cache._CACHE.clear()
