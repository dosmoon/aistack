"""Smoke tests for the /admin Web UI.

Verifies the index page renders and every fragment endpoint returns
HTML 200 with the expected structure. Real backends are not required —
ASR import-probes return what's installed, TTS/Ollama upstreams are
unavailable in CI which the templates handle (empty state).
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from aistack import _gpu_lock, _model_cache
from aistack.admin import log_buffer
from aistack.main import app


def test_admin_index_renders():
    with TestClient(app) as c:
        r = c.get("/admin")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        body = r.text
        assert "aistack admin" in body
        assert "GPU slot" in body
        assert "Models inventory" in body
        # HTMX is referenced so live-refresh works in the browser.
        assert "htmx" in body.lower()


def test_admin_trailing_slash():
    with TestClient(app) as c:
        r = c.get("/admin/")
        assert r.status_code == 200


def test_fragment_lock_idle_and_busy():
    with TestClient(app) as c:
        r = c.get("/admin/fragments/lock")
        assert r.status_code == 200
        assert "IDLE" in r.text

        _gpu_lock.try_acquire_or_503("llm")
        try:
            r = c.get("/admin/fragments/lock")
            assert "BUSY" in r.text
            assert "llm" in r.text
        finally:
            _gpu_lock.release()


def test_fragment_cache_empty_and_populated():
    with TestClient(app) as c:
        r = c.get("/admin/fragments/cache")
        assert r.status_code == 200
        assert "No models currently resident" in r.text

        _model_cache.put(
            "fake-provider", "fake-key", object(), category="asr-main"
        )
        try:
            r = c.get("/admin/fragments/cache")
            assert "fake-provider" in r.text
            assert "asr-main" in r.text
        finally:
            _model_cache._CACHE.clear()


def test_fragment_models_returns_html():
    with TestClient(app) as c:
        r = c.get("/admin/fragments/models")
        assert r.status_code == 200
        # Either a table or the empty-state message — both are valid HTML.
        assert "<" in r.text


def test_fragment_gpu_returns_html():
    with TestClient(app) as c:
        r = c.get("/admin/fragments/gpu")
        assert r.status_code == 200
        # Either device stats or the "No GPU stats" empty state.
        assert "GPU" in r.text or "No GPU stats" in r.text


def test_fragment_logs_reflects_ring_buffer():
    import logging

    log_buffer.clear()
    logging.getLogger("aistack.test").info("hello-from-test-line")

    with TestClient(app) as c:
        r = c.get("/admin/fragments/logs")
        assert r.status_code == 200
        assert "hello-from-test-line" in r.text


def test_fragment_logs_n_param_clamped():
    with TestClient(app) as c:
        r = c.get("/admin/fragments/logs?n=999999")
        assert r.status_code == 200
        # No crash; capacity-clamped server-side.


def test_reset_asr_state_json_response():
    """Plain POST (no HTMX header) returns the count + remaining cache stats."""
    _model_cache._CACHE.clear()
    _model_cache.put("fake-asr-main", "k1", object(), category="asr-main")
    _model_cache.put("fake-asr-aux", "k2", object(), category="asr-aux")
    _model_cache.put("fake-tts", "k3", object(), category="tts-main")
    try:
        with TestClient(app) as c:
            r = c.post("/admin/api/reset-asr-state")
            assert r.status_code == 200
            body = r.json()
            assert body["evicted"] == 2
            remaining_providers = {e["provider"] for e in body["remaining"]["loaded"]}
            assert remaining_providers == {"fake-tts"}
    finally:
        _model_cache._CACHE.clear()


def test_reset_asr_state_htmx_returns_cache_fragment():
    """HTMX requests get the re-rendered cache fragment for in-place swap."""
    _model_cache._CACHE.clear()
    _model_cache.put("fake-asr-main", "k1", object(), category="asr-main")
    try:
        with TestClient(app) as c:
            r = c.post(
                "/admin/api/reset-asr-state", headers={"HX-Request": "true"}
            )
            assert r.status_code == 200
            assert "text/html" in r.headers["content-type"]
            assert "No models currently resident" in r.text
    finally:
        _model_cache._CACHE.clear()


def test_reset_asr_state_empty_cache_is_noop():
    _model_cache._CACHE.clear()
    with TestClient(app) as c:
        r = c.post("/admin/api/reset-asr-state")
        assert r.status_code == 200
        assert r.json()["evicted"] == 0


def test_cache_fragment_shows_reset_button_when_asr_loaded():
    _model_cache._CACHE.clear()
    _model_cache.put("fake-asr", "k1", object(), category="asr-main")
    try:
        with TestClient(app) as c:
            r = c.get("/admin/fragments/cache")
            assert "Reset ASR state" in r.text
            assert "/admin/api/reset-asr-state" in r.text
    finally:
        _model_cache._CACHE.clear()


def test_cache_fragment_hides_reset_button_when_no_asr():
    _model_cache._CACHE.clear()
    _model_cache.put("fake-tts", "k1", object(), category="tts-main")
    try:
        with TestClient(app) as c:
            r = c.get("/admin/fragments/cache")
            assert "Reset ASR state" not in r.text
    finally:
        _model_cache._CACHE.clear()
