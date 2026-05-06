"""End-to-end HTTP smoke tests with mocked upstreams.

Verifies the lock + cancel wiring at the FastAPI route level, not just at
the lock primitive. Ollama and Qwen3-TTS upstreams are replaced with
httpx MockTransports so these tests run on a CI box with no GPU and no
external services.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from aistack import _gpu_lock
from aistack.backends.llm import ollama as ollama_backend
from aistack.tts import qwen3 as tts_qwen3
from aistack.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def fake_ollama(monkeypatch):
    """Replace ollama_backend.make_client() with one that uses MockTransport.

    Returns a list of every upstream JSON body the test code received,
    so assertions can poke at e.g. injected keep_alive.
    """
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content) if request.content else {})
        body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": seen[-1].get("model", "test"),
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"}
            ],
        }
        return httpx.Response(
            200, json=body, headers={"content-type": "application/json"}
        )

    transport = httpx.MockTransport(handler)

    def fake_make_client():
        return httpx.AsyncClient(
            base_url=ollama_backend.UPSTREAM,
            timeout=ollama_backend.DEFAULT_TIMEOUT,
            transport=transport,
        )

    monkeypatch.setattr(ollama_backend, "make_client", fake_make_client)
    return seen


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_llm_returns_503_when_slot_busy(client, fake_ollama):
    """If ASR is mid-flight (slot held), an LLM request must 503 immediately
    instead of forwarding to Ollama. This is the cross-capability claim."""
    _gpu_lock.try_acquire_or_503("asr")
    try:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 503
        assert r.headers.get("Retry-After") == "5"
        # Upstream must NOT have been called.
        assert fake_ollama == []
    finally:
        _gpu_lock.release()


def test_llm_releases_slot_on_success(client, fake_ollama):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert not _gpu_lock.is_busy(), "slot must be released after non-streaming response"
    # keep_alive default injected.
    assert fake_ollama[0].get("keep_alive") == "30s"


def test_llm_releases_slot_on_upstream_unreachable(client, monkeypatch):
    """ConnectError on Ollama side must still free the slot before 503."""
    transport = httpx.MockTransport(
        lambda req: (_ for _ in ()).throw(httpx.ConnectError("refused"))
    )

    def fake_make_client():
        return httpx.AsyncClient(
            base_url=ollama_backend.UPSTREAM,
            timeout=ollama_backend.DEFAULT_TIMEOUT,
            transport=transport,
        )

    monkeypatch.setattr(ollama_backend, "make_client", fake_make_client)

    r = client.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 503
    body = r.json()
    assert body["error"]["kind"] == "network"
    assert not _gpu_lock.is_busy(), "slot must be released after upstream error"


def test_tts_returns_503_when_slot_busy(client, monkeypatch):
    """LLM holding the slot must block TTS too — the symmetric direction."""
    # Stub the TTS upstream so even if the request slipped through it
    # wouldn't try to hit a real container.
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, content=b"audio")
    )
    monkeypatch.setattr(
        tts_qwen3, "make_client",
        lambda: httpx.AsyncClient(
            base_url=tts_qwen3.UPSTREAM, transport=transport
        ),
    )

    _gpu_lock.try_acquire_or_503("llm")
    try:
        r = client.post(
            "/v1/audio/speech", json={"model": "qwen", "input": "hello"}
        )
        assert r.status_code == 503
        assert r.headers.get("Retry-After") == "5"
    finally:
        _gpu_lock.release()
