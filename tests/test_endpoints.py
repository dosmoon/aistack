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


def test_models_lists_known_shape(client):
    """Contract test: /v1/models returns the OpenAI-shape envelope."""
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body.get("object") == "list"
    assert isinstance(body.get("data"), list)
    for entry in body["data"]:
        assert entry.get("object") == "model"
        assert isinstance(entry.get("id"), str) and entry["id"]
        assert isinstance(entry.get("owned_by"), str)
        caps = entry.get("capabilities")
        assert isinstance(caps, list) and caps, (
            f"every entry must declare non-empty capabilities[]; got: {entry}"
        )
        assert all(c in ("asr", "tts", "llm") for c in caps), (
            f"unexpected capability values: {caps}"
        )


def test_models_asr_entries_carry_languages(client):
    """Contract test: every real ASR entry carries a `languages` array.

    The auto routing alias is exempt — its languages are dynamically
    determined by what's installed at request time. Real ASR entries
    must declare their supported language set so language-aware pickers
    can filter without baking a per-backend table client-side.
    """
    r = client.get("/v1/models")
    body = r.json()
    asr_real = [
        e for e in body["data"]
        if "asr" in (e.get("capabilities") or [])
        and not e.get("is_routing_alias")
    ]
    if not asr_real:
        pytest.skip("no real ASR backends installed")
    for entry in asr_real:
        langs = entry.get("languages")
        assert isinstance(langs, list) and langs, (
            f"ASR entry missing non-empty languages[]: {entry}"
        )
        # ISO 639-1 codes are 2-3 letters lowercase; sanity-check shape.
        for code in langs:
            assert isinstance(code, str) and 2 <= len(code) <= 3 and code.islower(), (
                f"language code looks malformed: {code!r} in {entry['id']}"
            )


def test_models_includes_auto_routing_alias_when_asr_available(client):
    """Contract test: when ≥1 ASR backend is reachable, the `auto` routing
    alias is present and marked with is_routing_alias=true.

    aistack's smart-routing capability must be discoverable from the
    inventory; a consumer should not have to read prose docs to learn
    that "auto" is a valid model id.
    """
    r = client.get("/v1/models")
    body = r.json()
    asr_real = [
        e for e in body["data"]
        if "asr" in (e.get("capabilities") or [])
        and not e.get("is_routing_alias")
    ]
    if not asr_real:
        pytest.skip("no real ASR backends installed; auto would have nothing to route to")

    auto_entries = [e for e in body["data"] if e.get("id") == "auto"]
    assert auto_entries, (
        "expected an 'auto' routing-alias entry alongside real ASR entries"
    )
    assert len(auto_entries) == 1, "auto should appear exactly once"
    auto = auto_entries[0]
    assert auto.get("is_routing_alias") is True
    assert auto.get("capabilities") == ["asr"]
    # languages is intentionally absent on the alias — its routing
    # decision is per-request based on the language hint.
    assert "languages" not in auto, (
        "routing alias should not advertise a fixed languages set"
    )


def test_models_every_entry_declares_supports_streaming(client):
    """Contract test: every /v1/models entry carries `supports_streaming`.

    Picker UIs filter on this when the user wants real-time output.
    The field must be present on every entry (real models, routing
    aliases, all three capabilities) so consumers don't have to
    special-case missing fields per backend type.
    """
    r = client.get("/v1/models")
    body = r.json()
    for entry in body["data"]:
        assert "supports_streaming" in entry, (
            f"every entry must declare supports_streaming; got: {entry}"
        )
        assert isinstance(entry["supports_streaming"], bool), (
            f"supports_streaming must be bool; got: {entry['supports_streaming']!r}"
        )


def test_models_parakeet_does_not_advertise_streaming(client):
    """Contract test: Parakeet TDT v3 advertises supports_streaming=false.

    The native model.transcribe() is non-incremental and chunking it
    externally trades measurable accuracy for stream output. Until
    that trade-off is measured + accepted, Parakeet must not promise
    streaming. The gateway accepts stream=true on Parakeet but emits
    a single-event SSE downgrade response (out of scope of this test
    — covered separately when streaming is implemented).
    """
    r = client.get("/v1/models")
    body = r.json()
    by_id = {e["id"]: e for e in body["data"]}
    pk = by_id.get("nvidia/parakeet-tdt-0.6b-v3")
    if pk is None:
        pytest.skip("Parakeet not installed")
    assert pk["supports_streaming"] is False


def test_models_auto_streaming_reflects_pool(client):
    """Contract test: the auto routing alias's supports_streaming is the
    AND of the candidate pool — True only if every installed real ASR
    backend supports streaming.

    Currently Parakeet does not support streaming, so as long as
    Parakeet is in the installed set the auto alias must be False.
    Catches accidents where someone bumps auto to True without
    updating the join logic.
    """
    r = client.get("/v1/models")
    body = r.json()
    by_id = {e["id"]: e for e in body["data"]}

    auto = by_id.get("auto")
    if auto is None:
        pytest.skip("auto routing alias not present")

    real = [
        e for e in body["data"]
        if "asr" in e.get("capabilities", [])
        and not e.get("is_routing_alias")
    ]
    if not real:
        pytest.skip("no real ASR backends to verify the AND against")
    expected = all(e.get("supports_streaming") for e in real)
    assert auto["supports_streaming"] == expected, (
        "auto.supports_streaming must equal AND(real_entry.supports_streaming); "
        f"saw auto={auto['supports_streaming']!r}, real={[e['id']+':'+str(e['supports_streaming']) for e in real]}"
    )


def test_models_languages_match_known_backend_coverage(client):
    """Contract test: when SenseVoice is installed, its languages list
    contains the documented CJK + en/ja/ko set; when Parakeet is
    installed, its list contains the 25 European languages plus en.

    Catches accidental edits to the language tables that would break
    language-aware picker filtering downstream.
    """
    r = client.get("/v1/models")
    body = r.json()
    by_id = {e["id"]: e for e in body["data"]}

    sv = by_id.get("iic/SenseVoiceSmall")
    if sv is not None:
        langs = set(sv["languages"])
        # Documented coverage per docs/api/models.md.
        for code in ("zh", "yue", "en", "ja", "ko"):
            assert code in langs, (
                f"SenseVoice should advertise {code!r}; saw: {sv['languages']}"
            )

    pk = by_id.get("nvidia/parakeet-tdt-0.6b-v3")
    if pk is not None:
        langs = set(pk["languages"])
        for code in ("en", "fr", "de", "es", "ru", "uk"):
            assert code in langs, (
                f"Parakeet should advertise {code!r}; saw: {pk['languages']}"
            )


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
        # Body uses the standard envelope, not the legacy {"detail": ...} shape.
        body = r.json()
        assert "error" in body, f"expected envelope shape, got: {body}"
        assert body["error"]["kind"] == "network"
        assert body["error"]["provider"] == "aistack"
        assert "asr" in body["error"]["message"]  # holder name in message
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
