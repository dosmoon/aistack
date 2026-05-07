"""Ollama LLM backend.

aistack proxies `POST /v1/chat/completions` to a local (or LAN) Ollama
daemon and aggregates Ollama's installed models into `GET /v1/models`.
Ollama remains the actual LLM runtime — aistack adds:

  - Capability discovery: Ollama models surface in /v1/models with
    capabilities=["llm"] so consumers can build pickers.
  - Local resource scheduling: before forwarding an LLM request, we
    free aistack's own ASR slots (asr-main category) so the LLM
    inference does not OOM the GPU on tight VRAM.
  - keep_alive policy: when the client does not specify keep_alive,
    we inject "30s" so Ollama releases the model shortly after each
    chat completion. This minimizes contention with subsequent ASR
    calls. Clients that want different lifetime override explicitly.

Configuration via env var:
    AISTACK_OLLAMA_URL   default http://127.0.0.1:11434
"""

from __future__ import annotations

import os
import logging
from typing import AsyncIterator

import httpx

logger = logging.getLogger("aistack.backends.llm.ollama")

UPSTREAM = os.environ.get("AISTACK_OLLAMA_URL", "http://127.0.0.1:11434")

# Default keep_alive injected when the client omits it. Short enough that
# a downstream ASR call within a few minutes does not race with a
# resident Ollama model for VRAM, long enough that a sequence of chat
# completions does not pay the cold-load tax on every turn.
DEFAULT_KEEP_ALIVE = "30s"

# Ollama can take a while to load big models; chat completion itself
# can also stream over many seconds. The connect timeout is short
# (Ollama up or down decides quickly) but read is generous.
DEFAULT_TIMEOUT = httpx.Timeout(connect=2.0, read=600.0, write=30.0, pool=5.0)


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=UPSTREAM, timeout=DEFAULT_TIMEOUT)


async def is_healthy(client: httpx.AsyncClient | None = None) -> bool:
    """True when Ollama responds. Used by /v1/models to decide whether
    to include Ollama-served models in the inventory."""
    own = client is None
    if own:
        client = make_client()
    try:
        # Ollama exposes /api/tags; using it as a liveness probe also
        # warms the path used by model_entries().
        r = await client.get("/api/tags")
        return r.status_code == 200
    except httpx.HTTPError:
        return False
    finally:
        if own:
            await client.aclose()


async def model_entries() -> list[dict]:
    """Aggregate Ollama's installed models into /v1/models entries.

    Returns [] if Ollama is unreachable rather than raising — the
    /v1/models endpoint should never 500 just because one backend is
    down. Consumers see "no LLM available right now" naturally.
    """
    try:
        async with make_client() as client:
            r = await client.get("/api/tags")
            if r.status_code != 200:
                return []
            data = r.json()
    except httpx.HTTPError:
        return []
    except (ValueError, TypeError) as e:
        logger.warning("Ollama /api/tags returned non-JSON: %s", e)
        return []

    out: list[dict] = []
    for m in data.get("models", []):
        name = m.get("name") or m.get("model")
        if not name:
            continue
        out.append({
            "id": name,
            "object": "model",
            "owned_by": "ollama",
            "capabilities": ["llm"],
            # Ollama's /v1/chat/completions natively streams via SSE
            # (`stream: true`); aistack forwards the SSE body chunks
            # verbatim, so every Ollama-served LLM supports streaming.
            "supports_streaming": True,
        })
    return out


def inject_keep_alive(body: dict) -> dict:
    """Add aistack's default keep_alive when the client did not specify.

    The mutation is in-place and the dict is returned for chaining.
    Clients that explicitly pass `keep_alive` are respected — gateway
    only fills the gap.
    """
    if "keep_alive" not in body:
        body["keep_alive"] = DEFAULT_KEEP_ALIVE
    return body
