"""Qwen3-TTS upstream client.

Thin reverse proxy to the vLLM-Omni container running at QWEN3_TTS_UPSTREAM
(default http://127.0.0.1:17860). aistack adds no logic on top yet — it
forwards requests verbatim and streams the response back.

Future work (not D2):
  - Aggregate /v1/audio/voices across multiple TTS backends
  - Per-request telemetry (latency, RTF estimation)
  - Upstream readiness caching to short-circuit when container is down
"""

from __future__ import annotations

import os

import httpx

UPSTREAM = os.environ.get("AISTACK_QWEN3_TTS_UPSTREAM", "http://127.0.0.1:17860")
MODEL_ID = "qwen3-tts-12hz-0.6b-customvoice"

# Long timeout: first request after container start triggers torch.compile,
# CUDA Graph capture, and shm broadcast warmup (observed up to ~150s on
# 4060 Laptop). Steady-state requests are sub-second; we keep the read
# budget generous so cold-start traffic does not get clipped.
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=UPSTREAM, timeout=DEFAULT_TIMEOUT)


async def is_healthy(client: httpx.AsyncClient | None = None) -> bool:
    own = client is None
    if own:
        client = make_client()
    try:
        r = await client.get("/health")
        return r.status_code == 200
    except httpx.HTTPError:
        return False
    finally:
        if own:
            await client.aclose()
