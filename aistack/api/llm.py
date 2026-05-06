"""LLM HTTP route — `POST /v1/chat/completions`.

OpenAI-compatible chat completion endpoint that aistack reverse-proxies
to a local Ollama daemon. Adds two pieces of value over a direct call:

  1. Local-GPU scheduling: before forwarding, we evict aistack's own
     ASR main-model slot so the LLM inference does not OOM the GPU
     when VRAM is tight. The eviction is best-effort and synchronous —
     we wait for it to settle before forwarding.

  2. Sane keep_alive default: when the client omits the `keep_alive`
     field, we inject "30s" so Ollama releases the model shortly
     after the request completes. Sequential LLM calls within 30s
     reuse the loaded model; idle Ollama returns VRAM to whoever
     needs it next.

Streaming responses (`stream=true`) are forwarded chunk-by-chunk via
FastAPI StreamingResponse so the client sees first tokens as soon as
Ollama emits them.

Errors are wrapped in the standard envelope (see aistack/errors.py
and docs/api/errors.md). Connection-refused on Ollama maps to
`network` kind with an actionable message.
"""

from __future__ import annotations

import json
import logging

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from aistack import _gpu_lock, _model_cache
from aistack.backends.llm import ollama as ollama_backend

logger = logging.getLogger("aistack.api.llm")

router = APIRouter(tags=["llm"])

# Hop-by-hop headers that must not be forwarded across the proxy.
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


def _filter_response_headers(src: httpx.Headers) -> dict[str, str]:
    return {k: v for k, v in src.items() if k.lower() not in _HOP_BY_HOP}


def _evict_asr_for_llm() -> None:
    """Free aistack's ASR slot before forwarding to Ollama."""
    n = _model_cache.evict_category("asr-main")
    if n:
        logger.info("evicted %d ASR main model(s) before LLM forward", n)


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    """Forward an OpenAI-compatible chat-completion call to Ollama."""
    try:
        body_bytes = await request.body()
        try:
            body = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "kind": "malformed",
                        "provider": "aistack",
                        "message": "Request body is not valid JSON.",
                    }
                },
            )
    except Exception as e:
        logger.exception("Failed to read request body")
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "kind": "malformed",
                    "provider": "aistack",
                    "message": f"Could not read request body: {e}",
                }
            },
        )

    # Acquire the global GPU slot before forwarding. LLM inference on
    # Ollama uses the same VRAM as in-process ASR / Qwen3-TTS container,
    # so concurrent capabilities would OOM the worker on tight cards.
    # Released either on early-error below, or in the streaming/non-
    # streaming finally so it spans the entire upstream call.
    _gpu_lock.try_acquire_or_503("llm")

    slot_released = False

    def _release_once():
        nonlocal slot_released
        if not slot_released:
            slot_released = True
            _gpu_lock.release()

    try:
        # Make room on the GPU before Ollama loads its model.
        _evict_asr_for_llm()

        # Inject keep_alive default if the client didn't specify.
        ollama_backend.inject_keep_alive(body)

        streaming = bool(body.get("stream"))

        client = ollama_backend.make_client()
        try:
            upstream_req = client.build_request(
                "POST",
                "/v1/chat/completions",
                json=body,
                headers={"content-type": "application/json"},
            )
            upstream_resp = await client.send(upstream_req, stream=streaming)
        except httpx.ConnectError:
            await client.aclose()
            _release_once()
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "kind": "network",
                        "provider": "aistack",
                        "message": (
                            "Ollama is not reachable at "
                            f"{ollama_backend.UPSTREAM}. Start it with: ollama serve"
                        ),
                    }
                },
            )
        except httpx.HTTPError as e:
            await client.aclose()
            _release_once()
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "kind": "unknown",
                        "provider": "aistack",
                        "message": f"Ollama upstream error: {type(e).__name__}: {e}",
                    }
                },
            )

        if streaming:
            async def _body_iter():
                try:
                    async for chunk in upstream_resp.aiter_raw():
                        # Client gone — stop pulling tokens from Ollama.
                        # Closing upstream_resp below sends RST so Ollama's
                        # runner can abort generation rather than running
                        # to completion on a dead connection.
                        try:
                            if await request.is_disconnected():
                                logger.info(
                                    "client disconnected; aborting upstream LLM stream"
                                )
                                break
                        except Exception:
                            # is_disconnected() can raise on certain ASGI
                            # states; treat as "still connected" and keep going.
                            pass
                        yield chunk
                finally:
                    await upstream_resp.aclose()
                    await client.aclose()
                    _release_once()

            return StreamingResponse(
                _body_iter(),
                status_code=upstream_resp.status_code,
                headers=_filter_response_headers(upstream_resp.headers),
                media_type=upstream_resp.headers.get(
                    "content-type", "text/event-stream"),
            )

        # Non-streaming: read full body and return.
        try:
            content = await upstream_resp.aread()
        finally:
            await upstream_resp.aclose()
            await client.aclose()

        _release_once()
        return Response(
            content=content,
            status_code=upstream_resp.status_code,
            headers=_filter_response_headers(upstream_resp.headers),
            media_type=upstream_resp.headers.get(
                "content-type", "application/json"),
        )
    except BaseException:
        # Anything we didn't anticipate — make sure the slot is freed
        # before the exception propagates so the worker isn't deadlocked.
        _release_once()
        raise
