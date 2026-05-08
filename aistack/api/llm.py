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

Errors are wrapped in the standard envelope shape — see
aistack.errors. Connection-refused on Ollama maps to `network` kind
with an actionable message.
"""

from __future__ import annotations

import json
import logging

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from aistack import _gpu_lock, _model_cache
from aistack import observability as obs
from aistack.api._schemas import ErrorEnvelope
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


@router.post(
    "/v1/chat/completions",
    summary="Chat completion (Ollama proxy)",
    response_model=None,  # Pass-through Ollama JSON / SSE; documented via responses below.
    responses={
        200: {
            "description": (
                "Ollama's response, forwarded verbatim. Schema follows OpenAI's "
                "`/v1/chat/completions` contract — see "
                "https://platform.openai.com/docs/api-reference/chat for the "
                "field reference. When `stream=true` in the request, the "
                "response is a Server-Sent Events stream of OpenAI-shape "
                "delta chunks terminated by `data: [DONE]`."
            ),
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "description": "Non-streaming response. Pass-through OpenAI chat-completion shape.",
                    },
                },
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "description": "Streaming response (when stream=true). SSE chunks of OpenAI-shape deltas.",
                    },
                },
            },
        },
        400: {"model": ErrorEnvelope, "description": "Request body is not valid JSON."},
        502: {"model": ErrorEnvelope, "description": "Ollama upstream produced an unexpected error."},
        503: {
            "model": ErrorEnvelope,
            "description": (
                "Either the GPU slot is busy serving another inference (gateway-level), "
                "or Ollama is unreachable (e.g. the daemon is not running). The error "
                "envelope's `provider` field distinguishes the two."
            ),
        },
    },
)
async def chat_completions(request: Request) -> Response:
    """OpenAI-compatible chat completion endpoint, reverse-proxied to a
    local Ollama daemon (default `http://127.0.0.1:11434`).

    **Value-adds over a direct Ollama call.** Two things happen between
    the client and the upstream that justify routing through aistack:

    1. **GPU scheduling.** Before forwarding, aistack evicts its own
       in-process ASR `asr-main` cache entries so the LLM inference
       does not contend with a hot Whisper/Parakeet/SenseVoice for
       VRAM. The whole call holds the gateway's single GPU slot, so
       concurrent LLM/ASR/TTS requests get HTTP 503 with `Retry-After`.

    2. **`keep_alive` default.** When the client omits the
       `keep_alive` field, aistack injects `"30s"` so Ollama releases
       the model shortly after the completion. Sequential LLM calls
       within that window reuse the loaded model; idle releases VRAM
       back for ASR. Clients that want a different lifetime override
       explicitly.

    **Streaming.** When `stream=true` the response is forwarded
    chunk-by-chunk via FastAPI StreamingResponse, so the client sees
    first tokens as soon as Ollama emits them. Client disconnect
    propagates: aistack closes the upstream connection so Ollama's
    runner can abort generation rather than running to completion on
    a dead socket.

    **Cancellation.** Client disconnect mid-stream releases the GPU
    slot promptly and aborts the upstream call.

    **Request schema.** OpenAI-compatible — see the OpenAI Chat
    Completion API reference for field semantics. aistack does not
    transform the request body except to inject the `keep_alive`
    default; it forwards every other field verbatim.
    """
    obs_state = obs.state_for(request)
    try:
        body_bytes = await request.body()
        if obs_state is not None and obs_state.capture is not None:
            obs_state.capture.set_request_body(
                body_bytes, content_type=request.headers.get("content-type")
            )
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
        if obs_state is not None:
            obs_state.model = body.get("model")
            obs_state.extra["stream"] = streaming
            msgs = body.get("messages")
            if isinstance(msgs, list):
                obs_state.extra["message_count"] = len(msgs)

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
