"""TTS HTTP routes.

Exposes the OpenAI-compatible /v1/audio/* surface and proxies everything to
the configured upstream backend (currently Qwen3-TTS via vLLM-Omni).

The proxy is intentionally transparent: request body, headers (minus hop-by-hop),
and response body all flow through unchanged. aistack adds no business logic
at this layer — that is the responsibility of consumers like VideoCraft.

The OpenAI-compatible request/response schemas are owned by OpenAI's API
reference (https://platform.openai.com/docs/api-reference/audio); aistack
forwards them verbatim and does not redocument the field semantics.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from aistack import _gpu_lock
from aistack import observability as obs
from aistack.api._schemas import ErrorEnvelope
from aistack.tts import qwen3

logger = logging.getLogger("aistack.api.tts")

router = APIRouter(tags=["tts"])

# Hop-by-hop headers that must not be forwarded (RFC 7230 §6.1).
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def _filter_request_headers(src: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in src.items() if k.lower() not in _HOP_BY_HOP}


def _filter_response_headers(src: httpx.Headers) -> dict[str, str]:
    return {k: v for k, v in src.items() if k.lower() not in _HOP_BY_HOP}


_TTS_PROXY_RESPONSES = {
    200: {
        "description": (
            "Upstream Qwen3-TTS response, forwarded verbatim. For "
            "`POST /v1/audio/speech` the body is raw audio bytes "
            "(content-type per OpenAI spec); other paths under "
            "`/v1/audio/*` (clone-voice / list-voices / etc.) preserve "
            "the upstream's content-type and shape."
        ),
        "content": {
            "audio/mpeg": {"schema": {"type": "string", "format": "binary"}},
            "audio/wav": {"schema": {"type": "string", "format": "binary"}},
            "audio/opus": {"schema": {"type": "string", "format": "binary"}},
            "application/json": {
                "schema": {
                    "type": "object",
                    "description": "Non-audio responses (e.g. voice catalog), pass-through from Qwen3-TTS.",
                },
            },
        },
    },
    502: {"model": ErrorEnvelope, "description": "Qwen3-TTS upstream produced an unexpected error."},
    503: {
        "model": ErrorEnvelope,
        "description": (
            "Either the GPU slot is busy serving another inference (gateway-level), "
            "or the Qwen3-TTS container is unreachable. The error envelope's "
            "`provider` field distinguishes the two."
        ),
    },
}


_TTS_PROXY_DOC = """Transparent reverse proxy for /v1/audio/* to the
Qwen3-TTS-12Hz-0.6B-CustomVoice container.

**Transparent.** Request body, headers (minus hop-by-hop), and response
body all flow through unchanged. aistack does not transcode audio,
swap voices, or adapt OpenAI's spec — what Qwen3-TTS returns is what
the client receives. The OpenAI-compatible request/response schemas
are documented authoritatively at
https://platform.openai.com/docs/api-reference/audio.

**GPU scheduling.** Holds the global gateway GPU slot for the
duration of the upstream call. The Qwen3-TTS container generates on
the same physical GPU as in-process ASR / LLM workloads, so the slot
represents "GPU is doing inference" regardless of which process owns
the kernels. Concurrent requests get HTTP 503 with `Retry-After`.

**Streaming.** The upstream is consumed and forwarded chunk-by-chunk
so multi-MB audio responses don't buffer entirely in memory. Client
disconnect propagates: aistack closes the upstream connection so the
container can abort generation early.

**Error mapping.** ConnectError on the upstream → 503 with a hint to
start the docker compose stack. Other httpx errors → 502 with the
upstream exception type/message in the envelope.
"""


@router.post(
    "/v1/audio/{path:path}",
    operation_id="tts_proxy_post",
    summary="Proxy to Qwen3-TTS (POST)",
    response_model=None,
    responses=_TTS_PROXY_RESPONSES,
)
async def proxy_post(path: str, request: Request) -> Response:
    return await _proxy(path, request)


@router.get(
    "/v1/audio/{path:path}",
    operation_id="tts_proxy_get",
    summary="Proxy to Qwen3-TTS (GET)",
    response_model=None,
    responses=_TTS_PROXY_RESPONSES,
)
async def proxy_get(path: str, request: Request) -> Response:
    return await _proxy(path, request)


@router.delete(
    "/v1/audio/{path:path}",
    operation_id="tts_proxy_delete",
    summary="Proxy to Qwen3-TTS (DELETE)",
    response_model=None,
    responses=_TTS_PROXY_RESPONSES,
)
async def proxy_delete(path: str, request: Request) -> Response:
    return await _proxy(path, request)


# Attach the shared docstring to all three methods so /docs shows it
# regardless of which method the reader inspects.
proxy_post.__doc__ = _TTS_PROXY_DOC
proxy_get.__doc__ = _TTS_PROXY_DOC
proxy_delete.__doc__ = _TTS_PROXY_DOC


async def _proxy(path: str, request: Request) -> Response:
    upstream_path = f"/v1/audio/{path}"
    body = await request.body()
    fwd_headers = _filter_request_headers(dict(request.headers))

    obs_state = obs.state_for(request)
    if obs_state is not None:
        obs_state.extra["upstream_path"] = upstream_path
        obs_state.extra["request_bytes"] = len(body)
        # Try to pull the requested voice/model from a JSON body for
        # observability without disturbing the proxy bytes.
        ct = request.headers.get("content-type", "").lower()
        if "application/json" in ct and body:
            try:
                import json as _json
                parsed = _json.loads(body)
                if isinstance(parsed, dict):
                    obs_state.model = parsed.get("model") or obs_state.model
                    if "voice" in parsed:
                        obs_state.extra["voice"] = parsed["voice"]
            except Exception:
                pass
        if obs_state.capture is not None:
            obs_state.capture.set_request_body(
                body, content_type=request.headers.get("content-type")
            )

    _gpu_lock.try_acquire_or_503("tts")

    slot_released = False

    def _release_once():
        nonlocal slot_released
        if not slot_released:
            slot_released = True
            _gpu_lock.release()

    try:
        client = qwen3.make_client()
        try:
            # Stream the upstream response so large audio payloads don't buffer
            # entirely in memory before reaching the caller.
            upstream_req = client.build_request(
                request.method,
                upstream_path,
                content=body,
                headers=fwd_headers,
                params=request.query_params,
            )
            upstream_resp = await client.send(upstream_req, stream=True)
        except httpx.ConnectError:
            await client.aclose()
            _release_once()
            return Response(
                status_code=503,
                content=(
                    b'{"error":{"code":"upstream_unavailable",'
                    b'"message":"Qwen3-TTS container is not reachable. '
                    b'Start it with: docker compose -f docker/tts_qwen3/docker-compose.yml up -d"}}'
                ),
                media_type="application/json",
            )
        except httpx.HTTPError as exc:
            await client.aclose()
            _release_once()
            return Response(
                status_code=502,
                content=f'{{"error":{{"code":"upstream_error","message":"{type(exc).__name__}: {exc}"}}}}'.encode(),
                media_type="application/json",
            )

        async def _body_iter():
            try:
                async for chunk in upstream_resp.aiter_raw():
                    try:
                        if await request.is_disconnected():
                            logger.info(
                                "client disconnected; aborting upstream TTS stream"
                            )
                            break
                    except Exception:
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
            media_type=upstream_resp.headers.get("content-type"),
        )
    except BaseException:
        _release_once()
        raise
