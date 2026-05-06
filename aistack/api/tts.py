"""TTS HTTP routes.

Exposes the OpenAI-compatible /v1/audio/* surface and proxies everything to
the configured upstream backend (currently Qwen3-TTS via vLLM-Omni).

The proxy is intentionally transparent: request body, headers (minus hop-by-hop),
and response body all flow through unchanged. aistack adds no business logic
at this layer in D2 — that is the responsibility of consumers like VideoCraft.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from aistack.tts import qwen3

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


@router.api_route(
    "/v1/audio/{path:path}",
    methods=["GET", "POST", "DELETE"],
)
async def proxy(path: str, request: Request) -> Response:
    """Transparent reverse proxy for /v1/audio/* to the TTS upstream."""
    upstream_path = f"/v1/audio/{path}"
    body = await request.body()
    fwd_headers = _filter_request_headers(dict(request.headers))

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
        return Response(
            status_code=502,
            content=f'{{"error":{{"code":"upstream_error","message":"{type(exc).__name__}: {exc}"}}}}'.encode(),
            media_type="application/json",
        )

    async def _body_iter():
        try:
            async for chunk in upstream_resp.aiter_raw():
                yield chunk
        finally:
            await upstream_resp.aclose()
            await client.aclose()

    return StreamingResponse(
        _body_iter(),
        status_code=upstream_resp.status_code,
        headers=_filter_response_headers(upstream_resp.headers),
        media_type=upstream_resp.headers.get("content-type"),
    )
