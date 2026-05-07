"""X-Request-ID propagation middleware.

Honors an inbound `X-Request-ID` header (so upstream callers like
VideoCraft can stitch their trace IDs through aistack), or generates
a 16-hex-char id when absent. Stashes the id on `request.state.request_id`
for downstream use (logging, payload filenames, metrics extras) and
echoes it back as a response header so clients can correlate logs
without parsing the response body.

16 hex chars (= 64 bits of randomness) is short enough for human eyeballs
and far below the birthday-collision danger zone for any single instance's
realistic traffic.
"""
from __future__ import annotations

import secrets
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

HEADER_NAME = "X-Request-ID"
_MAX_FORWARDED_LEN = 128  # cap caller-supplied IDs; rejects garbage / abuse


def _generate() -> str:
    return secrets.token_hex(8)


def _normalize(value: str | None) -> str:
    if not value:
        return _generate()
    v = value.strip()
    if not v or len(v) > _MAX_FORWARDED_LEN:
        return _generate()
    # Allow ASCII letters/digits/_-:. (covers UUID, hex, slug-ish ids).
    if not all(c.isalnum() or c in "-_:." for c in v):
        return _generate()
    return v


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = _normalize(request.headers.get(HEADER_NAME))
        request.state.request_id = rid
        response = await call_next(request)
        # Don't overwrite if a downstream handler already set it.
        response.headers.setdefault(HEADER_NAME, rid)
        return response
