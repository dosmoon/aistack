"""X-Request-ID propagation — pure ASGI.

Honors an inbound `X-Request-ID` header (so upstream callers like
VideoCraft can stitch their trace IDs through aistack), or generates
a 16-hex-char id when absent. Stashes the id on `request.state.request_id`
for downstream code (logging, payload filenames, metrics extras) and
echoes it back as a response header so clients can correlate logs
without parsing the response body.

16 hex chars (= 64 bits of randomness) is short enough for human eyeballs
and far below the birthday-collision danger zone for any single instance's
realistic traffic.

Implemented as a pure-ASGI middleware (not BaseHTTPMiddleware) so that
mutations to `scope["state"]` propagate to inner middleware. Starlette's
BaseHTTPMiddleware breaks scope-state propagation in some flows
(notably TestClient + a downstream pure-ASGI middleware reading scope),
which would silently leave request_id="unknown" in metrics and access
log even though the response header echoed correctly.
"""
from __future__ import annotations

import secrets

from starlette.datastructures import MutableHeaders, State

HEADER_NAME = "X-Request-ID"
_HEADER_BYTES = b"x-request-id"
_MAX_FORWARDED_LEN = 128  # cap caller-supplied IDs; rejects garbage / abuse


def _generate() -> str:
    return secrets.token_hex(8)


def _normalize(value: str | None) -> str:
    if not value:
        return _generate()
    v = value.strip()
    if not v or len(v) > _MAX_FORWARDED_LEN:
        return _generate()
    if not all(c.isalnum() or c in "-_:." for c in v):
        return _generate()
    return v


def _read_inbound(scope) -> str | None:
    for k, v in scope.get("headers", []):
        if k == _HEADER_BYTES:
            try:
                return v.decode("latin-1")
            except Exception:
                return None
    return None


class RequestIdMiddleware:
    """Pure-ASGI middleware. Sets scope["state"].request_id; echoes header."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        rid = _normalize(_read_inbound(scope))

        # Lazily-create State so downstream middleware reading
        # `scope["state"].request_id` (e.g. ObservabilityMiddleware) sees it.
        if "state" not in scope or not isinstance(scope["state"], State):
            scope["state"] = State()
        scope["state"].request_id = rid

        async def _send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                if HEADER_NAME not in headers:
                    headers[HEADER_NAME] = rid
            await send(message)

        await self.app(scope, receive, _send_wrapper)
