"""ASGI middleware that brackets every HTTP request with observability work.

Pure-ASGI (not BaseHTTPMiddleware) so we can tee the response body without
buffering it — important for SSE/streaming responses where buffering
would defeat the streaming.

What it does:
  - On request start: open a payload.CaptureContext (if payload toggle on)
    and stash on `request.state.observability` together with a small dict
    that route handlers can drop extras into (audio_sec, rtf, etc.).
  - On every response chunk: tee bytes into the capture context.
  - On request finish: emit metrics.record + access_log.write +
    capture_ctx.finalize().

Categorization is by URL path prefix:
    /v1/audio/transcriptions  -> asr
    /v1/chat/                  -> llm
    /v1/audio/                  -> tts (anything under /v1/audio/* not ASR)
    everything else            -> "other" (admin, /health, /v1/models)

We don't observe "other" — too noisy and not interesting for performance
analysis. Health checks can drown the log otherwise.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from aistack.observability import access_log, config, metrics, payload

logger = logging.getLogger("aistack.obs.mw")


def _category_for(path: str, method: str) -> str | None:
    """Return capability name, or None to skip observation."""
    if method == "OPTIONS":
        return None
    if path.startswith("/v1/audio/transcriptions"):
        return "asr"
    if path.startswith("/v1/chat/"):
        return "llm"
    if path.startswith("/v1/audio/"):
        return "tts"
    return None


class ObservabilityState:
    """Mutable scratchpad for route handlers to drop request-scoped extras.

    Lives at request.state.observability. Routes can do things like:
        st = request.state.observability
        st.extra["audio_sec"] = duration
        st.extra["model"] = canonical_id

    The middleware copies `extra` into both metrics and access_log.
    """
    __slots__ = ("started_monotonic", "category", "capture", "extra",
                 "slot_wait_ms", "model")

    def __init__(self, category: str) -> None:
        self.started_monotonic = time.perf_counter()
        self.category = category
        self.capture: payload.CaptureContext | None = None
        self.extra: dict[str, Any] = {}
        self.slot_wait_ms: float = 0.0
        self.model: str | None = None


class ObservabilityMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        method = scope["method"]
        category = _category_for(path, method)
        if category is None:
            await self.app(scope, receive, send)
            return

        rid = _scope_request_id(scope)
        state = ObservabilityState(category)
        # Starlette's Request.state lazily creates a State() on first
        # access. Pre-create it here and attach our handle so routes
        # can read `request.state.observability`.
        from starlette.datastructures import State
        if "state" not in scope or not isinstance(scope["state"], State):
            scope["state"] = State()
        scope["state"].observability = state

        if config.is_enabled("payload"):
            state.capture = payload.begin(rid)

        status_code: int = 0
        response_headers: list[tuple[bytes, bytes]] = []

        async def _send_wrapper(message):
            nonlocal status_code, response_headers
            mtype = message["type"]
            if mtype == "http.response.start":
                status_code = message.get("status", 0)
                response_headers = message.get("headers", [])
            elif mtype == "http.response.body":
                body = message.get("body", b"")
                if state.capture is not None and body:
                    state.capture.append_response(body)
            await send(message)

        disconnected = False
        try:
            await self.app(scope, receive, _send_wrapper)
        except Exception:
            status_code = status_code or 500
            raise
        finally:
            elapsed_ms = (time.perf_counter() - state.started_monotonic) * 1000.0
            try:
                metrics.record(
                    category,
                    status_code=status_code,
                    latency_ms=elapsed_ms,
                    slot_wait_ms=state.slot_wait_ms,
                    request_id=rid,
                    extra={**state.extra, "model": state.model} if state.model else state.extra,
                    disconnected=disconnected,
                )
            except Exception:
                logger.exception("metrics.record failed")
            try:
                client = scope.get("client")
                client_str = f"{client[0]}:{client[1]}" if client else None
                access_log.write({
                    "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "request_id": rid,
                    "method": method,
                    "path": path,
                    "query": scope.get("query_string", b"").decode("latin-1") or None,
                    "status": status_code,
                    "category": category,
                    "model": state.model,
                    "latency_ms": round(elapsed_ms, 2),
                    "slot_wait_ms": round(state.slot_wait_ms, 2),
                    "client": client_str,
                    "extra": state.extra or None,
                })
            except Exception:
                logger.exception("access_log.write failed")
            if state.capture is not None:
                try:
                    state.capture.finalize({
                        "method": method,
                        "path": path,
                        "status": status_code,
                        "category": category,
                        "model": state.model,
                        "latency_ms": round(elapsed_ms, 2),
                        "slot_wait_ms": round(state.slot_wait_ms, 2),
                        "headers": _headers_dict(scope.get("headers", [])),
                        "response_headers": _headers_dict(response_headers),
                        "extra": state.extra,
                    })
                except Exception:
                    logger.exception("payload finalize failed")


def _scope_request_id(scope) -> str:
    """RequestIdMiddleware (BaseHTTPMiddleware) sets request.state.request_id
    on the State object at scope["state"]. Fall back to scanning headers."""
    try:
        st = scope.get("state")
        rid = getattr(st, "request_id", None)
        if rid:
            return rid
    except Exception:
        pass
    for k, v in scope.get("headers", []):
        if k == b"x-request-id":
            return v.decode("latin-1", errors="replace")[:128]
    return "unknown"


def _headers_dict(headers: list[tuple[bytes, bytes]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in headers:
        try:
            out[k.decode("latin-1")] = v.decode("latin-1")
        except Exception:
            continue
    return out
