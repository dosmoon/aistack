"""Observability layer — metrics, access logs, payload capture, request-ID.

Three independent feature flags, all default-on except payload capture
(which writes user data to disk). See `config.py` for env vars and
the runtime-mutable settings dict, and `docs/public/api/observability.md`
for the wire formats.

Components:
    config       — env-seeded mutable settings + toggle API
    request_id   — X-Request-ID middleware
    metrics      — in-process rolling histograms + counters
    access_log   — daily-rolling JSONL writer
    payload      — per-request mp3/json on disk with size+age sweeper
    middleware   — ASGI middleware that ties it all together at request boundary
"""

from aistack.observability import access_log, config, metrics, payload, request_id
from aistack.observability.middleware import ObservabilityMiddleware, ObservabilityState
from aistack.observability.request_id import RequestIdMiddleware

__all__ = [
    "access_log",
    "config",
    "metrics",
    "payload",
    "request_id",
    "ObservabilityMiddleware",
    "ObservabilityState",
    "RequestIdMiddleware",
    "state_for",
]


def state_for(request) -> ObservabilityState | None:
    """Convenience: return the per-request ObservabilityState if this
    request is being observed, else None. Routes use it to drop extras
    and capture-context calls without caring whether observability is on.

        st = obs.state_for(request)
        if st is not None:
            st.model = "iic/SenseVoiceSmall"
            st.extra["audio_sec"] = duration
            if st.capture is not None:
                st.capture.adopt_request_file(audio_path)
    """
    try:
        return getattr(request.state, "observability", None)
    except Exception:
        return None
