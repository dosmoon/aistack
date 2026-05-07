"""Regression: request_id set by RequestIdMiddleware must reach
ObservabilityMiddleware via scope["state"], even when no inbound
X-Request-ID header is supplied.

Earlier RequestIdMiddleware was BaseHTTPMiddleware-based; in TestClient
flows its writes to request.state did not propagate to a downstream
pure-ASGI middleware, leaving request_id="unknown" in the JSONL access
log (despite the response header echoing correctly). This regression
test fires a request without X-Request-ID and asserts the auto-
generated id appears in metrics.recent — which can only happen if
state propagation works.
"""
from __future__ import annotations

import re

from fastapi.testclient import TestClient

from aistack.main import app
from aistack.observability import metrics

_HEX16 = re.compile(r"^[0-9a-f]{16}$")


def test_auto_generated_request_id_reaches_metrics():
    metrics.reset()
    with TestClient(app) as client:
        # Use an observed endpoint that returns quickly without external deps.
        # /v1/audio/transcriptions with no file -> 422; still goes through
        # both middlewares so metrics.record gets called.
        r = client.post("/v1/audio/transcriptions", data={"model": "auto"})
        assert r.status_code in (400, 422, 500)
        echoed = r.headers.get("X-Request-ID")
        assert echoed is not None and _HEX16.match(echoed)

    snap = metrics.snapshot()["categories"].get("asr")
    assert snap is not None, "asr category should have been observed"
    rids = [s.get("request_id") for s in snap["recent"]]
    assert echoed in rids, (
        f"auto-generated request_id {echoed!r} did not propagate from "
        f"RequestIdMiddleware to ObservabilityMiddleware; saw {rids!r}"
    )


def test_supplied_request_id_reaches_metrics():
    metrics.reset()
    with TestClient(app) as client:
        r = client.post(
            "/v1/audio/transcriptions",
            data={"model": "auto"},
            headers={"X-Request-ID": "vc-pipeline-step-7"},
        )
        assert r.headers.get("X-Request-ID") == "vc-pipeline-step-7"

    snap = metrics.snapshot()["categories"]["asr"]
    rids = [s.get("request_id") for s in snap["recent"]]
    assert "vc-pipeline-step-7" in rids
