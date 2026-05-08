"""In-process metrics: rolling latency histograms + counters.

Per category (asr / llm / tts) we keep:
  - total request count, broken down by status_class
  - latency_ms samples (rolling window — METRICS_WINDOW_SEC)
  - slot_wait_ms samples (rolling window)
  - free-form extras carried in `last_samples` (last 50 requests, for
    spot-checking what's been happening)

We do not depend on prometheus_client — three reasons:
  1. zero new pip dependency
  2. native histogram with proper percentiles, not bucket-only
  3. rolling window beats process-lifetime counters for "is performance
     regressing right now"

`snapshot()` is called by /admin/api/metrics and the metrics dashboard
fragment; it returns a JSON-serialisable dict whose authoritative shape
is the `MetricsSnapshot` Pydantic model in `aistack.api._schemas`.

status_class taxonomy:
    "2xx" — success
    "4xx" — client error (validation, unknown model)
    "5xx" — server error (provider crash, internal)
    "503-busy" — slot mutex rejection; called out separately because
                 it's load-shedding, not a real failure
    "client-disconnect" — request aborted before completion
"""
from __future__ import annotations

import bisect
import threading
import time
from collections import defaultdict, deque
from typing import Any

from aistack.observability import config

# Power-of-roughly-2 buckets, anchored to ASR/LLM realistic ranges.
# Used only for the "histogram" rendering; percentiles are computed
# from the actual sample window, not the buckets.
BUCKETS_MS: list[int] = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000]

STATUS_CLASSES = ("2xx", "4xx", "5xx", "503-busy", "client-disconnect")


def _classify(status_code: int, *, slot_busy: bool = False, disconnected: bool = False) -> str:
    if disconnected:
        return "client-disconnect"
    if slot_busy:
        return "503-busy"
    if 200 <= status_code < 400:
        return "2xx"
    if 400 <= status_code < 500:
        return "4xx"
    return "5xx"


def is_slot_busy_response(status_code: int, response_headers: list) -> bool:
    """Detect a GPU-slot-busy 503 response. The marker is a 503 status
    accompanied by a `Retry-After` header — `_gpu_lock._busy_exception`
    sets both, no other 503 path in aistack does. Used by the
    observability middleware to classify load-shedding separately from
    real 5xx failures."""
    if status_code != 503:
        return False
    for k, _v in response_headers or []:
        try:
            if k.decode("latin-1").lower() == "retry-after":
                return True
        except Exception:
            continue
    return False


class _RollingSamples:
    """Time-windowed sample buffer. (timestamp, value) pairs; older than
    METRICS_WINDOW_SEC drop on next access. Bounded growth: deque eviction
    on every record() call keeps memory ~ window_sec × peak_qps."""

    __slots__ = ("samples",)

    def __init__(self) -> None:
        self.samples: deque[tuple[float, float]] = deque()

    def add(self, value: float) -> None:
        now = time.monotonic()
        cutoff = now - config.METRICS_WINDOW_SEC
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        self.samples.append((now, value))

    def values(self) -> list[float]:
        cutoff = time.monotonic() - config.METRICS_WINDOW_SEC
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        return [v for _, v in self.samples]

    def percentile(self, p: float) -> float:
        vs = sorted(self.values())
        if not vs:
            return 0.0
        idx = max(0, min(len(vs) - 1, int(round((p / 100.0) * (len(vs) - 1)))))
        return vs[idx]

    def histogram(self) -> dict[str, int]:
        out = {f"<={b}": 0 for b in BUCKETS_MS}
        out[f">{BUCKETS_MS[-1]}"] = 0
        for v in self.values():
            i = bisect.bisect_left(BUCKETS_MS, v)
            if i == len(BUCKETS_MS):
                out[f">{BUCKETS_MS[-1]}"] += 1
            else:
                out[f"<={BUCKETS_MS[i]}"] += 1
        return out


class _CategoryMetrics:
    __slots__ = ("counters", "latency", "slot_wait", "last_samples")

    def __init__(self) -> None:
        self.counters: dict[str, int] = {cls: 0 for cls in STATUS_CLASSES}
        self.latency = _RollingSamples()
        self.slot_wait = _RollingSamples()
        self.last_samples: deque[dict[str, Any]] = deque(maxlen=50)


_LOCK = threading.Lock()
_CATEGORIES: dict[str, _CategoryMetrics] = defaultdict(_CategoryMetrics)
_STARTED_AT = time.monotonic()


def record(
    category: str,
    *,
    status_code: int,
    latency_ms: float,
    slot_wait_ms: float = 0.0,
    request_id: str | None = None,
    extra: dict[str, Any] | None = None,
    disconnected: bool = False,
    slot_busy: bool = False,
) -> None:
    """Record a single completed request."""
    if not config.is_enabled("metrics"):
        return
    cls = _classify(status_code, slot_busy=slot_busy, disconnected=disconnected)
    with _LOCK:
        cm = _CATEGORIES[category]
        cm.counters[cls] += 1
        # Don't pollute latency/slot_wait with non-success classes that
        # would skew p99 — but DO record them so the operator can see
        # whether errors are quick or slow.
        cm.latency.add(latency_ms)
        if slot_wait_ms > 0:
            cm.slot_wait.add(slot_wait_ms)
        cm.last_samples.append({
            "ts": time.time(),
            "request_id": request_id,
            "status": status_code,
            "class": cls,
            "latency_ms": round(latency_ms, 2),
            "slot_wait_ms": round(slot_wait_ms, 2),
            "extra": extra or {},
        })




def snapshot() -> dict[str, Any]:
    """JSON-serialisable snapshot for `GET /admin/api/metrics`.

    The wire-format authority is the `MetricsSnapshot` Pydantic schema
    in `aistack.api._schemas`. The shape is:

        {
          "uptime_sec":  float,    # process uptime
          "window_sec":  int,      # rolling-window duration
          "categories": {
              "<asr|llm|tts>": {   # only categories with traffic
                  "total":              int,
                  "by_class":           {<status_class>: count, ...},
                  "error_count":        int,
                  "error_rate":         float,
                  "slot_503":           int,
                  "disconnected":       int,
                  "throughput_per_min": float,
                  "latency_ms":         {p50, p95, p99, max, samples, histogram},
                  "slot_wait_ms":       {p50, p95, p99, samples},
                  "recent":             [<last 50 sample dicts>],
              },
              ...
          }
        }

    Caller is responsible for serialising to JSON; this function
    returns plain Python types.
    """
    with _LOCK:
        out: dict[str, Any] = {
            "uptime_sec": round(time.monotonic() - _STARTED_AT, 1),
            "window_sec": config.METRICS_WINDOW_SEC,
            "categories": {},
        }
        for cat, cm in _CATEGORIES.items():
            total = sum(cm.counters.values())
            errors = cm.counters["4xx"] + cm.counters["5xx"]
            slot_503 = cm.counters["503-busy"]
            disc = cm.counters["client-disconnect"]
            out["categories"][cat] = {
                "total": total,
                "by_class": dict(cm.counters),
                "error_count": errors,
                "error_rate": (errors / total) if total else 0.0,
                "slot_503": slot_503,
                "disconnected": disc,
                "throughput_per_min": _qpm(cm.latency),
                "latency_ms": {
                    "p50": round(cm.latency.percentile(50), 1),
                    "p95": round(cm.latency.percentile(95), 1),
                    "p99": round(cm.latency.percentile(99), 1),
                    "max": round(max(cm.latency.values(), default=0.0), 1),
                    "samples": len(cm.latency.values()),
                    "histogram": cm.latency.histogram(),
                },
                "slot_wait_ms": {
                    "p50": round(cm.slot_wait.percentile(50), 1),
                    "p95": round(cm.slot_wait.percentile(95), 1),
                    "p99": round(cm.slot_wait.percentile(99), 1),
                    "samples": len(cm.slot_wait.values()),
                },
                "recent": list(cm.last_samples),
            }
        return out


def _qpm(samples: _RollingSamples) -> float:
    """Approx requests-per-minute over the rolling window."""
    n = len(samples.values())
    if n == 0:
        return 0.0
    span_min = max(config.METRICS_WINDOW_SEC / 60.0, 1.0 / 60.0)
    return round(n / span_min, 2)


def reset() -> None:
    """For tests."""
    global _STARTED_AT
    with _LOCK:
        _CATEGORIES.clear()
        _STARTED_AT = time.monotonic()
