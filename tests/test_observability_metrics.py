"""Metrics rolling-histogram and percentile correctness."""
from __future__ import annotations

import threading

import pytest

from aistack.observability import config, metrics


@pytest.fixture(autouse=True)
def _reset_metrics():
    metrics.reset()
    # Force the toggle on regardless of env state so tests don't depend
    # on the host's AISTACK_OBS_METRICS.
    prior = config.is_enabled("metrics")
    config.set_enabled("metrics", True)
    yield
    config.set_enabled("metrics", prior)


def test_records_and_percentiles_match_known_input():
    # 100 evenly-spaced samples so percentiles are unambiguous regardless
    # of the half-rounding rule (banker's vs away-from-zero).
    for ms in range(1, 101):
        metrics.record("asr", status_code=200, latency_ms=ms)
    snap = metrics.snapshot()["categories"]["asr"]
    assert snap["total"] == 100
    assert 49.0 <= snap["latency_ms"]["p50"] <= 51.0
    assert 94.0 <= snap["latency_ms"]["p95"] <= 96.0
    assert 98.0 <= snap["latency_ms"]["p99"] <= 100.0
    assert snap["latency_ms"]["max"] == 100.0
    assert snap["error_count"] == 0
    assert snap["error_rate"] == 0.0


def test_status_classification():
    metrics.record("llm", status_code=200, latency_ms=5)
    metrics.record("llm", status_code=400, latency_ms=2)
    metrics.record("llm", status_code=500, latency_ms=3)
    metrics.record("llm", status_code=200, latency_ms=1, disconnected=True)
    metrics.record("llm", status_code=503, latency_ms=4, slot_busy=True)
    snap = metrics.snapshot()["categories"]["llm"]
    assert snap["by_class"]["2xx"] == 1
    assert snap["by_class"]["4xx"] == 1
    assert snap["by_class"]["5xx"] == 1
    assert snap["by_class"]["client-disconnect"] == 1
    assert snap["by_class"]["503-busy"] == 1
    # 503-busy and client-disconnect are not counted as errors.
    assert snap["error_count"] == 2
    assert snap["slot_503"] == 1


def test_slot_busy_response_detection():
    # A 503 with Retry-After is the slot-busy marker.
    assert metrics.is_slot_busy_response(503, [(b"retry-after", b"5")])
    assert metrics.is_slot_busy_response(503, [(b"Retry-After", b"5")])
    # A 503 without Retry-After is a real upstream failure.
    assert not metrics.is_slot_busy_response(503, [(b"content-type", b"application/json")])
    # Non-503 status never counts as slot-busy.
    assert not metrics.is_slot_busy_response(500, [(b"retry-after", b"5")])


def test_concurrent_record_no_loss():
    N = 500
    THREADS = 10

    def worker():
        for i in range(N):
            metrics.record("tts", status_code=200, latency_ms=10 + i % 90)

    ts = [threading.Thread(target=worker) for _ in range(THREADS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    snap = metrics.snapshot()["categories"]["tts"]
    assert snap["total"] == N * THREADS


def test_disabled_toggle_skips_record():
    config.set_enabled("metrics", False)
    metrics.record("asr", status_code=200, latency_ms=99)
    metrics.record("asr", status_code=503, latency_ms=2, slot_busy=True)
    assert metrics.snapshot()["categories"] == {}


def test_histogram_buckets_present_in_snapshot():
    for ms in (5, 50, 500, 5000):
        metrics.record("asr", status_code=200, latency_ms=ms)
    hist = metrics.snapshot()["categories"]["asr"]["latency_ms"]["histogram"]
    # Exactly one sample in each of these buckets.
    assert hist["<=10"] == 1
    assert hist["<=50"] == 1
    assert hist["<=500"] == 1
    assert hist["<=5000"] == 1
