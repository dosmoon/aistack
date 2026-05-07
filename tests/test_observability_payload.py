"""payload capture: file layout, redaction, sweep policy."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from aistack.observability import config, payload


@pytest.fixture
def tmp_payload_dir(tmp_path, monkeypatch):
    """Repoint payload root at a tmp dir for the duration of the test."""
    monkeypatch.setattr(config, "PAYLOAD_DIR", tmp_path / "captures")
    config.set_enabled("payload", True)
    yield tmp_path / "captures"
    config.set_enabled("payload", False)


def test_capture_writes_request_response_meta(tmp_payload_dir):
    ctx = payload.begin("abc123")
    assert ctx is not None
    ctx.set_request_body(b"hello world", content_type="text/plain")
    ctx.append_response(b"resp ")
    ctx.append_response(b"chunk-2")
    ctx.finalize({
        "method": "POST",
        "path": "/v1/audio/transcriptions",
        "status": 200,
        "headers": {"Authorization": "Bearer secret", "X-Foo": "bar"},
    })
    # Find the day dir.
    day_dirs = list(tmp_payload_dir.iterdir())
    assert len(day_dirs) == 1
    req_dir = day_dirs[0] / "abc123"
    assert (req_dir / "req.bin").read_bytes() == b"hello world"
    assert (req_dir / "resp.bin").read_bytes() == b"resp chunk-2"
    meta = json.loads((req_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == 200
    assert meta["resp_bytes"] == len(b"resp chunk-2")
    # Redaction
    assert meta["headers"]["Authorization"] == "***"
    assert meta["headers"]["X-Foo"] == "bar"


def test_response_truncation(tmp_payload_dir, monkeypatch):
    monkeypatch.setattr(config, "PAYLOAD_RESP_MAX_BYTES", 10)
    ctx = payload.begin("trunc")
    assert ctx is not None
    ctx.append_response(b"0123456789")  # exactly at limit -> still ok
    ctx.append_response(b"OVERFLOW")
    ctx.finalize({"status": 200})
    day = next(tmp_payload_dir.iterdir())
    meta = json.loads((day / "trunc" / "meta.json").read_text(encoding="utf-8"))
    assert meta["resp_truncated"] is True
    # Truncated, but the byte count reflects total observed (not just written).
    assert meta["resp_bytes"] == len(b"0123456789") + len(b"OVERFLOW")


def test_sweep_size_limit(tmp_payload_dir, monkeypatch):
    monkeypatch.setattr(config, "PAYLOAD_MAX_BYTES", 50)
    monkeypatch.setattr(config, "PAYLOAD_MAX_DAYS", 365)  # disable age limit

    for i in range(5):
        ctx = payload.begin(f"req{i}")
        ctx.set_request_body(b"x" * 30)  # each dir ~30 bytes + meta overhead
        ctx.finalize({"status": 200})
        time.sleep(0.02)  # ensure mtime ordering

    before = payload.usage()
    assert before["count"] == 5
    res = payload.sweep()
    after = payload.usage()
    # Some old dirs must have been removed by the size pass.
    assert res["removed_size"] > 0
    assert after["count"] < before["count"]


def test_sweep_age_limit(tmp_payload_dir, monkeypatch):
    monkeypatch.setattr(config, "PAYLOAD_MAX_DAYS", 0)  # everything is too old
    monkeypatch.setattr(config, "PAYLOAD_MAX_BYTES", 1024 ** 4)

    ctx = payload.begin("doomed")
    ctx.set_request_body(b"goodbye")
    ctx.finalize({"status": 200})
    # Force mtime into the past.
    target_dir = next(tmp_payload_dir.iterdir()) / "doomed"
    old = time.time() - 86400
    os.utime(target_dir, (old, old))

    res = payload.sweep()
    assert res["removed_age"] == 1
    assert payload.usage()["count"] == 0


def test_disabled_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PAYLOAD_DIR", tmp_path / "captures")
    config.set_enabled("payload", False)
    assert payload.begin("anything") is None


def test_clear_all(tmp_payload_dir):
    for i in range(3):
        ctx = payload.begin(f"r{i}")
        ctx.set_request_body(b"a")
        ctx.finalize({"status": 200})
    n = payload.clear_all()
    assert n == 3
    assert payload.usage()["count"] == 0
