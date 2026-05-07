"""X-Request-ID middleware: passthrough, generation, response echo."""
from __future__ import annotations

import re

from fastapi.testclient import TestClient

from aistack.main import app

_HEX16 = re.compile(r"^[0-9a-f]{16}$")


def test_generates_when_missing():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        rid = r.headers.get("X-Request-ID")
        assert rid is not None
        assert _HEX16.match(rid)


def test_echoes_caller_supplied_id():
    with TestClient(app) as client:
        r = client.get("/health", headers={"X-Request-ID": "videocraft-job-42"})
        assert r.headers.get("X-Request-ID") == "videocraft-job-42"


def test_replaces_garbage_id():
    with TestClient(app) as client:
        r = client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
        rid = r.headers.get("X-Request-ID")
        assert rid != "bad id with spaces"
        assert _HEX16.match(rid)


def test_ids_unique_across_requests():
    with TestClient(app) as client:
        r1 = client.get("/health")
        r2 = client.get("/health")
        assert r1.headers["X-Request-ID"] != r2.headers["X-Request-ID"]


def test_overlong_id_replaced():
    with TestClient(app) as client:
        r = client.get("/health", headers={"X-Request-ID": "a" * 1024})
        rid = r.headers.get("X-Request-ID")
        assert _HEX16.match(rid)
