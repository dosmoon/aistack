"""Observability feature flags + paths.

env-seeded at import; admin /admin/api/observability/toggle mutates the
process-local dict, not the env. Restart returns to env defaults — by
design, matches the rest of aistack's "env is the durable knob" style
and avoids a settings.json philosophy split.

Three independent toggles:
    AISTACK_OBS_METRICS      on|off  (default on — < 1µs/req cost)
    AISTACK_OBS_ACCESS_LOG   on|off  (default on — JSONL appends)
    AISTACK_OBS_PAYLOAD      on|off  (default off — writes audio bytes to disk)

Storage knobs (read once at import, not runtime-mutable):
    AISTACK_OBS_LOG_DIR              ./logs
    AISTACK_OBS_PAYLOAD_DIR          <HF_HOME>/../aistack_captures (or ./captures)
    AISTACK_OBS_PAYLOAD_MAX_GB       5
    AISTACK_OBS_PAYLOAD_MAX_DAYS     7
    AISTACK_OBS_METRICS_WINDOW_MIN   60   (rolling histogram window)
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

_TRUTHY = {"1", "on", "true", "yes", "y"}
_FALSY = {"0", "off", "false", "no", "n", ""}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in _TRUTHY:
        return True
    if v in _FALSY:
        return False
    return default


def _default_payload_dir() -> Path:
    explicit = os.environ.get("AISTACK_OBS_PAYLOAD_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    hf = os.environ.get("HF_HOME")
    if hf:
        return (Path(hf).expanduser().resolve().parent / "aistack_captures")
    return Path.cwd() / "captures"


def _default_log_dir() -> Path:
    return Path(os.environ.get("AISTACK_OBS_LOG_DIR", "./logs")).expanduser().resolve()


# Mutable runtime toggles. Locked because admin POST and request handlers
# can read/write concurrently.
_LOCK = threading.Lock()
_TOGGLES: dict[str, bool] = {
    "metrics": _env_bool("AISTACK_OBS_METRICS", True),
    "access_log": _env_bool("AISTACK_OBS_ACCESS_LOG", True),
    "payload": _env_bool("AISTACK_OBS_PAYLOAD", False),
}

# Path constants — captured once. Changing these requires restart, same
# as AISTACK_MODEL_KEEP_ALIVE_SEC etc.
LOG_DIR: Path = _default_log_dir()
PAYLOAD_DIR: Path = _default_payload_dir()
PAYLOAD_MAX_BYTES: int = int(float(os.environ.get("AISTACK_OBS_PAYLOAD_MAX_GB", "5")) * (1024 ** 3))
PAYLOAD_MAX_DAYS: int = int(os.environ.get("AISTACK_OBS_PAYLOAD_MAX_DAYS", "7"))
METRICS_WINDOW_SEC: int = int(os.environ.get("AISTACK_OBS_METRICS_WINDOW_MIN", "60")) * 60
PAYLOAD_RESP_MAX_BYTES: int = int(float(os.environ.get("AISTACK_OBS_PAYLOAD_RESP_MAX_MB", "50")) * (1024 ** 2))


VALID_KEYS = ("metrics", "access_log", "payload")


def is_enabled(key: str) -> bool:
    with _LOCK:
        return _TOGGLES.get(key, False)


def set_enabled(key: str, value: bool) -> None:
    if key not in VALID_KEYS:
        raise ValueError(f"unknown observability toggle: {key!r}")
    with _LOCK:
        _TOGGLES[key] = bool(value)


def snapshot() -> dict[str, bool]:
    with _LOCK:
        return dict(_TOGGLES)
