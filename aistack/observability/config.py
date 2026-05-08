"""Observability runtime toggles.

Static fields (paths, size budgets, defaults) come from
`aistack.config.config.observability`; this module owns just the
mutable runtime toggle dict that admin POST flips at runtime. Restart
returns to env defaults — by design, matches the rest of aistack's
"env is the durable knob" style and avoids a settings.json philosophy
split.

See docs/configuration.md for the full env list.
"""
from __future__ import annotations

import threading
from pathlib import Path

from aistack.config import config

# ── Static fields, re-exported for backwards-compatible call sites ──────────
LOG_DIR: Path = config.observability.log_dir
PAYLOAD_DIR: Path = config.observability.payload_dir
PAYLOAD_MAX_BYTES: int = config.observability.payload_max_bytes
PAYLOAD_MAX_DAYS: int = config.observability.payload_max_days
METRICS_WINDOW_SEC: int = config.observability.metrics_window_sec
PAYLOAD_RESP_MAX_BYTES: int = config.observability.payload_resp_max_bytes


# ── Runtime-mutable toggles (admin can flip these at runtime) ──────────────
_LOCK = threading.Lock()
_TOGGLES: dict[str, bool] = {
    "metrics": config.observability.metrics_default,
    "access_log": config.observability.access_log_default,
    "payload": config.observability.payload_default,
}


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
