"""Cross-provider model cache with idle-timeout eviction.

Every ASR provider (faster_whisper / parakeet / sensevoice) used to keep
its own per-module dict of loaded models. Combined with three providers'
~4 GB total weights and no eviction, a single comparison test could pin
all three resident for the lifetime of the process.

This module replaces those per-provider dicts with one shared, thread-safe
LRU-by-idle-timestamp cache. A daemon thread scans every minute and drops
entries that haven't been used for KEEP_ALIVE_SEC. On eviction we trigger
gc.collect() and torch.cuda.empty_cache() so RAM/VRAM is actually freed
rather than waiting for Python's reference counting to catch up.

Configuration via env vars (set before service start):
    AISTACK_MODEL_KEEP_ALIVE_SEC   default 300  (5 min)
    AISTACK_MODEL_SCAN_INTERVAL_SEC default 60  (1 min)

The eviction thread is started lazily on first put().
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger("aistack.cache")

KEEP_ALIVE_SEC = float(os.environ.get("AISTACK_MODEL_KEEP_ALIVE_SEC", "300"))
SCAN_INTERVAL_SEC = float(os.environ.get("AISTACK_MODEL_SCAN_INTERVAL_SEC", "60"))

# {(provider, key) -> {"model": obj, "last_used": monotonic_seconds}}
_CACHE: dict[tuple, dict] = {}
_LOCK = threading.RLock()
_EVICTOR_STARTED = False


def _hashable(key: Any) -> Any:
    """Coerce a cache key to something usable as a dict key."""
    if isinstance(key, (str, int, float, bool, tuple)) or key is None:
        return key
    return str(key)


def get(provider: str, key: Any) -> Any | None:
    """Return cached model and bump its last_used timestamp; None if missing."""
    composite = (provider, _hashable(key))
    with _LOCK:
        entry = _CACHE.get(composite)
        if entry is None:
            return None
        entry["last_used"] = time.monotonic()
        return entry["model"]


def put(provider: str, key: Any, model: Any) -> None:
    """Record a freshly-loaded model. Starts the evictor on first call."""
    composite = (provider, _hashable(key))
    with _LOCK:
        _CACHE[composite] = {"model": model, "last_used": time.monotonic()}
    _ensure_evictor()


def evict_idle(now: float | None = None) -> int:
    """Drop entries idle longer than KEEP_ALIVE_SEC. Returns count evicted."""
    if now is None:
        now = time.monotonic()
    evicted: list[tuple] = []
    with _LOCK:
        for composite in list(_CACHE.keys()):
            entry = _CACHE[composite]
            idle = now - entry["last_used"]
            if idle > KEEP_ALIVE_SEC:
                del _CACHE[composite]
                evicted.append(composite)
    if evicted:
        gc.collect()
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        for provider, key in evicted:
            logger.info(
                "evicted idle model provider=%s key=%s (idle > %ss)",
                provider, key, int(KEEP_ALIVE_SEC),
            )
    return len(evicted)


def stats() -> dict:
    """Snapshot of cache state — useful for diagnostics endpoints."""
    now = time.monotonic()
    with _LOCK:
        items = []
        for (provider, key), entry in _CACHE.items():
            items.append({
                "provider": provider,
                "key": str(key),
                "idle_sec": int(now - entry["last_used"]),
            })
    return {
        "keep_alive_sec": int(KEEP_ALIVE_SEC),
        "scan_interval_sec": int(SCAN_INTERVAL_SEC),
        "loaded": items,
    }


def _ensure_evictor() -> None:
    global _EVICTOR_STARTED
    with _LOCK:
        if _EVICTOR_STARTED:
            return
        _EVICTOR_STARTED = True
    t = threading.Thread(
        target=_evictor_loop,
        daemon=True,
        name="aistack-cache-evict",
    )
    t.start()
    logger.info(
        "model cache evictor started: keep_alive=%ss scan=%ss",
        int(KEEP_ALIVE_SEC), int(SCAN_INTERVAL_SEC),
    )


def _evictor_loop() -> None:
    while True:
        try:
            time.sleep(SCAN_INTERVAL_SEC)
            evict_idle()
        except Exception:
            logger.exception("evictor loop iteration failed; continuing")
