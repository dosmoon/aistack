"""Daily-rolling JSONL access log.

One line per completed request. Producers enqueue dicts; a single
background daemon thread serialises and flushes — keeps the request
hot path off the disk.

File naming: <LOG_DIR>/access-YYYY-MM-DD.jsonl  (UTC).

Fields are documented in docs/public/api/observability.md. The writer never
raises into the request thread: queue.put() is non-blocking and a full
queue silently drops records (with a one-shot warning) so a wedged
disk doesn't take down inference.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aistack.observability import config

logger = logging.getLogger("aistack.obs.access_log")

_QUEUE: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=10000)
_THREAD: threading.Thread | None = None
_THREAD_LOCK = threading.Lock()
_DROP_WARNED = False


def _ensure_thread_started() -> None:
    global _THREAD
    if _THREAD is not None and _THREAD.is_alive():
        return
    with _THREAD_LOCK:
        if _THREAD is not None and _THREAD.is_alive():
            return
        _THREAD = threading.Thread(target=_writer_loop, name="aistack-access-log",
                                   daemon=True)
        _THREAD.start()


def _current_path() -> Path:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return config.LOG_DIR / f"access-{day}.jsonl"


def _writer_loop() -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    pending: list[str] = []
    last_flush = time.monotonic()
    while True:
        try:
            item = _QUEUE.get(timeout=1.0)
        except queue.Empty:
            item = None
        if item is not None:
            try:
                pending.append(json.dumps(item, ensure_ascii=False, default=str))
            except Exception:
                logger.exception("failed to serialise access log entry")
        now = time.monotonic()
        if pending and (len(pending) >= 100 or now - last_flush >= 1.0):
            try:
                path = _current_path()
                # Open per-flush so day rollover Just Works without bookkeeping.
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write("\n".join(pending))
                    fh.write("\n")
            except Exception:
                logger.exception("failed to flush access log to %s", path)
            pending.clear()
            last_flush = now


def write(record: dict[str, Any]) -> None:
    """Enqueue a record. Non-blocking; drops on overflow."""
    global _DROP_WARNED
    if not config.is_enabled("access_log"):
        return
    _ensure_thread_started()
    try:
        _QUEUE.put_nowait(record)
    except queue.Full:
        if not _DROP_WARNED:
            _DROP_WARNED = True
            logger.warning("access log queue full; dropping records "
                           "(further drops will be silent)")


def flush_for_test(timeout_sec: float = 2.0) -> None:
    """Block until the queue is drained. Test-only."""
    deadline = time.monotonic() + timeout_sec
    while not _QUEUE.empty() and time.monotonic() < deadline:
        time.sleep(0.05)
    # Give the writer a moment to flush its pending list.
    time.sleep(0.2)
