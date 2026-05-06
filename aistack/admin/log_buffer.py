"""In-memory ring buffer for aistack log lines.

A logging.Handler that keeps the most recent N formatted log records
in a deque, so /admin can tail them without touching the filesystem.
The buffer is process-local and lost on restart — intentional, this
is for live monitoring not for audit.

Capacity is fixed at module import; bumping it requires a restart.
500 lines × ~120 chars ≈ 60 KB resident, negligible.
"""
from __future__ import annotations

import logging
import threading
from collections import deque

CAPACITY = 500

_buffer: deque[str] = deque(maxlen=CAPACITY)
_lock = threading.Lock()


class RingBufferHandler(logging.Handler):
    """Append every formatted record to the shared ring buffer."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:
            self.handleError(record)
            return
        with _lock:
            _buffer.append(line)


def install(logger: logging.Logger) -> RingBufferHandler:
    """Attach a RingBufferHandler to the given logger if not already present."""
    for existing in logger.handlers:
        if isinstance(existing, RingBufferHandler):
            return existing
    h = RingBufferHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    h.setLevel(logging.INFO)
    logger.addHandler(h)
    return h


def tail(n: int = 100) -> list[str]:
    """Return the last N buffered lines, oldest first."""
    with _lock:
        if n >= len(_buffer):
            return list(_buffer)
        return list(_buffer)[-n:]


def clear() -> None:
    """Drop all buffered lines. Intended for tests."""
    with _lock:
        _buffer.clear()
