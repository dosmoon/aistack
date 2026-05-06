"""Single-task GPU lock for aistack ASR endpoints.

The 8 GB VRAM on consumer hardware (e.g. RTX 4060 Laptop) cannot hold
multiple ASR backends with their inference workspaces simultaneously.
Concurrent requests would race for VRAM and OOM the worker.

Strategy: at most one ASR inference at a time, reject overflow with HTTP
503 + Retry-After. No queue — callers are expected to retry. Pairs with
the cache module's hot-swap-on-mismatch policy: a request that targets a
different model than the one currently resident triggers eviction +
load before it acquires the lock.

The lock is process-wide (threading.Lock, not asyncio.Lock) because the
actual heavy work runs in asyncio.to_thread workers. Only the ASR
endpoint guards against concurrent inference; TTS is a transparent
proxy to the Qwen3-TTS Docker sidecar and does no in-process GPU work,
so it is intentionally excluded.

Cross-process contention (Ollama, Qwen3-TTS Docker) is out of scope
here — those run in their own GPU contexts and must coordinate
externally (short keep_alive on Ollama side, sequential workflow at
the orchestration layer).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

from fastapi import HTTPException

_LOCK = threading.Lock()


@contextmanager
def busy_or_503(category: str = "asr", retry_after_sec: int = 5):
    """Acquire the GPU slot or raise HTTP 503.

    Non-blocking: if another request is mid-inference the call returns
    immediately with 503 and a Retry-After header. The caller (e.g.
    VideoCraft) decides whether to back off and retry.
    """
    if not _LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail=(
                f"aistack {category} backend is busy serving another request. "
                f"Retry after a few seconds."
            ),
            headers={"Retry-After": str(retry_after_sec)},
        )
    try:
        yield
    finally:
        _LOCK.release()


def is_busy() -> bool:
    """Snapshot of lock state — useful for diagnostics endpoints."""
    return _LOCK.locked()
