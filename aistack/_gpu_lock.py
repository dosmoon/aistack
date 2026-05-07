"""Single-task GPU lock spanning every aistack capability.

The 8 GB VRAM on consumer hardware (e.g. RTX 4060 Laptop) cannot hold
multiple inference workloads with their workspaces simultaneously.
aistack's stated mission is to act as the local AI capability gateway
that owns GPU scheduling — so this lock is process-wide and shared by
ASR (in-process), LLM proxy (Ollama), and TTS proxy (Qwen3-TTS Docker).
At most one capability holds the slot at a time; overflow returns HTTP
503 + Retry-After and the caller decides whether to back off.

Why one lock for everything (not per-capability):
    Concurrent ASR + LLM on 8 GB → OOM. Concurrent ASR + TTS-container
    inference → OOM. The proxies don't run inference in-process but the
    upstream they forward to does, and from a VRAM perspective it's the
    same GPU. The lock represents "the GPU is currently doing inference
    work for somebody" rather than "this Python process is doing CUDA
    work right now."

Two acquire APIs:
    busy_or_503        context manager. For sync work in to_thread (ASR).
    try_acquire/release manual pair. For async streaming proxies (LLM/TTS)
                       where the lock must be held across yields and
                       released in the stream's finally clause.

Both APIs use the same underlying threading.Lock so the choice of API
doesn't change the mutex semantics — it's only about whether the caller
can use a `with` block or needs explicit release.

Cross-process contention beyond the inference slot (e.g. Qwen3-TTS
container's resident VRAM reservation set by gpu_memory_utilization)
is configuration, not scheduling — see docker/tts_qwen3/ for tuning.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

from fastapi import HTTPException

_LOCK = threading.Lock()
_HOLDER: str | None = None


def _busy_exception(category: str, retry_after_sec: int) -> HTTPException:
    """Raise HTTPException whose detail is the standard error envelope.

    Pre-2026-05-07 this used a bare string detail, which caused the
    response body to be `{"detail": "..."}` — inconsistent with every
    other aistack error path that returns `{"error": {kind, provider,
    message}}`. The integration guide (§8) called this out as a
    documented carve-out clients had to special-case. main.py installs
    a custom HTTPException handler that detects envelope-shaped detail
    and emits it unwrapped, so the response body is the standard
    envelope shape.

    Kind = "network" because the slot-busy state is a transport-level
    back-pressure signal — the server is healthy and the client should
    retry after the Retry-After hint, identical client behavior to a
    transient network error.
    """
    holder = _HOLDER or "another request"
    return HTTPException(
        status_code=503,
        detail={
            "error": {
                "kind": "network",
                "provider": "aistack",
                "message": (
                    f"aistack GPU slot is busy (held by {holder}); "
                    f"rejected {category}. Retry after a few seconds."
                ),
            }
        },
        headers={"Retry-After": str(retry_after_sec)},
    )


def _emit_slot_busy(category: str) -> None:
    """Best-effort metrics hook for slot-rejected requests.

    Slot-busy 503s never reach the route handler, so the observability
    middleware's regular path doesn't see them as "asr" / "llm" / "tts"
    events — they look like ordinary 503s. Notify the metrics module
    here so the dashboard can show "rejected requests" separately from
    real errors. Lazy-imported to avoid a circular import at module load.
    """
    try:
        from aistack.observability import metrics as _obs_metrics
        _obs_metrics.record_slot_busy(category)
    except Exception:
        pass


@contextmanager
def busy_or_503(category: str = "asr", retry_after_sec: int = 5):
    """Acquire the GPU slot or raise HTTP 503. Sync context manager."""
    global _HOLDER
    if not _LOCK.acquire(blocking=False):
        _emit_slot_busy(category)
        raise _busy_exception(category, retry_after_sec)
    _HOLDER = category
    try:
        yield
    finally:
        _HOLDER = None
        _LOCK.release()


def try_acquire_or_503(category: str, retry_after_sec: int = 5) -> None:
    """Manual acquire for async streaming handlers.

    Pairs with `release()`. The caller MUST release in a finally clause,
    typically inside the StreamingResponse body iterator's finally so
    the slot stays held until the last byte is sent (or the client
    disconnects). Raises HTTPException(503) on contention — let FastAPI
    render it.
    """
    global _HOLDER
    if not _LOCK.acquire(blocking=False):
        _emit_slot_busy(category)
        raise _busy_exception(category, retry_after_sec)
    _HOLDER = category


def release() -> None:
    """Release a slot acquired via try_acquire_or_503. Idempotent-safe in
    finally clauses: a double-release on an unlocked lock raises
    RuntimeError, so callers must pair acquire/release exactly once."""
    global _HOLDER
    _HOLDER = None
    _LOCK.release()


def is_busy() -> bool:
    """Snapshot of lock state — useful for diagnostics endpoints."""
    return _LOCK.locked()


def current_holder() -> str | None:
    """Which capability currently holds the slot, or None."""
    return _HOLDER
