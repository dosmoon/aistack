"""On-disk request/response capture with size+age sweep.

Layout:
    <PAYLOAD_DIR>/<YYYY-MM-DD>/<request_id>/
        meta.json     metadata (route, headers, status, timing, size)
        req.bin       raw request body
        resp.bin      raw response body (truncated if > PAYLOAD_RESP_MAX_BYTES)

Why disk and not a DB:
    - mp3 / audio bytes are big and uninteresting to query
    - replaying a captured request is "ffmpeg -i req.bin" or
      "curl --data-binary @req.bin ..." with zero glue code
    - the sweeper is dumb and reliable

Sweep policy: on startup + every 30 min, evict oldest request directories
until total size <= PAYLOAD_MAX_BYTES AND newest is within PAYLOAD_MAX_DAYS.

Sensitive headers (Authorization, Cookie, X-Api-Key) are scrubbed from
meta.json. We don't promise capture content is safe to share, but we do
promise to not write the obvious credentials.
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from aistack.observability import config

logger = logging.getLogger("aistack.obs.payload")

_REDACT_HEADERS = {"authorization", "cookie", "x-api-key", "proxy-authorization"}

_SWEEPER_STARTED = False
_SWEEPER_LOCK = threading.Lock()


def _redact(headers: dict[str, str]) -> dict[str, str]:
    return {k: ("***" if k.lower() in _REDACT_HEADERS else v) for k, v in headers.items()}


def _request_dir(request_id: str) -> Path:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return config.PAYLOAD_DIR / day / request_id


class CaptureContext:
    """Per-request capture handle. Created by the middleware when payload
    capture is enabled at request start; passed through `request.state`.

    Routes call `set_request_body(bytes)` after they read the body, and
    `append_response(bytes)` as they yield response chunks (for streams).
    The middleware calls `finalize(meta)` at the end.
    """

    __slots__ = ("request_id", "_dir", "_req_path", "_resp_path",
                 "_resp_fh", "_resp_bytes", "_truncated", "_lock", "_disabled")

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._dir: Path | None = None
        self._req_path: Path | None = None
        self._resp_path: Path | None = None
        self._resp_fh = None
        self._resp_bytes = 0
        self._truncated = False
        self._lock = threading.Lock()
        # Captured at request-start; if toggled off mid-flight, we still
        # finish writing this one consistently. Toggling on mid-flight
        # has no effect — capture is a per-request decision.
        self._disabled = False

    def _ensure_dir(self) -> Path:
        if self._dir is None:
            self._dir = _request_dir(self.request_id)
            self._dir.mkdir(parents=True, exist_ok=True)
        return self._dir

    def set_request_body(self, body: bytes, *, content_type: str | None = None) -> None:
        if self._disabled:
            return
        try:
            d = self._ensure_dir()
            self._req_path = d / "req.bin"
            self._req_path.write_bytes(body)
            if content_type:
                (d / ".req.content-type").write_text(content_type, encoding="utf-8")
        except Exception:
            logger.exception("payload: failed to write request body")
            self._disabled = True

    def adopt_request_file(self, src_path: str) -> None:
        """For ASR multipart: file is already on disk. Copy (cheap relative
        to inference; no double-read into Python memory)."""
        if self._disabled:
            return
        try:
            d = self._ensure_dir()
            self._req_path = d / "req.bin"
            shutil.copyfile(src_path, self._req_path)
        except Exception:
            logger.exception("payload: failed to adopt request file %s", src_path)
            self._disabled = True

    def append_response(self, chunk: bytes) -> None:
        if self._disabled or not chunk:
            return
        with self._lock:
            if self._truncated:
                self._resp_bytes += len(chunk)
                return
            try:
                if self._resp_fh is None:
                    d = self._ensure_dir()
                    self._resp_path = d / "resp.bin"
                    self._resp_fh = open(self._resp_path, "wb")
                self._resp_fh.write(chunk)
                self._resp_bytes += len(chunk)
                if self._resp_bytes >= config.PAYLOAD_RESP_MAX_BYTES:
                    self._truncated = True
                    self._resp_fh.close()
                    self._resp_fh = None
            except Exception:
                logger.exception("payload: failed to append response chunk")
                self._disabled = True
                if self._resp_fh is not None:
                    try:
                        self._resp_fh.close()
                    except Exception:
                        pass
                    self._resp_fh = None

    def finalize(self, meta: dict[str, Any]) -> None:
        with self._lock:
            if self._resp_fh is not None:
                try:
                    self._resp_fh.close()
                except Exception:
                    pass
                self._resp_fh = None
        if self._disabled and self._dir is None:
            return
        try:
            d = self._ensure_dir()
            full_meta = {
                **meta,
                "request_id": self.request_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "resp_bytes": self._resp_bytes,
                "resp_truncated": self._truncated,
                "resp_truncate_limit_bytes": config.PAYLOAD_RESP_MAX_BYTES if self._truncated else None,
            }
            if "headers" in full_meta and isinstance(full_meta["headers"], dict):
                full_meta["headers"] = _redact(full_meta["headers"])
            (d / "meta.json").write_text(
                json.dumps(full_meta, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("payload: failed to write meta.json for %s", self.request_id)


def begin(request_id: str) -> CaptureContext | None:
    """Open a capture context if the toggle is on; else None."""
    if not config.is_enabled("payload"):
        return None
    _start_sweeper()
    return CaptureContext(request_id)


# ----- sweeper -----

def _start_sweeper() -> None:
    global _SWEEPER_STARTED
    if _SWEEPER_STARTED:
        return
    with _SWEEPER_LOCK:
        if _SWEEPER_STARTED:
            return
        config.PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
        t = threading.Thread(target=_sweeper_loop, name="aistack-payload-sweep",
                             daemon=True)
        t.start()
        _SWEEPER_STARTED = True


def _sweeper_loop() -> None:
    while True:
        try:
            sweep()
        except Exception:
            logger.exception("payload sweep failed")
        time.sleep(30 * 60)


def _iter_request_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for day_dir in root.iterdir():
        if not day_dir.is_dir():
            continue
        for req_dir in day_dir.iterdir():
            if req_dir.is_dir():
                yield req_dir


def _dir_size(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def usage() -> dict[str, Any]:
    """Bytes + count + earliest day. Cheap-ish (one stat per file)."""
    root = config.PAYLOAD_DIR
    if not root.exists():
        return {"bytes": 0, "count": 0, "earliest": None,
                "max_bytes": config.PAYLOAD_MAX_BYTES,
                "max_days": config.PAYLOAD_MAX_DAYS,
                "root": str(root)}
    total_bytes = 0
    count = 0
    earliest_mtime: float | None = None
    for d in _iter_request_dirs(root):
        count += 1
        total_bytes += _dir_size(d)
        m = d.stat().st_mtime
        if earliest_mtime is None or m < earliest_mtime:
            earliest_mtime = m
    return {
        "bytes": total_bytes,
        "count": count,
        "earliest": (datetime.fromtimestamp(earliest_mtime, timezone.utc).isoformat()
                     if earliest_mtime else None),
        "max_bytes": config.PAYLOAD_MAX_BYTES,
        "max_days": config.PAYLOAD_MAX_DAYS,
        "root": str(root),
    }


def sweep() -> dict[str, int]:
    """Apply age + size limits. Returns counts of removed dirs."""
    root = config.PAYLOAD_DIR
    if not root.exists():
        return {"removed_age": 0, "removed_size": 0}

    now = time.time()
    age_cutoff = now - config.PAYLOAD_MAX_DAYS * 86400

    # Collect dirs sorted oldest-first (mtime).
    dirs: list[tuple[float, Path, int]] = []
    for d in _iter_request_dirs(root):
        try:
            m = d.stat().st_mtime
        except OSError:
            continue
        dirs.append((m, d, _dir_size(d)))
    dirs.sort(key=lambda t: t[0])

    removed_age = 0
    survivors: list[tuple[float, Path, int]] = []
    for m, d, sz in dirs:
        if m < age_cutoff:
            shutil.rmtree(d, ignore_errors=True)
            removed_age += 1
        else:
            survivors.append((m, d, sz))

    total = sum(sz for _, _, sz in survivors)
    removed_size = 0
    while total > config.PAYLOAD_MAX_BYTES and survivors:
        _m, d, sz = survivors.pop(0)
        shutil.rmtree(d, ignore_errors=True)
        total -= sz
        removed_size += 1

    # Drop emptied day dirs.
    for day_dir in list(root.iterdir()):
        if day_dir.is_dir() and not any(day_dir.iterdir()):
            try:
                day_dir.rmdir()
            except OSError:
                pass

    if removed_age or removed_size:
        logger.info("payload sweep: removed %d (age) + %d (size)",
                    removed_age, removed_size)
    return {"removed_age": removed_age, "removed_size": removed_size}


def clear_all() -> int:
    """Wipe everything. Used by admin's [清空] button."""
    root = config.PAYLOAD_DIR
    if not root.exists():
        return 0
    n = 0
    for d in list(_iter_request_dirs(root)):
        shutil.rmtree(d, ignore_errors=True)
        n += 1
    for day_dir in list(root.iterdir()):
        if day_dir.is_dir() and not any(day_dir.iterdir()):
            try:
                day_dir.rmdir()
            except OSError:
                pass
    return n
