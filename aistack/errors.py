"""Structured error contract for aistack providers.

Slimmed down from VideoCraft's `core.ai.errors`. Server-side providers
raise `AIError(Kind.X, provider, message)`; the FastAPI layer converts
them into HTTP responses with a stable JSON envelope so clients can
branch on `error.kind` without parsing free-form messages.
"""

from __future__ import annotations

from enum import Enum


class Kind(Enum):
    NETWORK    = "network"       # model download / disk / IO failure (retryable)
    MALFORMED  = "malformed"     # bad input (audio file missing, unsupported format)
    OVERFLOW   = "overflow"      # input too large for the chosen model / VRAM
    CANCELLED  = "cancelled"     # client disconnected mid-request
    UNKNOWN    = "unknown"       # unclassified — surface raw for logs


class AIError(Exception):
    """Structured provider error.

    The HTTP layer uses (kind, provider, message) to build a JSON envelope:
        {"error": {"kind": "...", "provider": "...", "message": "..."}}

    Args:
        kind: one of Kind enum values.
        provider: provider identifier (e.g. "faster-whisper", "parakeet").
        message: human-readable text, safe to surface to clients.
        raw: original exception for server-side logging only.
    """

    def __init__(
        self,
        kind: Kind,
        provider: str,
        message: str,
        raw: Exception | None = None,
    ):
        super().__init__(message)
        self.kind = kind
        self.provider = provider
        self.message = message
        self.raw = raw

    def __str__(self) -> str:
        return f"[{self.kind.value}/{self.provider}] {self.message}"

    def to_envelope(self) -> dict:
        return {
            "error": {
                "kind": self.kind.value,
                "provider": self.provider,
                "message": self.message,
            }
        }


def http_status_for(kind: Kind) -> int:
    """Map a Kind to an HTTP status code for the response."""
    return {
        Kind.MALFORMED: 400,
        Kind.OVERFLOW: 413,
        Kind.CANCELLED: 499,  # nginx convention for client-closed-request
        Kind.NETWORK: 503,
        Kind.UNKNOWN: 500,
    }.get(kind, 500)
