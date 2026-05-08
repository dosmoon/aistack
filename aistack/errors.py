"""Structured error contract for aistack providers.

Slimmed down from VideoCraft's `core.ai.errors`. Server-side providers
raise `AIError(Kind.X, provider, message)`; the FastAPI layer converts
them into HTTP responses with a stable JSON envelope so clients can
branch on `error.kind` without parsing free-form messages.

The wire format is documented authoritatively by the Pydantic schemas
in `aistack.api._schemas` (ErrorBody / ErrorEnvelope); this module is
the source of truth for the *taxonomy* (which kinds exist, what each
means, what HTTP status each maps to).
"""

from __future__ import annotations

from enum import Enum


class Kind(Enum):
    """Stable, machine-readable error classification.

    Consumers branch on `error.kind` (this enum's `.value`) to decide
    how to handle a non-2xx response without parsing free-form
    `error.message` text. The set is deliberately small — five
    categories that map one-to-one onto distinct consumer reactions.

    | kind        | maps to HTTP | typical cause                              | consumer reaction      |
    |-------------|--------------|--------------------------------------------|------------------------|
    | `network`   | 503          | model download / disk / IO / upstream down | retry with backoff     |
    | `malformed` | 400          | invalid input (bad audio, unknown model)   | fix request, do not retry |
    | `overflow`  | 413          | input exceeds chosen model's VRAM budget   | shrink input or pick smaller model |
    | `cancelled` | 499          | client disconnected mid-request            | already abandoned; informational |
    | `unknown`   | 500          | unclassified server-side failure           | log + retry once       |

    The mapping is enforced by `http_status_for()` below. Adding new
    kinds is an additive `/v1` change; renaming or removing requires
    `/v2`.
    """

    NETWORK = "network"
    MALFORMED = "malformed"
    OVERFLOW = "overflow"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class AIError(Exception):
    """Structured provider error.

    Server-side code raises this exception; the FastAPI layer converts
    it into an HTTP response with a stable JSON envelope:

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
        """Serialize to the standard error envelope dict.

        Returns a JSON-ready dict matching the ErrorEnvelope Pydantic
        schema (aistack.api._schemas.ErrorEnvelope). The shape is fixed:

            {"error": {"kind": <Kind.value>,
                       "provider": <str>,
                       "message": <str>}}

        The FastAPI layer wraps the result in a JSONResponse with the
        status code computed by `http_status_for(self.kind)`.
        """
        return {
            "error": {
                "kind": self.kind.value,
                "provider": self.provider,
                "message": self.message,
            }
        }


def http_status_for(kind: Kind) -> int:
    """Map a Kind to its HTTP status code.

    The mapping is the canonical contract — clients may rely on it
    without inspecting `error.kind`, though branching on `kind` is
    recommended because status codes carry less semantic precision
    (e.g. 503 alone does not distinguish "Ollama is down" from "GPU
    slot busy, retry shortly").

    | Kind        | status | rationale                                    |
    |-------------|--------|----------------------------------------------|
    | MALFORMED   | 400    | client error — bad input                     |
    | OVERFLOW    | 413    | payload too large for the chosen processor   |
    | CANCELLED   | 499    | nginx convention — client closed request     |
    | NETWORK     | 503    | service unavailable / temporarily unreachable |
    | UNKNOWN     | 500    | server error                                 |

    Any kind not in the table maps to 500 (defensive default for
    forward compatibility).
    """
    return {
        Kind.MALFORMED: 400,
        Kind.OVERFLOW: 413,
        Kind.CANCELLED: 499,  # nginx convention for client-closed-request
        Kind.NETWORK: 503,
        Kind.UNKNOWN: 500,
    }.get(kind, 500)
