"""Pydantic schemas for the public HTTP API.

These models exist primarily to drive the auto-generated OpenAPI spec
(which feeds both FastAPI's /docs Swagger UI and the build-time markdown
generator at scripts/gen_api_reference.py). They are the single source
of truth for request/response shapes; the markdown reference docs are
re-generated from these models on every build.

Where a route's actual handler returns a plain dict for performance or
flexibility (e.g. ASR's three response_format variants) the Pydantic
model still appears in the OpenAPI `responses` block so the schema is
exposed without forcing dict→model serialization on the hot path.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ── Errors ──────────────────────────────────────────────────────────────────

class ErrorBody(BaseModel):
    """Inner error object shared by every aistack error response.

    Consumers branch on `kind` (machine-readable, stable enum) and
    surface `message` (human-readable, safe to display). The `provider`
    field identifies which subsystem produced the error so logs can be
    filtered by backend.
    """

    kind: Literal[
        "network", "malformed", "overflow", "cancelled", "unknown"
    ] = Field(
        description=(
            "Stable machine-readable error class. Mapping to HTTP status: "
            "malformed=400, overflow=413, cancelled=499, network=503, "
            "unknown=500."
        ),
    )
    provider: str = Field(
        description=(
            "Identifier of the subsystem that raised the error: 'aistack' "
            "for gateway-level failures, 'Faster-Whisper' / 'Parakeet' / "
            "'SenseVoice' for ASR backends, 'Qwen3-TTS' for TTS, 'ollama' "
            "for the LLM proxy."
        ),
        examples=["aistack", "Faster-Whisper", "ollama"],
    )
    message: str = Field(
        description="Human-readable description, safe to display to end users.",
    )


class ErrorEnvelope(BaseModel):
    """Wire format for every non-2xx response from aistack.

    The shape is identical regardless of which endpoint produced the
    error, so consumers can write one error-handling helper and reuse it
    across capabilities.
    """

    error: ErrorBody


# ── Transcription (ASR) ─────────────────────────────────────────────────────

class TranscriptionWord(BaseModel):
    """Single word with timestamps. Present in verbose_json responses
    when the backend supports word-level timing (faster-whisper does;
    Parakeet and SenseVoice produce these via aistack's word-stitching
    pipeline)."""

    start: float = Field(description="Word start time in seconds from audio origin.")
    end: float = Field(description="Word end time in seconds from audio origin.")
    word: str = Field(description="The word text. Punctuation may be attached or stripped depending on the backend.")


class TranscriptionSegment(BaseModel):
    """One segment of a transcription. Granularity depends on the
    `segment_granularity` request parameter and the backend's native
    segmentation strategy."""

    id: int = Field(description="Zero-based segment index within the response.")
    start: float = Field(description="Segment start time in seconds.")
    end: float = Field(description="Segment end time in seconds.")
    text: str = Field(description="Segment text content.")
    words: list[TranscriptionWord] | None = Field(
        default=None,
        description="Word-level timestamps within this segment, when available.",
    )
    avg_logprob: float | None = Field(
        default=None,
        description="Mean per-token log probability. Only emitted by faster-whisper. Useful for filtering low-confidence segments.",
    )
    no_speech_prob: float | None = Field(
        default=None,
        description="Probability that this segment contains no speech. Only emitted by faster-whisper.",
    )
    compression_ratio: float | None = Field(
        default=None,
        description="Decoded-text compression ratio. Only emitted by faster-whisper. Values > 2.4 typically indicate hallucinated repetition.",
    )
    temperature: float | None = Field(
        default=None,
        description="Sampling temperature used for this segment. Only emitted by faster-whisper.",
    )


class TranscriptionResponse(BaseModel):
    """Response shape for `POST /v1/audio/transcriptions`.

    The shape is unified across the `json` and `verbose_json`
    response_format variants; which fields are populated depends on the
    request:

    - `response_format=json` (default, OpenAI minimal) → only `text`
    - `response_format=verbose_json` → all fields populated
    - `response_format=text` → not this schema; route returns a plain
      text body (`text/plain` content type)
    - `stream=true` → not this schema; route returns Server-Sent Events
      with `transcript.text.delta` events and a final
      `transcript.text.done`

    The verbose-only fields are typed Optional so the same schema can
    represent both shapes; consumers branching on `response_format`
    know which fields to expect.
    """

    model_config = ConfigDict(json_schema_extra={"example": {
        "language": "en",
        "duration": 142.7,
        "text": "Welcome to the show. Today we'll talk about ...",
        "segments": [
            {"id": 0, "start": 0.0, "end": 4.1, "text": "Welcome to the show.",
             "words": [{"start": 0.0, "end": 0.6, "word": "Welcome"}]},
        ],
        "words": [{"start": 0.0, "end": 0.6, "word": "Welcome"}],
    }})

    text: str = Field(description="The full transcribed text. Populated for both json and verbose_json formats.")
    language: str | None = Field(
        default=None,
        description="ISO 639-1 code of the detected (or hinted) source language. Verbose only.",
    )
    duration: float | None = Field(
        default=None,
        description="Audio duration in seconds, from the backend's own measurement. Verbose only.",
    )
    segments: list[TranscriptionSegment] | None = Field(
        default=None,
        description="Per-segment breakdown. Segmentation strategy depends on `segment_granularity` and the backend. Verbose only.",
    )
    words: list[TranscriptionWord] | None = Field(
        default=None,
        description="Flat per-word timestamp list. Convenient for clients that want word timing without traversing segments. Verbose only.",
    )


# ── Models inventory ────────────────────────────────────────────────────────

class ModelEntry(BaseModel):
    """One entry in the `/v1/models` inventory.

    OpenAI-compatible base fields (id, object, owned_by) are augmented
    with aistack extension fields (capabilities, languages,
    supports_streaming, is_routing_alias) so consumers can build
    capability-aware pickers without per-backend lookup tables.
    """

    id: str = Field(
        description="Model identifier. May be a HuggingFace model id, a Whisper size alias, an Ollama tag, or the routing alias 'auto'.",
        examples=["whisper-large-v3", "nvidia/parakeet-tdt-0.6b-v3", "qwen2.5:14b", "auto"],
    )
    object: Literal["model"] = Field(default="model", description="Always 'model' (OpenAI compatibility).")
    owned_by: str = Field(
        description="Origin of the model weights or routing decision.",
        examples=["openai", "nvidia", "alibaba", "qwen", "ollama", "aistack"],
    )
    capabilities: list[Literal["asr", "tts", "llm"]] = Field(
        description="Which gateway capabilities this entry can serve. Single-element list in current versions; reserved as a list for future multi-capability entries.",
    )
    languages: list[str] | None = Field(
        default=None,
        description="ISO 639-1 codes the backend can transcribe. ASR-only field; absent on TTS / LLM entries.",
    )
    supports_streaming: bool | None = Field(
        default=None,
        description="True when stream=true on this model produces real incremental SSE output. False means the gateway accepts stream=true but emits a single-event SSE response with a warning. Absent on entries where streaming is irrelevant.",
    )
    is_routing_alias: bool | None = Field(
        default=None,
        description="True for the 'auto' entry (routing decision, not a real model).",
    )


class ModelsList(BaseModel):
    """Response shape for `GET /v1/models` — OpenAI-compatible."""

    object: Literal["list"] = Field(default="list", description="Always 'list' (OpenAI compatibility).")
    data: list[ModelEntry] = Field(description="One entry per servable model + the 'auto' routing alias when at least one ASR backend is reachable.")


# ── Health ──────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response shape for `GET /health`."""

    status: Literal["ok"] = Field(description="Always 'ok' when this endpoint responds. Connection refused / non-200 means the gateway is down.")
    version: str = Field(description="aistack package version (PEP 440).", examples=["0.0.1"])
