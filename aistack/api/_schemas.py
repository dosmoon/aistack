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


# ── Observability ───────────────────────────────────────────────────────────

class AccessLogRecord(BaseModel):
    """One line of the JSONL access log (`<LOG_DIR>/access-YYYY-MM-DD.jsonl`).

    Every observed HTTP request appends one of these to today's log.
    The category filter in `aistack.observability.middleware._category_for`
    keeps `/health` / `/v1/models` / `/admin/*` out of the log so it
    doesn't drown in noise.
    """

    ts: str = Field(description="ISO 8601 timestamp (UTC, millisecond precision) of when the response completed.")
    request_id: str = Field(description="Short hex correlation id, also returned to the client as the X-Request-ID header.")
    method: str = Field(description="HTTP method.", examples=["POST", "GET"])
    path: str = Field(description="Request path (no query string).", examples=["/v1/audio/transcriptions"])
    query: str | None = Field(default=None, description="Query string without the leading '?', or null if absent.")
    status: int = Field(description="HTTP status code returned to the client.")
    category: Literal["asr", "llm", "tts"] = Field(
        description="Capability category, derived from path prefix.",
    )
    model: str | None = Field(default=None, description="Canonical model id served by the request (when known).")
    latency_ms: float = Field(description="End-to-end request duration in milliseconds, measured at the ASGI layer.")
    slot_wait_ms: float = Field(description="Time spent waiting on the global GPU slot before the inference started, in ms. Zero if not measured.")
    client: str | None = Field(default=None, description="Client address as 'ip:port', or null if not exposed by ASGI.")
    extra: dict | None = Field(
        default=None,
        description="Per-route extras (e.g. ASR audio_sec / language / response_format / segment_granularity; LLM message_count / stream).",
    )


class MetricsLatencyStats(BaseModel):
    """Latency distribution for one capability category."""

    p50: float = Field(description="50th percentile (median) latency in milliseconds.")
    p95: float = Field(description="95th percentile latency in milliseconds.")
    p99: float = Field(description="99th percentile latency in milliseconds.")
    max: float = Field(description="Maximum observed latency in milliseconds within the rolling window.")
    samples: int = Field(description="Number of samples in the rolling window the percentiles were computed from.")
    histogram: dict[str, int] = Field(
        description="Coarse histogram with power-of-2 buckets (`<=10`, `<=25`, ... `<=60000`, `>60000`). Counts samples per bucket.",
    )


class MetricsSlotWaitStats(BaseModel):
    """GPU-slot wait distribution for one capability category."""

    p50: float = Field(description="50th percentile slot wait in milliseconds.")
    p95: float = Field(description="95th percentile slot wait in milliseconds.")
    p99: float = Field(description="99th percentile slot wait in milliseconds.")
    samples: int = Field(description="Number of recorded slot waits in the rolling window.")


class MetricsRecentSample(BaseModel):
    """One entry in the per-category rolling tail of recent requests.

    Capped at the last 50 samples per category. Useful for spot-checks
    of "what just happened" without sifting the full access log.
    """

    ts: float = Field(description="Unix timestamp (seconds, with sub-second precision) of when the request completed.")
    request_id: str | None = Field(default=None, description="Short hex correlation id (matches the access log entry).")
    status: int = Field(description="HTTP status code returned to the client.")
    class_: Literal["2xx", "4xx", "5xx", "503-busy", "client-disconnect"] = Field(
        alias="class",
        description=(
            "Status class. '503-busy' is load-shedding (slot mutex rejection — counted separately so a busy gateway "
            "doesn't look broken). 'client-disconnect' is informational, not an error."
        ),
    )
    latency_ms: float = Field(description="End-to-end request duration in milliseconds.")
    slot_wait_ms: float = Field(description="Time waiting for the GPU slot in milliseconds.")
    extra: dict = Field(description="Per-request extras (model id, audio_sec, etc.). Free-form.")


class MetricsCategorySnapshot(BaseModel):
    """Rolling-window metrics for one capability category (asr / llm / tts)."""

    total: int = Field(description="Total requests counted in this category since process start (not windowed).")
    by_class: dict[str, int] = Field(
        description="Counts per status_class. Keys: 2xx / 4xx / 5xx / 503-busy / client-disconnect.",
    )
    error_count: int = Field(description="4xx + 5xx count. Excludes 503-busy and client-disconnect.")
    error_rate: float = Field(description="error_count / total. 0.0 when total is 0.")
    slot_503: int = Field(description="Count of 503 responses caused by GPU slot contention (load-shedding).")
    disconnected: int = Field(description="Count of requests aborted by client disconnect mid-flight.")
    throughput_per_min: float = Field(description="Approximate requests-per-minute over the rolling window.")
    latency_ms: MetricsLatencyStats
    slot_wait_ms: MetricsSlotWaitStats
    recent: list[MetricsRecentSample] = Field(
        description="Last ≤50 requests in this category, newest last.",
    )


class MetricsSnapshot(BaseModel):
    """Response shape for `GET /admin/api/metrics`.

    Built by `aistack.observability.metrics.snapshot()`. Stable across
    `/v1` — adding new categories or new top-level keys is allowed,
    renaming or removing requires a version bump.
    """

    uptime_sec: float = Field(description="Process uptime in seconds.")
    window_sec: int = Field(description="Rolling window duration the percentiles are computed over.")
    categories: dict[str, MetricsCategorySnapshot] = Field(
        description="Per-capability metrics. Keys: 'asr', 'llm', 'tts' (only those that received traffic since startup).",
    )
