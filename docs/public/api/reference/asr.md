---
title: ASR — speech to text
description: Auto-generated reference for POST /v1/audio/transcriptions. OpenAI Whisper API compatible.
sidebar:
  order: 11
---

<!-- AUTO-GENERATED: do not edit. Source: aistack/api/* docstrings + Pydantic models in aistack/api/_schemas.py, rendered by scripts/gen_api_reference.py. -->

## `POST /v1/audio/transcriptions`

**Transcribe an audio file**

OpenAI Whisper API compatible — clients written against OpenAI's
`/v1/audio/transcriptions` work with no code changes other than the
base URL.

**Backend selection.** The `model` field accepts any of:
- `whisper-{size}` (`whisper-tiny` ... `whisper-large-v3-turbo`,
  `whisper-distil-large-v3`) → faster-whisper / CTranslate2
- `whisper-1` → maps to `whisper-small` for OpenAI legacy compat
- bare size alias (`small`, `medium`, ...) → faster-whisper
- `parakeet` or `nvidia/parakeet-tdt-0.6b-v3` → NVIDIA NeMo
- `sensevoice` or `iic/SenseVoiceSmall` → Alibaba FunASR
- `auto` (or empty) → aistack router picks based on `language`:
  CJK → SenseVoice, European → Parakeet, else → Whisper-small.
  Falls back gracefully when a preferred backend is not installed.

**GPU scheduling.** aistack runs at most one inference at a time
across ASR / LLM / TTS. Concurrent requests get HTTP 503 with a
`Retry-After` header. The blocking inference runs in a worker
thread; the FastAPI event loop stays responsive (`/health` keeps
answering, and the disconnect watcher can cancel a running
transcription cooperatively).

**Streaming.** When `stream=true` the response is `text/event-stream`
with `data: {...}\n\n` frames per SSE convention. Event types:
`transcript.text.delta` (one per emitted segment, with `text` and
optional `start`/`end`/`words`), and `transcript.text.done` once at
the end. Models advertising `supports_streaming=false` in
`/v1/models` (currently Parakeet) emit a single `warning` event
followed by a single `delta` containing the full transcription —
the gateway will not silently chunk a non-streaming model and pay
the WER cost.

**Cancellation.** If the client disconnects mid-request, aistack
sets a cooperative cancel token between segments, releases the GPU
slot, and returns 499 (nginx convention) for clients that re-poll.

**Errors.** Every non-2xx response uses the standard envelope
`{error: {kind, provider, message}}`. See the errors documentation
for status code semantics.

### Request body

**Content type:** `multipart/form-data`

Schema: [`Body_transcribe_v1_audio_transcriptions_post`](#schema-body_transcribe_v1_audio_transcriptions_post)

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | string | yes | Audio file (any ffmpeg-readable format: mp3, mp4, wav, m4a, flac, ogg, webm, mkv). |
| `model` | string | no | Provider/model selector. Empty or 'auto' = pick best installed backend for the given language. Otherwise: whisper-{size} \| parakeet \| sensevoice. |
| `language` | string \| null | no | ISO 639-1 code (e.g. 'en', 'zh'). Omit for auto-detect. |
| `response_format` | string | no | json \| verbose_json \| text. Ignored when stream=true. |
| `translate` | boolean | no | If true, transcribe to English instead of source language. Only Whisper-family models support translation; Parakeet and SenseVoice reject with 400. |
| `stream` | boolean | no | If true, return Server-Sent Events with one transcript.text.delta per decoded segment, ending with transcript.text.done. Models with supports_streaming=false in /v1/models still accept this and emit a warning event followed by a single delta. response_format is ignored when stream=true. |
| `segment_granularity` | string | no | How to group word timestamps into the `segments` field. 'sentence' (default) returns full sentences — right input for line-by-line LLM translation, semantic search, agent reasoning. 'subtitle' returns SRT-cue-sized segments (≤70 chars, 1–7s) for clients that emit SRT/VTT directly without a downstream cue-sizing pass. Affects Parakeet only; faster-whisper and SenseVoice produce VAD-driven segments natively. |

### Responses

#### `200`

Successful Response

- `application/json` → [`TranscriptionResponse`](#schema-transcriptionresponse)
- `text/plain` → string
- `text/event-stream` → string

#### `400`

Malformed request — unsupported response_format / segment_granularity / model id.

- `application/json` → [`ErrorEnvelope`](#schema-errorenvelope)

#### `413`

Audio too large for the chosen backend's VRAM budget.

- `application/json` → [`ErrorEnvelope`](#schema-errorenvelope)

#### `422`

Validation Error

- `application/json` → [`HTTPValidationError`](#schema-httpvalidationerror)

#### `499`

Client disconnected mid-request (nginx convention).

- `application/json` → [`ErrorEnvelope`](#schema-errorenvelope)

#### `500`

Unexpected internal failure.

- `application/json` → [`ErrorEnvelope`](#schema-errorenvelope)

#### `503`

GPU slot busy — gateway is serving another inference. Retry-After is set.

- `application/json` → [`ErrorEnvelope`](#schema-errorenvelope)

---

## Schemas

### `ErrorBody` {#schema-errorbody}

Inner error object shared by every aistack error response.

Consumers branch on `kind` (machine-readable, stable enum) and
surface `message` (human-readable, safe to display). The `provider`
field identifies which subsystem produced the error so logs can be
filtered by backend.

| Field | Type | Required | Description |
|---|---|---|---|
| `kind` | enum (`'network'`, `'malformed'`, `'overflow'`, `'cancelled'`, `'unknown'`) | yes | Stable machine-readable error class. Mapping to HTTP status: malformed=400, overflow=413, cancelled=499, network=503, unknown=500. |
| `provider` | string | yes | Identifier of the subsystem that raised the error: 'aistack' for gateway-level failures, 'Faster-Whisper' / 'Parakeet' / 'SenseVoice' for ASR backends, 'Qwen3-TTS' for TTS, 'ollama' for the LLM proxy. |
| `message` | string | yes | Human-readable description, safe to display to end users. |

### `ErrorEnvelope` {#schema-errorenvelope}

Wire format for every non-2xx response from aistack.

The shape is identical regardless of which endpoint produced the
error, so consumers can write one error-handling helper and reuse it
across capabilities.

| Field | Type | Required | Description |
|---|---|---|---|
| `error` | [`ErrorBody`](#schema-errorbody) | yes |  |

### `TranscriptionResponse` {#schema-transcriptionresponse}

Response shape for `POST /v1/audio/transcriptions`.

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

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | yes | The full transcribed text. Populated for both json and verbose_json formats. |
| `language` | string \| null | no | ISO 639-1 code of the detected (or hinted) source language. Verbose only. |
| `duration` | number \| null | no | Audio duration in seconds, from the backend's own measurement. Verbose only. |
| `segments` | array of [`TranscriptionSegment`](#schema-transcriptionsegment) \| null | no | Per-segment breakdown. Segmentation strategy depends on `segment_granularity` and the backend. Verbose only. |
| `words` | array of [`TranscriptionWord`](#schema-transcriptionword) \| null | no | Flat per-word timestamp list. Convenient for clients that want word timing without traversing segments. Verbose only. |

### `TranscriptionSegment` {#schema-transcriptionsegment}

One segment of a transcription. Granularity depends on the
`segment_granularity` request parameter and the backend's native
segmentation strategy.

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | integer | yes | Zero-based segment index within the response. |
| `start` | number | yes | Segment start time in seconds. |
| `end` | number | yes | Segment end time in seconds. |
| `text` | string | yes | Segment text content. |
| `words` | array of [`TranscriptionWord`](#schema-transcriptionword) \| null | no | Word-level timestamps within this segment, when available. |
| `avg_logprob` | number \| null | no | Mean per-token log probability. Only emitted by faster-whisper. Useful for filtering low-confidence segments. |
| `no_speech_prob` | number \| null | no | Probability that this segment contains no speech. Only emitted by faster-whisper. |
| `compression_ratio` | number \| null | no | Decoded-text compression ratio. Only emitted by faster-whisper. Values > 2.4 typically indicate hallucinated repetition. |
| `temperature` | number \| null | no | Sampling temperature used for this segment. Only emitted by faster-whisper. |

### `TranscriptionWord` {#schema-transcriptionword}

Single word with timestamps. Present in verbose_json responses
when the backend supports word-level timing (faster-whisper does;
Parakeet and SenseVoice produce these via aistack's word-stitching
pipeline).

| Field | Type | Required | Description |
|---|---|---|---|
| `start` | number | yes | Word start time in seconds from audio origin. |
| `end` | number | yes | Word end time in seconds from audio origin. |
| `word` | string | yes | The word text. Punctuation may be attached or stripped depending on the backend. |
