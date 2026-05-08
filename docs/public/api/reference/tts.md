---
title: TTS — text to speech
description: Auto-generated reference for the /v1/audio/{path} reverse proxy to Qwen3-TTS.
sidebar:
  order: 12
---

<!-- AUTO-GENERATED: do not edit. Source: aistack/api/* docstrings + Pydantic models in aistack/api/_schemas.py, rendered by scripts/gen_api_reference.py. -->

## `POST /v1/audio/{path}`

**Proxy to Qwen3-TTS (POST)**

Transparent reverse proxy for /v1/audio/* to the
Qwen3-TTS-12Hz-0.6B-CustomVoice container.

**Transparent.** Request body, headers (minus hop-by-hop), and response
body all flow through unchanged. aistack does not transcode audio,
swap voices, or adapt OpenAI's spec — what Qwen3-TTS returns is what
the client receives. The OpenAI-compatible request/response schemas
are documented authoritatively at
https://platform.openai.com/docs/api-reference/audio.

**GPU scheduling.** Holds the global gateway GPU slot for the
duration of the upstream call. The Qwen3-TTS container generates on
the same physical GPU as in-process ASR / LLM workloads, so the slot
represents "GPU is doing inference" regardless of which process owns
the kernels. Concurrent requests get HTTP 503 with `Retry-After`.

**Streaming.** The upstream is consumed and forwarded chunk-by-chunk
so multi-MB audio responses don't buffer entirely in memory. Client
disconnect propagates: aistack closes the upstream connection so the
container can abort generation early.

**Error mapping.** ConnectError on the upstream → 503 with a hint to
start the docker compose stack. Other httpx errors → 502 with the
upstream exception type/message in the envelope.

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `path` | path | string | yes |  |

### Responses

#### `200`

Upstream Qwen3-TTS response, forwarded verbatim. For `POST /v1/audio/speech` the body is raw audio bytes (content-type per OpenAI spec); other paths under `/v1/audio/*` (clone-voice / list-voices / etc.) preserve the upstream's content-type and shape.

- `application/json` → object
- `audio/mpeg` → string (binary)
- `audio/wav` → string (binary)
- `audio/opus` → string (binary)

#### `422`

Validation Error

- `application/json` → [`HTTPValidationError`](#schema-httpvalidationerror)

#### `502`

Qwen3-TTS upstream produced an unexpected error.

- `application/json` → [`ErrorEnvelope`](#schema-errorenvelope)

#### `503`

Either the GPU slot is busy serving another inference (gateway-level), or the Qwen3-TTS container is unreachable. The error envelope's `provider` field distinguishes the two.

- `application/json` → [`ErrorEnvelope`](#schema-errorenvelope)

## `GET /v1/audio/{path}`

**Proxy to Qwen3-TTS (GET)**

Transparent reverse proxy for /v1/audio/* to the
Qwen3-TTS-12Hz-0.6B-CustomVoice container.

**Transparent.** Request body, headers (minus hop-by-hop), and response
body all flow through unchanged. aistack does not transcode audio,
swap voices, or adapt OpenAI's spec — what Qwen3-TTS returns is what
the client receives. The OpenAI-compatible request/response schemas
are documented authoritatively at
https://platform.openai.com/docs/api-reference/audio.

**GPU scheduling.** Holds the global gateway GPU slot for the
duration of the upstream call. The Qwen3-TTS container generates on
the same physical GPU as in-process ASR / LLM workloads, so the slot
represents "GPU is doing inference" regardless of which process owns
the kernels. Concurrent requests get HTTP 503 with `Retry-After`.

**Streaming.** The upstream is consumed and forwarded chunk-by-chunk
so multi-MB audio responses don't buffer entirely in memory. Client
disconnect propagates: aistack closes the upstream connection so the
container can abort generation early.

**Error mapping.** ConnectError on the upstream → 503 with a hint to
start the docker compose stack. Other httpx errors → 502 with the
upstream exception type/message in the envelope.

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `path` | path | string | yes |  |

### Responses

#### `200`

Upstream Qwen3-TTS response, forwarded verbatim. For `POST /v1/audio/speech` the body is raw audio bytes (content-type per OpenAI spec); other paths under `/v1/audio/*` (clone-voice / list-voices / etc.) preserve the upstream's content-type and shape.

- `application/json` → object
- `audio/mpeg` → string (binary)
- `audio/wav` → string (binary)
- `audio/opus` → string (binary)

#### `422`

Validation Error

- `application/json` → [`HTTPValidationError`](#schema-httpvalidationerror)

#### `502`

Qwen3-TTS upstream produced an unexpected error.

- `application/json` → [`ErrorEnvelope`](#schema-errorenvelope)

#### `503`

Either the GPU slot is busy serving another inference (gateway-level), or the Qwen3-TTS container is unreachable. The error envelope's `provider` field distinguishes the two.

- `application/json` → [`ErrorEnvelope`](#schema-errorenvelope)

## `DELETE /v1/audio/{path}`

**Proxy to Qwen3-TTS (DELETE)**

Transparent reverse proxy for /v1/audio/* to the
Qwen3-TTS-12Hz-0.6B-CustomVoice container.

**Transparent.** Request body, headers (minus hop-by-hop), and response
body all flow through unchanged. aistack does not transcode audio,
swap voices, or adapt OpenAI's spec — what Qwen3-TTS returns is what
the client receives. The OpenAI-compatible request/response schemas
are documented authoritatively at
https://platform.openai.com/docs/api-reference/audio.

**GPU scheduling.** Holds the global gateway GPU slot for the
duration of the upstream call. The Qwen3-TTS container generates on
the same physical GPU as in-process ASR / LLM workloads, so the slot
represents "GPU is doing inference" regardless of which process owns
the kernels. Concurrent requests get HTTP 503 with `Retry-After`.

**Streaming.** The upstream is consumed and forwarded chunk-by-chunk
so multi-MB audio responses don't buffer entirely in memory. Client
disconnect propagates: aistack closes the upstream connection so the
container can abort generation early.

**Error mapping.** ConnectError on the upstream → 503 with a hint to
start the docker compose stack. Other httpx errors → 502 with the
upstream exception type/message in the envelope.

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `path` | path | string | yes |  |

### Responses

#### `200`

Upstream Qwen3-TTS response, forwarded verbatim. For `POST /v1/audio/speech` the body is raw audio bytes (content-type per OpenAI spec); other paths under `/v1/audio/*` (clone-voice / list-voices / etc.) preserve the upstream's content-type and shape.

- `application/json` → object
- `audio/mpeg` → string (binary)
- `audio/wav` → string (binary)
- `audio/opus` → string (binary)

#### `422`

Validation Error

- `application/json` → [`HTTPValidationError`](#schema-httpvalidationerror)

#### `502`

Qwen3-TTS upstream produced an unexpected error.

- `application/json` → [`ErrorEnvelope`](#schema-errorenvelope)

#### `503`

Either the GPU slot is busy serving another inference (gateway-level), or the Qwen3-TTS container is unreachable. The error envelope's `provider` field distinguishes the two.

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
