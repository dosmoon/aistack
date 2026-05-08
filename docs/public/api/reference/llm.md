---
title: LLM — chat completion
description: Auto-generated reference for POST /v1/chat/completions, reverse-proxied to local Ollama.
sidebar:
  order: 13
---

<!-- AUTO-GENERATED: do not edit. Source: aistack/api/* docstrings + Pydantic models in aistack/api/_schemas.py, rendered by scripts/gen_api_reference.py. -->

## `POST /v1/chat/completions`

**Chat completion (Ollama proxy)**

OpenAI-compatible chat completion endpoint, reverse-proxied to a
local Ollama daemon (default `http://127.0.0.1:11434`).

**Value-adds over a direct Ollama call.** Two things happen between
the client and the upstream that justify routing through aistack:

1. **GPU scheduling.** Before forwarding, aistack evicts its own
   in-process ASR `asr-main` cache entries so the LLM inference
   does not contend with a hot Whisper/Parakeet/SenseVoice for
   VRAM. The whole call holds the gateway's single GPU slot, so
   concurrent LLM/ASR/TTS requests get HTTP 503 with `Retry-After`.

2. **`keep_alive` default.** When the client omits the
   `keep_alive` field, aistack injects `"30s"` so Ollama releases
   the model shortly after the completion. Sequential LLM calls
   within that window reuse the loaded model; idle releases VRAM
   back for ASR. Clients that want a different lifetime override
   explicitly.

**Streaming.** When `stream=true` the response is forwarded
chunk-by-chunk via FastAPI StreamingResponse, so the client sees
first tokens as soon as Ollama emits them. Client disconnect
propagates: aistack closes the upstream connection so Ollama's
runner can abort generation rather than running to completion on
a dead socket.

**Cancellation.** Client disconnect mid-stream releases the GPU
slot promptly and aborts the upstream call.

**Request schema.** OpenAI-compatible — see the OpenAI Chat
Completion API reference for field semantics. aistack does not
transform the request body except to inject the `keep_alive`
default; it forwards every other field verbatim.

### Responses

#### `200`

Ollama's response, forwarded verbatim. Schema follows OpenAI's `/v1/chat/completions` contract — see https://platform.openai.com/docs/api-reference/chat for the field reference. When `stream=true` in the request, the response is a Server-Sent Events stream of OpenAI-shape delta chunks terminated by `data: [DONE]`.

- `application/json` → object
- `text/event-stream` → string

#### `400`

Request body is not valid JSON.

- `application/json` → [`ErrorEnvelope`](#schema-errorenvelope)

#### `502`

Ollama upstream produced an unexpected error.

- `application/json` → [`ErrorEnvelope`](#schema-errorenvelope)

#### `503`

Either the GPU slot is busy serving another inference (gateway-level), or Ollama is unreachable (e.g. the daemon is not running). The error envelope's `provider` field distinguishes the two.

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
