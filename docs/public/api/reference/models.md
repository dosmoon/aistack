---
title: Inventory & health
description: Auto-generated reference for GET /health and GET /v1/models — what the gateway can serve right now.
sidebar:
  order: 10
---

<!-- AUTO-GENERATED: do not edit. Source: aistack/api/* docstrings + Pydantic models in aistack/api/_schemas.py, rendered by scripts/gen_api_reference.py. -->

## `GET /health`

**Liveness probe**

Returns 200 with a small JSON body when the worker is ready.

Connection refused or non-200 means aistack is down or still
starting; consumers should surface "service unreachable" with a
hint to start the dev server. This endpoint never blocks on
backend health — it only confirms the FastAPI worker itself is
alive.

### Responses

#### `200`

Successful Response

- `application/json` → [`HealthResponse`](#schema-healthresponse)

## `GET /v1/models`

**List servable models**

OpenAI-compatible model list. Lists only backends that will actually
serve a request right now:

  - ASR entries are emitted for every provider whose ML library is
    importable in the running venv (pure import probe — no model
    weights are loaded).
  - The TTS entry is emitted only if the Qwen3-TTS upstream
    container responds to its `/health` probe.
  - LLM entries are aggregated from Ollama's `/api/tags` when the
    daemon is reachable; an unreachable Ollama silently contributes
    zero entries rather than failing the whole listing.

Clients can read this once on startup to know which capabilities are
available, build language-aware pickers, and skip a 503 round-trip
on first call. The response shape is OpenAI-compatible plus the
aistack extension fields `capabilities`, `languages`,
`supports_streaming`, and `is_routing_alias` per entry — see the
ModelEntry schema.

### Responses

#### `200`

Inventory of every model the gateway can serve right now, across ASR / TTS / LLM, plus the 'auto' routing alias when at least one ASR backend is reachable.

- `application/json` → [`ModelsList`](#schema-modelslist)

---

## Schemas

### `HealthResponse` {#schema-healthresponse}

Response shape for `GET /health`.

| Field | Type | Required | Description |
|---|---|---|---|
| `status` | string | yes | Always 'ok' when this endpoint responds. Connection refused / non-200 means the gateway is down. |
| `version` | string | yes | aistack package version (PEP 440). |

### `ModelEntry` {#schema-modelentry}

One entry in the `/v1/models` inventory.

OpenAI-compatible base fields (id, object, owned_by) are augmented
with aistack extension fields (capabilities, languages,
supports_streaming, is_routing_alias) so consumers can build
capability-aware pickers without per-backend lookup tables.

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Model identifier. May be a HuggingFace model id, a Whisper size alias, an Ollama tag, or the routing alias 'auto'. |
| `object` | string | no | Always 'model' (OpenAI compatibility). |
| `owned_by` | string | yes | Origin of the model weights or routing decision. |
| `capabilities` | array of enum (`'asr'`, `'tts'`, `'llm'`) | yes | Which gateway capabilities this entry can serve. Single-element list in current versions; reserved as a list for future multi-capability entries. |
| `languages` | array of string \| null | no | ISO 639-1 codes the backend can transcribe. ASR-only field; absent on TTS / LLM entries. |
| `supports_streaming` | boolean \| null | no | True when stream=true on this model produces real incremental SSE output. False means the gateway accepts stream=true but emits a single-event SSE response with a warning. Absent on entries where streaming is irrelevant. |
| `is_routing_alias` | boolean \| null | no | True for the 'auto' entry (routing decision, not a real model). |

### `ModelsList` {#schema-modelslist}

Response shape for `GET /v1/models` — OpenAI-compatible.

| Field | Type | Required | Description |
|---|---|---|---|
| `object` | string | no | Always 'list' (OpenAI compatibility). |
| `data` | array of [`ModelEntry`](#schema-modelentry) | yes | One entry per servable model + the 'auto' routing alias when at least one ASR backend is reachable. |
