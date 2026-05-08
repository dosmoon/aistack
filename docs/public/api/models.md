---
title: GET /v1/models
description: List models the gateway can serve right now, with capabilities, languages, streaming support, and routing aliases.
sidebar:
  order: 2
---

# `GET /v1/models`

Lists every model that is currently servable by aistack — across all
backends and all capabilities — plus any aistack-provided routing
aliases. Consumers use this to populate model pickers, filter by
language for ASR, and decide whether a given capability is available
before sending a request.

The full request and response schema lives in the auto-generated
[reference for inventory & health](./reference/models/). This page
covers the design rationale: how aistack extends OpenAI's `/v1/models`
shape, what dynamic reachability means, and when to call the endpoint.

## Why aistack extends OpenAI's shape

OpenAI's `/v1/models` returns plain entries with `id` / `object` /
`owned_by`. That is enough to *render* a picker but not enough to
*build* one — a consumer can't tell which entries are ASR vs LLM, what
languages a transcription model supports, or whether streaming will
work. aistack adds four extension fields per entry to close that gap:

- `capabilities` — `["asr"]` / `["tts"]` / `["llm"]`. Lets a picker
  filter by task instead of guessing from the id string.
- `languages` — ISO 639-1 codes the ASR backend can transcribe.
  Absent on TTS / LLM entries.
- `supports_streaming` — whether `stream=true` produces a real
  incremental SSE response on this model. False for backends that
  would only fake streaming at a quality cost (e.g. Parakeet).
- `is_routing_alias` — marks the virtual `id="auto"` entry that
  aistack resolves to a real backend at request time.

OpenAI-only clients ignore the unknown fields and still get a working
picker. aistack-aware clients use them to filter and group precisely.

## The `auto` routing alias

When at least one ASR backend is installed, the response includes a
synthetic entry with `id="auto"` and `is_routing_alias=true`. Sending
`model=auto` to `POST /v1/audio/transcriptions` lets the gateway pick:

- CJK / tonal language hint → SenseVoice (when installed)
- European language hint covered by Parakeet → Parakeet
- Anything else, or no language hint → faster-whisper-small

The alias falls back gracefully when a preferred backend is not
installed, so consumers can ship `model=auto` without per-deployment
backend probing.

The alias does **not** carry a `languages` field — it resolves to
whichever installed real backend best fits the request's `language`
hint, so its language coverage is the union of installed backends.
Its `supports_streaming` is the AND of the candidate pool: True only
when every installed real backend supports streaming.

## Reachability semantics

The list is **dynamic**. A model only appears if its backend can
actually serve a request right now:

| Backend                                          | Visible only when ...                                  |
|--------------------------------------------------|--------------------------------------------------------|
| ASR (faster-whisper, Parakeet, SenseVoice)       | the corresponding Python library imports in the venv  |
| TTS (Qwen3-TTS)                                  | the Docker container responds to its own `/health`    |
| LLM (Ollama)                                     | aistack can reach `localhost:11434` (daemon up)       |

If you start aistack but do not start the TTS Docker container, the
TTS entry is omitted from `/v1/models` and `POST /v1/audio/speech`
returns `503 network`. Consumers should treat the model list as a
**capability inventory**, not a static catalog — refresh it when the
deployment topology changes (a backend was installed, the TTS
container started, Ollama was restarted).

## When to call

- **At client startup** — cache the list, populate UI pickers.
- **When the user opens a "pick model" dialog** — refresh, in case
  the user just installed a new backend or started Ollama.
- **Not on every inference call** — there is no reason to re-fetch
  before each transcription. The endpoint is cheap (import probes
  plus an HTTP HEAD check) but is not free.

## Stability

`id`, `object`, and `owned_by` are part of the OpenAI-spec contract
and stable within `/v1`. `capabilities`, `languages`,
`supports_streaming`, and `is_routing_alias` are aistack extensions;
new values may be added (additive), but existing values never change
meaning. Whether a specific model id is present depends on installed
backends and is not a contract guarantee.
