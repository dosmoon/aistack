---
title: HTTP API
description: aistack's HTTP contract — ASR, TTS, LLM proxy, models inventory, errors, and observability.
sidebar:
  order: 0
---

# aistack HTTP API

This section **is the contract aistack publishes to consumers**. Any
client — CLI tools, GUI applications like VideoCraft, agent frameworks,
future dosmoon products — integrates against what is documented here.
aistack does not adapt to any particular consumer; consumers conform
to this contract.

Internal implementation choices (which inference engine, where the GPU
lives, how scheduling works) are **not** part of the contract — they
may change without API version bumps.

## Two flavours of documentation in this section

- **Narrative pages** (this page, [`integration`](./integration/),
  [`asr`](./asr/), [`tts`](./tts/), [`llm`](./llm/),
  [`models`](./models/), [`errors`](./errors/),
  [`observability`](./observability/)) explain *why* the contract
  looks the way it does — design choices, integration journeys,
  trade-offs.
- **[Auto-generated reference](./reference/)** lists *what* every
  endpoint accepts and returns — request/response field tables,
  enum values, error codes, JSON schemas. The reference is generated
  from the live FastAPI app's OpenAPI spec on every build, so it
  cannot drift from the running code.

If you are integrating aistack for the first time, start with the
[Integration Guide](./integration/). When you need a specific field's
type or an exhaustive list of error codes, jump to the
[Reference](./reference/).

## Versioning policy

Endpoints are prefixed with `/v1/`. The contract within `/v1/*`
follows additive backward compatibility:

- Adding new endpoints — allowed
- Adding new optional fields to requests — allowed
- Adding new fields to responses — allowed
- Tightening field validation — discouraged; only when current behavior is buggy
- Removing or renaming any field — requires `/v2`
- Changing response shape — requires `/v2`

When `/v2` is introduced, `/v1` will continue serving for at least one
release cycle so consumers can migrate without coupled deploys.

## Base URL

```
http://127.0.0.1:11500
```

The bind address is configured server-side and may be different in
shared deployments (LAN, cloud GPU rental). Consumers should treat the
base URL as configuration, not a constant.

## Authentication

aistack runs **unauthenticated** by default. It is intended for
localhost or private-LAN deployment. If you expose aistack on a public
network, put it behind a reverse proxy or VPN — there is no per-request
auth in the protocol itself.

## Protocol style

OpenAI API compatible where the OpenAI spec applies:

- `POST /v1/audio/transcriptions` mirrors OpenAI's Whisper API.
- `POST /v1/audio/speech` mirrors OpenAI's TTS API.
- `POST /v1/chat/completions` mirrors OpenAI's chat completion API.
- `GET /v1/models` mirrors OpenAI's model list shape, with aistack
  extension fields (`capabilities`, `languages`, `is_routing_alias`,
  `supports_streaming`) added per entry.

Where aistack adds capabilities beyond the OpenAI spec (e.g.
SenseVoice's `task_type` parameter, vLLM-Omni's voice clone fields),
they are exposed as extra optional fields. Standard OpenAI clients
that ignore unknown fields work without modification.

## Streaming

All three capability endpoints support streaming where the underlying
model supports it. Discovery is via the `supports_streaming` field
on each `/v1/models` entry; the wire format is the standard SSE
shape OpenAI uses for that capability (`transcript.text.delta` /
`transcript.text.done` for ASR, OpenAI chat-completion deltas for
LLM, raw audio chunks via vLLM-Omni's streaming endpoint for TTS).
ASR adds an aistack `warning` event for the rare case where a
selected model does not natively stream — see
[`integration.md` §4](./integration/#streaming-transcription-with-streamtrue).

## Errors

All non-2xx responses use a common envelope:

```json
{
  "error": {
    "kind": "network",
    "provider": "aistack",
    "message": "aistack service is not reachable. ..."
  }
}
```

Consumers branch on `error.kind` (machine-readable) and surface
`error.message` (human-readable, safe to display). The
[errors page](./errors/) explains the design and the consumer
pattern; the [reference](./reference/) lists every kind value and
its HTTP status mapping.

## Live API explorer

While aistack is running, FastAPI's auto-generated Swagger UI is at:

```
http://127.0.0.1:11500/docs
```

It always reflects the running version's actual schema. The
[Reference section](./reference/) is the build-time rendering of
the same OpenAPI spec, so the two should agree exactly.
