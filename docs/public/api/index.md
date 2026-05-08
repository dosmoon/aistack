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

## Where to start

If you are integrating aistack for the first time, **read
[`integration.md`](integration.md) first** — it walks through the
typical consumer journey from capability discovery to error handling
in one coherent narrative.

The per-endpoint pages on this index are the field-by-field reference
material that integration.md links into.

## Versioning policy

Endpoints are prefixed with `/v1/`. The contract within `/v1/*`
follows additive backward compatibility:

| Change | Allowed in `/v1`? |
|---|---|
| Adding new endpoints | ✅ |
| Adding new optional fields to requests | ✅ |
| Adding new fields to responses | ✅ |
| Tightening field validation | ⚠️ Avoid; only when current behavior is buggy |
| Removing or renaming any field | ❌ requires `/v2` |
| Changing response shape | ❌ requires `/v2` |

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
  `supports_streaming`) added per entry. See [models.md](models.md).

Where aistack adds capabilities beyond the OpenAI spec (e.g.
SenseVoice's `task_type` parameter, vLLM-Omni's voice clone fields),
they are exposed as extra optional fields. Standard OpenAI clients
that ignore unknown fields work without modification.

### Streaming

All three capability endpoints support streaming where the underlying
model supports it. Discovery is via the `supports_streaming` field
on each `/v1/models` entry; the wire format is the standard SSE
shape OpenAI uses for that capability (`transcript.text.delta` /
`transcript.text.done` for ASR, OpenAI chat-completion deltas for
LLM, raw audio chunks via vLLM-Omni's streaming endpoint for TTS).
ASR adds an aistack `warning` event for the rare case where a
selected model does not natively stream — see
[`integration.md` §4](integration.md#streaming-transcription-with-streamtrue).

## Endpoints

| Endpoint | Method | Doc |
|---|---|---|
| `/health` | GET | (this page — see below) |
| `/v1/models` | GET | [models.md](models.md) |
| `/v1/audio/transcriptions` | POST | [asr.md](asr.md) |
| `/v1/audio/speech` | POST | [tts.md](tts.md) |
| `/v1/chat/completions` | POST | [llm.md](llm.md) |

For a tour that combines all of these into a working integration,
read [`integration.md`](integration.md).

For the **performance & availability analysis** layer (built-in metrics,
JSONL access logs, request/response capture, `X-Request-ID` propagation,
`/admin/api/metrics` JSON endpoint), see
[`observability.md`](observability.md).

## `GET /health`

Liveness probe. Returns 200 with a small JSON body when the worker is
ready to accept requests.

```bash
curl http://127.0.0.1:11500/health
```

```json
{
  "status": "ok",
  "version": "0.0.1"
}
```

A failed health check (connection refused, non-200) means aistack is
down or still starting; consumers should surface this as "service
unreachable" with a hint to start the dev server.

## Errors

All non-2xx responses use a common envelope. See [errors.md](errors.md)
for the full contract, including HTTP status codes per error kind.

```json
{
  "error": {
    "kind": "network",
    "provider": "aistack",
    "message": "aistack service is not reachable. ..."
  }
}
```

## Live API explorer

While aistack is running, FastAPI's auto-generated Swagger UI is at:

```
http://127.0.0.1:11500/docs
```

It always reflects the running version's actual schema. For human
explanation (request semantics, error scenarios, examples, design
rationale) see the per-endpoint markdown files in this directory.
