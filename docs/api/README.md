# aistack HTTP API

This is the public API surface of aistack — the contract that consumers
(VideoCraft and any future client) depend on. Internal implementation
choices (which inference engine, where the GPU lives, how scheduling
works) are documented under `docs/design/` and `docs/selection/` and
are **not** part of the contract: they may change without API version
bumps.

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
- `GET /v1/models` mirrors OpenAI's model list shape, with an
  aistack-specific `capabilities` field added per entry.

Where aistack adds capabilities beyond the OpenAI spec (e.g.
SenseVoice's `task_type` parameter, vLLM-Omni's voice clone fields),
they are exposed as extra optional fields. Standard OpenAI clients
that ignore unknown fields work without modification.

## Endpoints

| Endpoint | Method | Doc |
|---|---|---|
| `/health` | GET | (this README — see below) |
| `/v1/models` | GET | [models.md](models.md) |
| `/v1/audio/transcriptions` | POST | [asr.md](asr.md) |
| `/v1/audio/speech` | POST | [tts.md](tts.md) |
| `/v1/chat/completions` | POST | [llm.md](llm.md) *(D6)* |

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
