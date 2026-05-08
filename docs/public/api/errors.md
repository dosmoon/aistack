---
title: Error envelope
description: Common JSON error envelope shared by every aistack endpoint. Branch on error.kind, surface error.message.
sidebar:
  order: 6
---

# Error envelope

All non-2xx responses from aistack use a single JSON envelope. Consumers
can branch on `error.kind` (machine-readable) and surface
`error.message` (human-readable, safe to display). The envelope is
identical regardless of which endpoint produced it.

## Schema

```json
{
  "error": {
    "kind":     "network | malformed | overflow | cancelled | unknown",
    "provider": "aistack | Faster-Whisper | Parakeet | SenseVoice | ...",
    "message":  "Free-form human-readable description."
  }
}
```

Every field is required. The `provider` field identifies which
component reported the failure; "aistack" means the gateway itself
(routing, validation, upstream connectivity), while a specific provider
name (e.g. "Parakeet") means a backend rejected the request. Consumers
that want to attribute failures should branch on `provider`.

## Kinds

| Kind | Meaning | HTTP | Retryable? |
|---|---|---|---|
| `malformed` | Bad input — file missing, format unsupported, unknown model id | 400 | ❌ Caller must fix the request |
| `overflow` | Input too large for the chosen model / VRAM | 413 | ⚠️ Try smaller model or shorter clip |
| `cancelled` | Client disconnected mid-request | 499 | n/a (the caller already left) |
| `network` | Upstream backend unreachable; service not running; model load failed | 503 | ✅ Retry after 5s |
| `unknown` | Internal error not classified by the catalog | 500 | ⚠️ Surface raw, don't auto-retry |

`499` (`cancelled`) follows the nginx convention for "client closed
request" — it is not a standard HTTP code but is widely understood.

## Examples

### `malformed` — unknown model id

```bash
curl -X POST http://127.0.0.1:11500/v1/audio/transcriptions \
     -F file=@audio.mp3 -F model=bogus-model
```

```http
HTTP/1.1 400 Bad Request
Content-Type: application/json

{
  "error": {
    "kind": "malformed",
    "provider": "aistack",
    "message": "Unknown model: 'bogus-model'. Use whisper-{size}, parakeet, or sensevoice."
  }
}
```

### `network` — backend not installed

When the user requests Parakeet but `nemo_toolkit` is not in the venv:

```http
HTTP/1.1 503 Service Unavailable
Content-Type: application/json

{
  "error": {
    "kind": "network",
    "provider": "Parakeet",
    "message": "NeMo toolkit not installed. Run: pip install nemo_toolkit[asr]"
  }
}
```

### `network` — TTS upstream container down

When Qwen3-TTS Docker is not running:

```http
HTTP/1.1 503 Service Unavailable
Content-Type: application/json

{
  "error": {
    "kind": "network",
    "provider": "aistack",
    "message": "Qwen3-TTS container is not reachable. Start it with: docker compose -f docker/tts_qwen3/docker-compose.yml up -d"
  }
}
```

### `503` from the GPU lock — busy, not an error envelope

The single-task GPU lock returns plain FastAPI `HTTPException` with a
`Retry-After` header, not the envelope above. Consumers should treat
this as a transient busy signal:

```http
HTTP/1.1 503 Service Unavailable
Retry-After: 5
Content-Type: application/json

{
  "error": {
    "kind": "network",
    "provider": "aistack",
    "message": "aistack GPU slot is busy (held by asr); rejected llm. Retry after a few seconds."
  }
}
```

The slot-busy 503 uses the same envelope as every other error path —
`kind="network"` because the slot-busy state is a transport-level
back-pressure signal (server is healthy, retry after the
`Retry-After` hint). Callers should detect it specifically by the
`Retry-After` header rather than by `kind` alone, since other
`network` errors (Ollama unreachable, model download failed) are also
503 and would otherwise be indistinguishable.

## Consumer-side handling pattern

```python
import httpx

def call_aistack(method, url, **kw):
    r = httpx.request(method, url, **kw)
    if r.status_code == 200:
        return r.json()
    if r.status_code == 503 and r.headers.get("Retry-After"):
        # Single-task busy signal — retry after the suggested delay
        raise BusyError(retry_after=int(r.headers["Retry-After"]))
    try:
        env = r.json().get("error", {})
        kind = env.get("kind", "unknown")
        provider = env.get("provider", "aistack")
        message = env.get("message", r.text)
    except (ValueError, AttributeError):
        kind, provider, message = "unknown", "aistack", r.text
    raise AistackError(kind, provider, message, status=r.status_code)
```

## Stability

The set of `kind` values is part of the contract:

- Adding new kinds is allowed in `/v1`.
- Renaming or removing existing kinds requires `/v2`.

The `message` text is **not** stable — wording may change between
releases. Code must branch on `kind`, not on string matching `message`.
