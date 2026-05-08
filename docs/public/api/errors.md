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

The wire-format schema (every field, every kind value, the HTTP
status mapping) lives in the auto-generated
[reference for the error envelope](./reference/asr/#schema-errorenvelope).
This page covers the design rationale and the consumer-side pattern —
the *why* and *how to use it*, not the *what*.

## Why one envelope across all endpoints

Three reasons aistack uses the same `{error: {kind, provider, message}}`
shape for every non-2xx response:

1. **One handler per consumer.** A client that integrates with ASR,
   TTS, and LLM writes the error path once, not three times.
2. **`kind` is stable; `message` is not.** Code branches on `kind`
   (a small, stable enum); humans read `message` (free-form, may be
   reworded between releases). String-matching `message` is the bug;
   branching on `kind` is the contract.
3. **`provider` attributes the failure.** When something goes wrong
   you want to know "is it the gateway routing, my upstream Ollama,
   or the Whisper backend?" — `provider` answers without log
   spelunking. `"aistack"` means the gateway itself; a backend name
   like `"Parakeet"` means that subsystem rejected the request.

## Status code semantics worth knowing

The kind → status mapping is in the
[reference table](./reference/asr/#schema-errorenvelope). Two
non-obvious points:

- **`499` (`cancelled`)** follows nginx's "client closed request"
  convention — it is not a standard HTTP code but is widely
  understood. The caller has already left, so this status is
  informational only; it shows up in your access logs but never in
  a live response.
- **`503` is overloaded** between two real situations: "Ollama is
  down / model failed to load" and "GPU slot is busy serving
  another inference, retry shortly." Distinguish them by the
  `Retry-After` header — present only on the slot-busy path. See
  the back-pressure section below.

## Error examples

### Unknown model id (`malformed`, 400)

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

### Backend not installed (`network`, 503)

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

### TTS upstream container down (`network`, 503)

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

### GPU slot busy (`network` + `Retry-After`, 503)

Single-task GPU lock back-pressure — server is healthy, just
already busy serving another inference:

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

The slot-busy path uses the same envelope shape as every other
`503` because it *is* a transport-level back-pressure signal.
Distinguish it from "Ollama is down" by the `Retry-After` header —
present only on the slot-busy path. Other `network` errors (model
download failure, upstream daemon unreachable) are also `503` and
would otherwise be indistinguishable.

## Consumer-side handling pattern

A reference Python implementation that handles all five kinds plus
the slot-busy retry case:

```python
import httpx

class AistackError(Exception):
    def __init__(self, kind, provider, message, status):
        self.kind, self.provider, self.message, self.status = kind, provider, message, status
        super().__init__(f"[{kind}/{provider}] {message}")

class BusyError(Exception):
    def __init__(self, retry_after):
        self.retry_after = retry_after

def call_aistack(method, url, **kw):
    r = httpx.request(method, url, **kw)
    if r.status_code == 200:
        return r.json()
    if r.status_code == 503 and r.headers.get("Retry-After"):
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
