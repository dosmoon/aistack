---
title: Integration Guide
description: How to integrate aistack from a client — capability discovery, requests, errors, streaming, and the full consumer journey.
sidebar:
  order: 1
---

# aistack Integration Guide

> **This document is the contract aistack publishes to consumers.** It
> is the single authoritative source for "what aistack does and how to
> use it" from any client — CLI tools, GUI apps like VideoCraft, agent
> frameworks, future dosmoon products. aistack does not adapt to any
> particular consumer's needs; consumers conform to what is documented
> here.
>
> If you are integrating aistack for the first time, start here. The
> per-endpoint reference pages (`asr.md`, `tts.md`, `llm.md`,
> `models.md`, `errors.md`) drill into specifics; this guide stitches
> them into a coherent integration journey.

---

## 1. What aistack is

aistack is a **local AI capability gateway**. A single OpenAI-API-
compatible server (default `127.0.0.1:11500`) that exposes three
capabilities — speech-to-text (ASR), text-to-speech (TTS), and chat
completion (LLM) — and owns local GPU scheduling so consumers do not
have to.

| You as consumer | aistack |
|---|---|
| Send OpenAI-shape HTTP requests. | Picks which backend serves the request. |
| Read OpenAI-shape responses. | Coordinates the local GPU across all backends. |
| Don't know or care which backend ran your request. | Hot-swaps models on memory pressure. |
| Don't ship ML libraries with your app. | Hosts the ML libraries locally or proxies them. |

The contract is intentionally narrow: stable HTTP shapes, a stable
error envelope, capability discovery via `/v1/models`, and 503 +
`Retry-After` as the universal "we are momentarily busy" signal.

---

## 2. Endpoints at a glance

| Method | Path | Purpose | Reference |
|---|---|---|---|
| GET | `/health` | Liveness probe. | This page §10. |
| GET | `/v1/models` | Capability inventory — what the gateway can serve right now. | [`models.md`](models.md) |
| POST | `/v1/audio/transcriptions` | Speech-to-text. | [`asr.md`](asr.md) |
| POST | `/v1/audio/speech` | Text-to-speech (and related TTS endpoints under `/v1/audio/*`). | [`tts.md`](tts.md) |
| POST | `/v1/chat/completions` | Chat completion (proxied to local Ollama). | [`llm.md`](llm.md) |

All endpoints are unauthenticated by default; aistack is meant to bind
on `127.0.0.1` or a private LAN. If exposed beyond that, put a
reverse proxy with auth in front.

---

## 3. Step 1 — Discover what's available

Always start by calling `GET /v1/models`. This is how the gateway
introduces itself; no other endpoint reveals which capabilities are
currently usable.

```bash
curl http://127.0.0.1:11500/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "auto",
      "object": "model",
      "owned_by": "aistack",
      "capabilities": ["asr"],
      "is_routing_alias": true
    },
    {
      "id": "whisper-small",
      "object": "model",
      "owned_by": "openai",
      "capabilities": ["asr"],
      "languages": ["en", "zh", "ja", "ko", "es", "fr", "..."]
    },
    {
      "id": "iic/SenseVoiceSmall",
      "object": "model",
      "owned_by": "alibaba",
      "capabilities": ["asr"],
      "languages": ["zh", "yue", "en", "ja", "ko"]
    },
    {
      "id": "qwen3-tts-12hz-0.6b-customvoice",
      "object": "model",
      "owned_by": "qwen",
      "capabilities": ["tts"]
    },
    {
      "id": "qwen3:4b",
      "object": "model",
      "owned_by": "ollama",
      "capabilities": ["llm"]
    }
  ]
}
```

### Field semantics

| Field | Type | Notes |
|---|---|---|
| `id` | string | Pass verbatim as `model` on capability endpoints. |
| `object` | string | Always `"model"` (OpenAI-spec required). |
| `owned_by` | string | Free-form attribution to the model author. Display only; do not branch on it. |
| `capabilities` | array of string | aistack extension. Subset of `["asr","tts","llm"]`. Filter the picker by this. |
| `languages` | array of string (ASR only) | aistack extension. ISO 639-1 codes the model can transcribe. Absent on TTS / LLM entries and on routing aliases. |
| `is_routing_alias` | boolean | aistack extension. When `true`, the entry is a virtual id that aistack resolves internally rather than a real model. Currently only `id="auto"` for ASR. |

### When to refresh

- Once at startup, cache by capability.
- Whenever the user opens a model picker (a backend may have just
  been installed, Ollama may have just started).
- **Not** before every inference call — the listing is cheap but not
  free, and stable on the seconds-to-minutes scale.

### What it means when an entry is missing

The list reflects **right now**, not a static catalog:

| Backend | Visible only when ... |
|---|---|
| ASR providers | the corresponding Python library is importable in the venv |
| TTS (Qwen3-TTS) | the Docker container responds to its own `/health` |
| LLM (Ollama) | aistack can reach Ollama's `/api/tags` |

If the TTS container is down, `/v1/audio/speech` will respond 503; if
Ollama is down, `/v1/chat/completions` will respond 503. Treat
`/v1/models` as the discovery layer that tells you *not* to dispatch
to a missing capability in the first place.

---

## 4. Step 2 — Transcribe audio (ASR)

`POST /v1/audio/transcriptions` mirrors OpenAI's Whisper API. Accepts
`multipart/form-data`. Returns the language, duration, full text,
per-segment timestamps, and per-word timestamps where the backend
supports them.

### Pick a model

Three real backends are exposed when their libraries are installed,
plus the `auto` routing alias.

| Picker choice | Behavior |
|---|---|
| `model=auto` | aistack picks based on the `language` form field: CJK → SenseVoice, European → Parakeet, else → faster-whisper-small. Falls back gracefully when a preferred backend is not installed. |
| `model=whisper-small` (or any whisper size) | faster-whisper / CTranslate2. Default general-purpose choice. |
| `model=parakeet` | NVIDIA Parakeet TDT 0.6B v3. Strongest English/European accuracy. Word-level timestamps from the model itself. |
| `model=sensevoice` | Alibaba SenseVoice Small. Best CJK; also handles English/Japanese/Korean. |

Most consumers should expose `auto` as the default option in their
picker; advanced users can pin a specific backend.

### Example: curl

```bash
curl -X POST http://127.0.0.1:11500/v1/audio/transcriptions \
  -F "file=@speech.mp3" \
  -F "model=auto" \
  -F "language=en" \
  -F "response_format=verbose_json"
```

### Example: Python (httpx)

```python
import httpx

with open("speech.mp3", "rb") as f:
    r = httpx.post(
        "http://127.0.0.1:11500/v1/audio/transcriptions",
        files={"file": f},
        data={
            "model": "auto",
            "language": "en",          # optional hint; drives auto routing
            "response_format": "verbose_json",
            "translate": "false",       # set true for Whisper-only "to English" mode
        },
        timeout=120.0,
    )
r.raise_for_status()
result = r.json()
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f} → {seg['end']:.2f}] {seg['text']}")
```

### Response shape (`response_format=verbose_json`)

```json
{
  "language": "en",
  "duration": 17.18,
  "text": "...",
  "segments": [
    {"id": 0, "start": 0.81, "end": 7.14, "text": "..."}
  ],
  "words": [
    {"start": 0.81, "end": 0.99, "word": "The"}
  ]
}
```

`words[]` is populated for every backend that supports word-level
timestamps; clients should treat its absence as "not available for
this backend / configuration", not as an error.

### Streaming transcription with `stream=true`

For long audio (or any case where the client wants partial results
as they become available), pass `stream=true` as a form field. The
response is `text/event-stream` instead of JSON; events follow
OpenAI's transcription streaming shape with one extension event.

Example:

```bash
curl -N -X POST http://127.0.0.1:11500/v1/audio/transcriptions \
  -F "file=@long_lecture.mp3" \
  -F "model=whisper-small" \
  -F "language=en" \
  -F "stream=true"
```

Wire format (one event per `data: { ... }\n\n` frame):

```
data: {"type": "transcript.text.delta", "delta": "Hello world.",
       "segment": {"start": 0.0, "end": 1.7,
                   "words": [{"start":0.0,"end":0.4,"word":"Hello"}, ...]}}

data: {"type": "transcript.text.delta", "delta": "This is the second segment.",
       "segment": {"start": 1.7, "end": 4.2, "words": [...]}}

... (more deltas as the model produces segments) ...

data: {"type": "transcript.text.done", "language": "en", "duration": 1020.0}
```

#### Event types

| `type` | Meaning |
|---|---|
| `transcript.text.delta` | Incremental segment. `delta` is the segment text; `segment` carries start/end in seconds and per-word timestamps. |
| `transcript.text.done` | End of transcription. Carries detected language and total duration. |
| `warning` (aistack extension) | Emitted **before** any delta when the chosen model does not support real streaming. Carries `code`, `model`, `message`. The transcription still arrives as a single delta after the warning. |
| `error` (aistack extension) | Mid-stream failure. Body matches the standard error envelope shape (`{kind, provider, message}`). No further events follow. |

#### Streaming-capable vs not

Models advertise their streaming behavior via `supports_streaming` in
`/v1/models`. As of the current contract:

- `whisper-small` (and any whisper size) — **streams natively**:
  one delta per decoded segment, no warning.
- `iic/SenseVoiceSmall` — **streams natively**: one delta per VAD chunk,
  no warning.
- `nvidia/parakeet-tdt-0.6b-v3` — **does not stream**: client gets a
  `warning` event up front, then one delta with the full text, then
  `transcript.text.done`. Selecting Parakeet via the picker is fine,
  but aware clients should hide it from streaming-only workflows by
  filtering on `supports_streaming`.
- `auto` — streams when the language hint routes to a streaming-capable
  backend, downgrades when it routes to a non-streaming one.
  `supports_streaming` on the alias entry is the AND of the candidate
  pool, so it is `false` whenever Parakeet is installed.

The downgrade path delivers the full transcription in one event rather
than failing the request, so OpenAI-shape clients that always send
`stream=true` get a working response. The `warning` event is the
discoverable signal; aware clients should branch on it.

#### Cancellation

Same pattern as the LLM stream (§6): close the HTTP connection. The
gateway polls `request.is_disconnected()` and propagates a cancel
token into the worker thread, which checks it between segments.
Long-audio transcriptions abort within ~1 second of disconnect for
streaming-capable backends; Parakeet (downgrade path) honors cancel
only at coarse boundaries.

#### `response_format` interaction

When `stream=true`, `response_format` is ignored — the response is
always SSE in the shape above. Consumers that need to choose between
plain text / json / verbose_json should leave `stream=false`.

---

## 5. Step 3 — Generate speech (TTS)

`POST /v1/audio/speech` is a transparent proxy to the locally-running
Qwen3-TTS container. Standard OpenAI fields are accepted (`model`,
`input`, `voice`, `response_format`); the upstream also exposes
extension fields for voice clone and voice design which pass through
unchanged.

### Example: minimal speech synthesis

```bash
curl -X POST http://127.0.0.1:11500/v1/audio/speech \
  -H "content-type: application/json" \
  -d '{
    "model": "qwen3-tts-12hz-0.6b-customvoice",
    "input": "Hello, this is a test of the local TTS gateway.",
    "voice": "alloy",
    "response_format": "wav"
  }' \
  --output out.wav
```

### Example: voice cloning (extension field passthrough)

```bash
curl -X POST http://127.0.0.1:11500/v1/audio/speech \
  -H "content-type: application/json" \
  -d '{
    "model": "qwen3-tts-12hz-0.6b-customvoice",
    "input": "Cloned-voice output.",
    "task_type": "voice_clone",
    "ref_audio": "/path/to/reference.wav",
    "ref_text": "Reference transcript matching the audio."
  }' \
  --output cloned.wav
```

The proxy is transparent — every field you pass and every byte you
get back come straight from Qwen3-TTS. See the upstream's
documentation for the full surface; aistack will keep relaying as the
upstream evolves.

---

## 6. Step 4 — Chat completion (LLM)

`POST /v1/chat/completions` mirrors OpenAI's chat API. Behind the
scenes aistack proxies to a locally-running Ollama daemon and adds two
gateway-level behaviors:

1. **`asr-main` cache eviction before forwarding.** If aistack was
   serving an ASR backend recently, its model is dropped from cache
   before the LLM request is forwarded so VRAM frees up for Ollama's
   model load.
2. **`keep_alive` default of `"30s"`** when the client omits the
   field. Sequential LLM calls within 30 seconds reuse the loaded
   model; idle Ollama returns VRAM to whoever needs it next. Override
   explicitly for long-running agent sessions.

### Example: non-streaming

```bash
curl -X POST http://127.0.0.1:11500/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{
    "model": "qwen3:4b",
    "messages": [
      {"role": "user", "content": "Translate to Chinese: hello world"}
    ]
  }'
```

### Example: streaming with cancel

```python
import httpx

req = {
    "model": "qwen3:4b",
    "messages": [{"role": "user", "content": "Write a short poem."}],
    "stream": True,
}

with httpx.stream("POST",
                   "http://127.0.0.1:11500/v1/chat/completions",
                   json=req,
                   timeout=600.0) as r:
    for line in r.iter_lines():
        if not line.startswith("data: "):
            continue
        if line == "data: [DONE]":
            break
        chunk = json.loads(line[6:])
        delta = chunk["choices"][0]["delta"].get("content", "")
        print(delta, end="", flush=True)
        if user_pressed_cancel():
            break   # closing the stream propagates a TCP RST upstream;
                    # aistack stops pulling from Ollama and frees the slot.
```

You do not need a separate cancel endpoint. **Closing the HTTP
response is the cancel signal.** aistack's stream loop polls for
client disconnect and propagates the abort to the upstream model.

---

## 7. The single-task GPU slot

This is the most important gateway behavior to understand because it
shapes how clients should structure parallelism.

**At most one inference workload runs on the GPU at any moment**,
across all three capabilities. Concurrent calls to ASR + LLM (or LLM
+ TTS, etc.) result in:

- The first request acquires the slot and runs.
- The second request returns immediately with HTTP **503** + a
  `Retry-After: 5` header and an error envelope of kind `network`.
- The client decides whether to back off and retry, or surface an
  error.

This is intentional. On 8 GB consumer cards, concurrent inference
across capabilities OOMs the worker; serializing is the only way to
make the gateway stable. On bigger hardware the same policy means
predictable resource accounting at no real throughput cost.

**Client guidance**:

- Treat 503 + `Retry-After` as a **transport-level back-pressure
  signal**, not a fatal error. Sleep the indicated seconds and retry.
- Do not pipeline calls into aistack expecting parallelism — design
  for serial dispatch with retry on contention.
- For agent loops that interleave ASR / LLM / TTS, run the steps
  sequentially. aistack's hot-swap policy will keep VRAM available
  even though the model changes between steps.

A simple Python retry helper:

```python
import time, httpx

def call_with_retry(method, url, max_attempts=5, **kw):
    for attempt in range(max_attempts):
        r = httpx.request(method, url, **kw)
        if r.status_code != 503:
            return r
        retry = float(r.headers.get("Retry-After", "5"))
        time.sleep(retry)
    r.raise_for_status()
```

---

## 8. Error envelope

Every non-2xx response carries this shape, including the slot-busy
503 from §7:

```json
{
  "error": {
    "kind": "network | malformed | overflow | cancelled | unknown",
    "provider": "aistack | Faster-Whisper | Parakeet | SenseVoice | ...",
    "message": "human-readable details safe to surface to users"
  }
}
```

Branch on `error.kind`. The five kinds mean:

| Kind | When | HTTP status | Client response |
|---|---|---|---|
| `network` | Upstream backend unreachable, model download failed, transport error | 503 | Show "service is down, please start it"; retry after delay |
| `malformed` | Bad input — unsupported audio format, missing field, unknown model id | 400 | Show error to user; do not retry |
| `overflow` | Input too large for the chosen model / VRAM | 413 | Suggest a smaller model or shorter input |
| `cancelled` | Client disconnected mid-request | 499 | Usually no UI needed; user already knows they cancelled |
| `unknown` | Anything that did not fit the categories above | 500 | Log and surface message; do not retry without diagnosis |

Full reference: [`errors.md`](errors.md).

---

## 9. Capability-specific notes

### ASR

- The `language` form field is a hint, not a constraint. faster-
  whisper auto-detects when `language` is omitted; SenseVoice ignores
  the hint when the audio's actual language conflicts.
- `response_format`: `json` (minimal `{text}`), `verbose_json` (full
  shape with segments + words), `text` (plain text body).
- `translate=true` only works on Whisper-family backends. Parakeet
  and SenseVoice raise `malformed` if asked to translate.
- For long audio (>5 min), Parakeet on 8 GB hardware uses the local-
  attention encoder mode automatically — accuracy is near-identical
  to full attention but the memory ceiling moves from ~3 min to
  effectively unbounded.

### TTS

- The proxy is transparent; if Qwen3-TTS adds new fields upstream,
  aistack relays them without code change.
- The TTS container holds a fixed VRAM reservation at startup
  (configurable via `gpu_memory_utilization` in
  `docker/tts_qwen3/docker-compose.yml`). On 8 GB cards, lower it to
  0.5 if the gateway feels tight.

### LLM

- Streaming uses Server-Sent Events with the standard OpenAI shape.
- `keep_alive` accepts the same string values Ollama does (`"30s"`,
  `"5m"`, `"0"`, etc.). aistack injects `"30s"` only when the field
  is absent.
- Cloud-only LLMs (DeepSeek, Claude, Gemini) are explicitly *out of
  scope* for aistack. Call those APIs directly from your client;
  there is no gain to proxying them.

---

## 10. Health check

```bash
curl http://127.0.0.1:11500/health
```

```json
{"status": "ok", "version": "0.0.1"}
```

Connection refused means aistack is not running. A 200 means the
worker is alive but does not certify any specific backend — combine
with `/v1/models` to find out which capabilities are usable.

---

## 11. Versioning & stability

### What is contract within `/v1`

The following are stable and only ever change in additive ways within
`/v1`:

- The set of endpoints listed in §2.
- Field names and types in `/v1/models` entries.
- The error envelope shape (`error.kind`, `error.provider`,
  `error.message`).
- The five error kinds in §8 and their HTTP status mappings.
- The 503 + `Retry-After` semantics for slot contention.
- The OpenAI-shape request/response bodies for the three capability
  endpoints.

### What is **not** contract

- Whether a specific model id appears in `/v1/models`. That depends on
  installed backends and is environment-specific.
- Specific timing characteristics (latency, throughput).
- Internal headers, log lines, or admin UI shape.
- Whether a particular backend is hosted in-process or proxied.

### Breaking changes

A breaking change to anything in the contract list above requires
bumping to `/v2`. `/v1` will continue to serve for at least one
release cycle so consumers can migrate without coupled deploys.

---

## 12. Where to look next

| You want to ... | Read |
|---|---|
| Understand the inventory response in detail | [`models.md`](models.md) |
| Build an ASR client | [`asr.md`](asr.md) |
| Build a TTS client | [`tts.md`](tts.md) |
| Build an LLM client | [`llm.md`](llm.md) |
| Branch on errors precisely | [`errors.md`](errors.md) |
| Stitch your trace IDs through aistack (optional `X-Request-ID`) | [`observability.md`](observability.md#optional-send-x-request-id-for-cross-system-log-correlation) |
| Read the metrics / access log / payload capture aistack records about your traffic | [`observability.md`](observability.md) |
| Understand aistack's internal architecture (not needed for integration) | aistack repo's `docs/design/architecture.md` (internal-only) |

If something behaves differently from what this guide promises, the
guide is the authority — please file an issue against
`github.com/dosmoon/aistack`.
