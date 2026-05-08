---
title: POST /v1/audio/speech
description: Text-to-speech endpoint. OpenAI TTS API compatible. Proxies to the Qwen3-TTS model running in a local Docker container.
sidebar:
  order: 4
---

# `POST /v1/audio/speech`

Generates speech audio from text. OpenAI TTS API compatible at the
field level — clients written against OpenAI's `/v1/audio/speech` work
unchanged. aistack proxies to the Qwen3-TTS-12Hz-0.6B-CustomVoice
model running inside the `aistack-qwen3-tts` Docker container.

## Request

`Content-Type: application/json`

OpenAI-compatible fields:

| Field | Required | Type | Description |
|---|---|---|---|
| `input` | yes | string | Text to synthesize. UTF-8. |
| `voice` | yes | string | Voice id. Defaults: `vivian` (the model's pretrained voices include `vivian`, `dylan`, `aiden`, `eric`, `ono_anna`, `ryan`, ...). |
| `response_format` | no | string | `"wav"` (effective default; vLLM-Omni emits 24 kHz mono PCM WAV regardless of this hint right now). |
| `model` | no | string | Currently the upstream container serves a single model and rejects mismatched ids. Omit unless you need to target a specific model in a future multi-model deployment. |

Qwen3-TTS extension fields (not in OpenAI spec; aistack passes them
through to vLLM-Omni):

| Field | Type | Description |
|---|---|---|
| `task_type` | string | `"CustomVoice"` (default) — pretrained voices. `"VoiceClone"` — clone from `ref_audio`. `"VoiceDesign"` — synthesize a voice per `instructions`. |
| `language` | string | `"English"`, `"Chinese"`, etc. Free-form upstream string, not ISO 639-1. |
| `instructions` | string | When `task_type=VoiceDesign`, describes the voice characteristics (timbre, age, emotion). |
| `ref_audio` | string | When `task_type=VoiceClone`, a URL / base64 / `file://` path to the voice sample. |
| `ref_text` | string | Transcript of `ref_audio` for VoiceClone alignment. |
| `max_new_tokens` | int | Generation cap; clip excessively long output. |

### Example

```bash
curl -X POST http://127.0.0.1:11500/v1/audio/speech \
     -H "Content-Type: application/json" \
     -d '{
           "input": "Hello world from aistack",
           "voice": "vivian",
           "task_type": "CustomVoice",
           "language": "English"
         }' \
     --output out.wav
```

## Response

```http
HTTP/1.1 200 OK
Content-Type: audio/wav
```

Raw audio bytes. The body is the entire WAV file (RIFF header + PCM
samples) — `/v1/audio/speech` itself is non-streaming.

Streaming synthesis is reachable through the proxy via vLLM-Omni's
`/v1/audio/speech/stream` endpoint (see *Pass-through endpoints*
below). The TTS entry in [`/v1/models`](models.md) advertises
`supports_streaming: true` because of this pass-through path; the
streaming wire format is whatever vLLM-Omni emits there (chunked
audio bytes), not the `transcript.text.delta` SSE shape used by ASR.

aistack does not transcode. Output is whatever vLLM-Omni emits — at
the time of writing, **24 kHz mono 16-bit PCM WAV** regardless of the
`response_format` hint. If you need MP3 or another container, transcode
client-side with ffmpeg.

## Pass-through endpoints

The full Qwen3-TTS extended surface is reachable via aistack:

| Path | Purpose |
|---|---|
| `POST /v1/audio/speech` | Standard / cloned / instructed synthesis (this doc) |
| `POST /v1/audio/speech/stream` | Streaming synthesis — chunks audio as it generates |
| `POST /v1/audio/speech/batch` | Batched synthesis (multiple inputs per request) |
| `GET  /v1/audio/voices` | List available voices |
| `POST /v1/audio/voices` | Register a new voice |
| `DELETE /v1/audio/voices/{name}` | Remove a registered voice |

aistack proxies these paths verbatim — request body and response are
untouched. Refer to the [vLLM-Omni Qwen3-TTS docs](https://github.com/QwenLM/Qwen3-TTS)
for their full schemas. Future aistack versions may add value-added
behavior at this layer (telemetry, voice-list aggregation across
multi-backend deployments) without changing the over-the-wire format.

## Cold start

The first request to a freshly started Docker container triggers
`torch.compile` + CUDA Graph capture inside vLLM-Omni — this takes
~60 to ~150 seconds depending on the request and machine state. The
proxy timeout is set to **600 seconds** to absorb the worst case;
clients should display "warming up" rather than treating long latency
as a hang.

After warmup, steady-state latency on RTX 4060 Laptop is typically
RTF 0.7-1.1 (a few hundred milliseconds for a short utterance).

## Concurrency

vLLM-Omni handles its own request queueing inside the Docker
container. aistack does **not** apply the single-task GPU lock to
TTS — TTS work happens in a separate process with a separate CUDA
context, so blocking it at the gateway layer would just delay
requests for no benefit. If both ASR and TTS run concurrently on
the same physical GPU, they coexist or contend at the driver level;
aistack does not arbitrate.

## Error scenarios

All errors use the standard envelope from [errors.md](errors.md).

| HTTP | `kind` | Cause |
|---|---|---|
| 400 | `malformed` | Empty `input`, invalid `voice` for the active model, malformed `ref_audio` URL |
| 503 | `network` | Qwen3-TTS Docker container not running. Recover with `docker compose -f docker/tts_qwen3/docker-compose.yml up -d`. |
| 502 | `unknown` | vLLM-Omni returned a non-200 we could not decode |
| 504 | `network` | Cold-start exceeded the 600s read timeout — the engine is in a stuck state. Restart the container. |

## Stability

The OpenAI-compatible field layer (`input`, `voice`, `response_format`,
`model`) is stable within `/v1`. The Qwen3-TTS extension fields
(`task_type`, `language`, `ref_audio`, ...) are documented as a
stable contract for the duration that aistack ships Qwen3-TTS as the
TTS backend; if a future TTS backend exposes a different surface,
aistack will normalize but the extension-field set may change.

The pass-through endpoints (`/v1/audio/speech/stream`, `/voices`, ...)
follow vLLM-Omni's contract and are subject to upstream changes.
