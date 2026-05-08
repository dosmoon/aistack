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

aistack adds **no business logic** at this layer; it is a transparent
reverse proxy. Request body and response body flow through unchanged
(minus hop-by-hop headers).

The full request and response schema lives in the auto-generated
[TTS reference](./reference/tts/). For OpenAI-compatible field
semantics (`input`, `voice`, `response_format`, `model`), the
authoritative reference is OpenAI's
[Audio API documentation](https://platform.openai.com/docs/api-reference/audio).
This page covers the *why* — the transparent-proxy stance, output
format quirks, the pass-through endpoint surface, cold-start
behaviour, and how aistack does **not** arbitrate TTS concurrency.

## Why a transparent proxy

Three reasons aistack does not transform TTS requests/responses:

1. **OpenAI's contract is the contract.** Clients written for
   OpenAI's TTS endpoint work unchanged against aistack. Adding
   business logic in the middle would require keeping it in sync
   with OpenAI's evolving spec — drift surface for no value.
2. **The upstream owns the model.** Qwen3-TTS extension fields
   (voice cloning, voice design, batched synthesis) are the
   upstream's domain. aistack passes them through so consumers
   reach the full feature set.
3. **Audio is bytes, not JSON.** Transcoding (WAV → MP3, sample-rate
   conversion) at the gateway would burn CPU and add latency for
   every request. Clients that need a different format transcode
   client-side with ffmpeg.

## Output format

Output is whatever vLLM-Omni emits — at the time of writing,
**24 kHz mono 16-bit PCM WAV** regardless of the `response_format`
hint in the request. The vLLM-Omni server may add MP3/Opus/FLAC
encoding in a future release; aistack will pass that through
without code changes.

If you need a specific format today, transcode client-side:

```bash
ffmpeg -i out.wav -codec:a libmp3lame out.mp3
```

## Pass-through endpoint surface

The full Qwen3-TTS extended surface is reachable through aistack
under `/v1/audio/*`:

- `POST /v1/audio/speech` — synthesis (custom voice / cloned voice / designed voice)
- `POST /v1/audio/speech/stream` — streaming synthesis (chunks audio as it generates)
- `POST /v1/audio/speech/batch` — batched synthesis
- `GET /v1/audio/voices` — list available voices
- `POST /v1/audio/voices` — register a new voice
- `DELETE /v1/audio/voices/{name}` — remove a registered voice

aistack proxies all of these verbatim. Their schemas are owned by
[vLLM-Omni's Qwen3-TTS docs](https://github.com/QwenLM/Qwen3-TTS);
aistack does not redocument them. Future aistack versions may add
value-added behaviour (telemetry, voice-list aggregation across
multi-backend deployments) without changing the over-the-wire format.

The `/v1/models` entry for TTS advertises `supports_streaming: true`
because of the `/v1/audio/speech/stream` pass-through path; the
streaming wire format is whatever vLLM-Omni emits there (chunked
audio bytes), **not** the `transcript.text.delta` SSE shape used by
ASR.

## Cold start

The first request to a freshly started Docker container triggers
`torch.compile` + CUDA Graph capture inside vLLM-Omni — this takes
**~60 to ~150 seconds** depending on the request and machine state.
The proxy timeout is set to 600 seconds to absorb the worst case;
clients should display "warming up" rather than treating long
latency as a hang.

After warmup, steady-state latency on RTX 4060 Laptop is typically
RTF 0.7–1.1 (a few hundred milliseconds for a short utterance).

## Concurrency

aistack **does** apply the single global GPU slot to TTS requests:
the Qwen3-TTS container shares the physical GPU with in-process ASR
and the LLM proxy, and the slot represents "GPU is doing inference"
regardless of which process owns the kernels. Concurrent requests
across capabilities get HTTP 503 with `Retry-After`.

vLLM-Omni handles its own request queueing inside the Docker
container. aistack does not implement a queue at the gateway layer
— if you want a queue, build it client-side around the busy signal.

## Recovery from container down

The most common failure mode is "Docker container is not running":

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

The error message includes the recovery command. Containers that
crash mid-cold-start can wedge in a state where requests time out at
the proxy's 600 s read timeout — restart the container in that case.

## Stability

The OpenAI-compatible field layer (`input`, `voice`,
`response_format`, `model`) is stable within `/v1`. The Qwen3-TTS
extension fields (`task_type`, `language`, `ref_audio`, ...) are
the upstream's contract and are documented authoritatively in the
[vLLM-Omni Qwen3-TTS repo](https://github.com/QwenLM/Qwen3-TTS).

The pass-through endpoint paths (`/v1/audio/speech/stream`,
`/voices`, ...) follow vLLM-Omni's contract; if a future TTS
backend exposes a different surface, aistack will document the
mapping in a separate migration note rather than silently rewrite
the contract.
