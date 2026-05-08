---
title: POST /v1/audio/transcriptions
description: Speech-to-text endpoint. OpenAI Whisper API compatible. Supports faster-whisper, NVIDIA Parakeet, Alibaba SenseVoice, and language-aware auto routing.
sidebar:
  order: 3
---

# `POST /v1/audio/transcriptions`

Transcribes an audio file. OpenAI Whisper API compatible — clients
written against OpenAI's Whisper endpoint can point at aistack with
no code changes other than the base URL.

## Request

`Content-Type: multipart/form-data`

| Field | Required | Type | Description |
|---|---|---|---|
| `file` | yes | binary | Audio file. Any ffmpeg-readable format: mp3, mp4, wav, m4a, flac, ogg, webm, mkv. |
| `model` | yes | string | Model id or short alias. See *Model selection* below. |
| `language` | no | string | ISO 639-1 hint (`"en"`, `"zh"`, `"ja"`, ...). Omit or pass `null`/`""` for auto-detect. |
| `response_format` | no | string | `"json"` (default) \| `"verbose_json"` \| `"text"`. Ignored when `stream=true`. |
| `translate` | no | bool | If `true`, transcribe to English instead of source language. Whisper-family only; Parakeet / SenseVoice reject with 400. |
| `stream` | no | bool | If `true`, return Server-Sent Events with one `transcript.text.delta` per segment, ending with `transcript.text.done`. See [`integration.md` §4 — Streaming transcription](integration.md#streaming-transcription-with-streamtrue) for the wire format. Models advertising `supports_streaming=false` in `/v1/models` (currently Parakeet) emit a `warning` event followed by a single delta containing the full transcription. |
| `segment_granularity` | no | string | `"sentence"` (default) \| `"subtitle"`. How to group word timestamps in the `segments` field. **Affects Parakeet only**; faster-whisper and SenseVoice produce VAD-driven sentence-level segments natively. See [Segment granularity](#segment-granularity) below. |

### Model selection

The `model` field accepts any of:

| Form | Backend | Notes |
|---|---|---|
| `whisper-tiny` `whisper-base` `whisper-small` `whisper-medium` `whisper-large-v3` `whisper-large-v3-turbo` `whisper-distil-large-v3` | faster-whisper (CTranslate2) | All Whisper sizes from the OpenAI catalog. |
| `whisper-1` | faster-whisper | Maps to `whisper-small` for OpenAI-spec compatibility (OpenAI's only public Whisper id). |
| `tiny` `base` `small` `medium` ... | faster-whisper | Bare size aliases — same as the `whisper-` prefix forms. |
| `parakeet` or `nvidia/parakeet-tdt-0.6b-v3` | NVIDIA NeMo | 25 European languages, ASR-only (no `translate`). Does not support streaming — see *Streaming* below. |
| `sensevoice` or `iic/SenseVoiceSmall` | Alibaba FunASR | Mandarin / Cantonese / Japanese / Korean / English. Best for CJK content. |
| `auto` (or empty) | aistack router | Picks based on `language`: CJK → SenseVoice, European → Parakeet, else → Whisper-small. Falls back gracefully when a preferred backend is not installed. |

### Multipart example

```bash
curl -X POST http://127.0.0.1:11500/v1/audio/transcriptions \
     -F file=@speech.mp3 \
     -F model=whisper-small \
     -F language=en \
     -F response_format=verbose_json
```

## Response

The shape depends on `response_format`.

### `response_format=json` (default)

```json
{
  "text": "Warm steady state. They should be fast."
}
```

OpenAI-spec minimal shape. Use when only the transcript text is needed.

### `response_format=verbose_json`

```json
{
  "language": "en",
  "duration": 3.28,
  "text":     "Warm steady state. They should be fast.",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.6,
      "text": "Warm steady state. They should be fast.",
      "avg_logprob": -0.504,
      "no_speech_prob": 0.053,
      "compression_ratio": 0.886,
      "temperature": 0.0
    }
  ],
  "words": [
    { "start": 0.0,  "end": 0.44, "word": "Warm" },
    { "start": 0.44, "end": 0.78, "word": "steady" },
    { "start": 0.78, "end": 1.26, "word": "state." },
    { "start": 1.66, "end": 1.82, "word": "They" },
    { "start": 1.82, "end": 2.02, "word": "should" },
    { "start": 2.02, "end": 2.22, "word": "be" },
    { "start": 2.22, "end": 2.60, "word": "fast." }
  ]
}
```

The shape mirrors OpenAI's verbose_json. Word-level timestamps are
included by all aistack ASR backends; segment-level diagnostics
(`avg_logprob`, `no_speech_prob`, `compression_ratio`, `temperature`)
are populated when the backend reports them and absent otherwise — so
clients should treat them as optional.

### `response_format=text`

```http
HTTP/1.1 200 OK
Content-Type: text/plain

Warm steady state. They should be fast.
```

Plain text body, no JSON wrapping.

## Segment granularity

The `segments` array in `verbose_json` (and the per-segment SSE deltas
under `stream=true`) can be grouped two ways:

| Value | Use this when ... | Example cue from the same audio |
|---|---|---|
| `"sentence"` (default) | downstream consumes segments as semantic units — line-by-line LLM translation, semantic search, agent reasoning, alignment with another transcription. Each segment is a complete sentence; cue length is unbounded (within a 30 s safety cap). | `"I'll be filling in for Caroline today, obviously."` |
| `"subtitle"` | downstream emits SRT/VTT directly without doing its own cue-sizing. Each segment is ≤ 70 chars and 1–7 s, following the stable-ts / subtitle-localisation industry standard. **Cuts mid-sentence when the sentence is too long.** | `"So I'll have a brief remarks here and then we'll get to your"` |

### Why default to `"sentence"`

`"subtitle"` was the right shape for one specific consumer (a direct
SRT writer), but feeding subtitle-sized cues into a per-row LLM
translator produces broken translations: half-clauses lose tense and
referent, the LLM has to guess the missing context, target-language
output becomes incoherent.

The industry convention (OpenAI Whisper verbose_json, WhisperX,
stable-ts, FunASR) is **`segments` = sentence-level**, with SRT
cue-sizing as a separate downstream step. aistack follows the same
convention.

### When to use `"subtitle"`

- A pipeline that **only** generates SRT/VTT and doesn't do
  any further transformation. Saves implementing your own cue-sizing.
- A pipeline that already has good downstream cue-sizing and just
  wants aistack to provide the final formatting for an ASR-only path.

### When to stick with `"sentence"` (default)

- **Any LLM translation pipeline.** Even if the ultimate output is SRT,
  translate first at sentence granularity, then cue-size the
  translated text — translation quality dominates everything else.
- Semantic search, alignment, summarisation, agent reasoning over the
  transcript.
- Any consumer that does its own sentence detection and would have to
  glue subtitle cues back together.

For pipelines that need both (translate sentences AND emit SRT),
the recommended flow is: `segment_granularity=sentence` for translation,
then run a downstream cue-sizer (your own, or a library like stable-ts)
on the translated sentence-level segments to produce SRT.

## Backend-specific behaviors to know

### faster-whisper (Whisper sizes)

- Default `device="auto"` selects CUDA when `torch.cuda.is_available()`,
  otherwise CPU (int8 quantization on CPU).
- Smaller sizes are faster but less accurate; pick by the speed/quality
  trade-off your use case needs.
- Supports `translate=true` to convert non-English audio to English.

### Parakeet (NVIDIA NeMo)

- Auto-detects across 25 European languages plus English. The
  `language` hint is recorded in the response but does not constrain
  the model — the model decides.
- `translate=true` is **not supported**; the request returns a 400
  `malformed` error.
- Best for European-language ASR; do not route Mandarin/Cantonese/
  Japanese/Korean here (see SenseVoice).

### SenseVoice (Alibaba FunASR)

- Strongest for Mandarin and Cantonese; also covers Japanese, Korean,
  and English (English output omits inter-word spaces — a quirk of the
  CJK-centric tokenizer; post-process if needed).
- `language` accepts `"auto"`, `"zh"`, `"en"`, `"yue"`, `"ja"`, `"ko"`.
- `translate=true` is **not supported**; returns 400 `malformed`.
- VAD + main model are loaded as a pair; on first use both load
  sequentially (~20-50s cold start on CPU, ~20s on GPU).

## Cold start

The first request to a given backend pays the model load cost. On
subsequent requests within the cache idle window (default 5 min,
configurable via `AISTACK_MODEL_KEEP_ALIVE_SEC`), the model is hot
and only inference latency applies.

Approximate first-call latencies on RTX 4060 Laptop:

| Backend | Cold (load + first inference) | Warm (inference only, 200s audio) |
|---|---|---|
| faster-whisper-small | ~14s | ~10s |
| Parakeet TDT 0.6B | ~25s | ~2s |
| SenseVoice Small | ~20s | ~6s |

CPU-mode latencies are several times higher; see [runtimes.md](../selection/runtimes.md)
for the full benchmark and the rationale behind selective GPU
deployment on tight VRAM.

## Concurrency

aistack serializes ASR inference: at most one transcription at a time
on this worker. Concurrent requests get an immediate **HTTP 503** with
`Retry-After: 5` (no queueing). See [errors.md](errors.md) for the
exact response shape. Callers should respect the retry hint and back
off rather than hammering.

## Error scenarios

All errors use the standard envelope from [errors.md](errors.md).
Common cases for this endpoint:

| HTTP | `kind` | Cause |
|---|---|---|
| 400 | `malformed` | Audio file missing, unknown `model`, `translate=true` on a non-Whisper model |
| 413 | `overflow` | Audio longer than the chosen backend's window (rare; mostly applies to fixed-length transducers) |
| 499 | `cancelled` | Client disconnected mid-request |
| 503 | `network` | Backend not installed (e.g. NeMo missing for `parakeet`); model download failed |
| 503 | (no envelope, has `Retry-After`) | Another ASR call is in progress; busy signal |
| 500 | `unknown` | Unclassified failure; report as a bug |

## Stability

OpenAI-spec request fields (`file`, `model`, `language`,
`response_format`) and response shapes (`text`, `verbose_json`) are
stable within `/v1`. The `translate` field follows OpenAI-spec.

The set of accepted `model` values **changes** as backends are added
or removed; treat it as configuration, not contract. Use `GET /v1/models`
to discover the current set.
