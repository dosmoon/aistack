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

The full request and response schema (every form field, every response
variant, every error code) lives in the auto-generated
[ASR reference](./reference/asr/). This page covers the *why* —
backend selection rationale, segment-granularity design choices,
backend-specific behaviours, performance envelope, and concurrency
contract.

## Choosing a backend

The `model` field accepts faster-whisper sizes (`whisper-tiny` ...
`whisper-large-v3-turbo`, `whisper-distil-large-v3`), the Parakeet
HuggingFace id, the SenseVoice HuggingFace id, or the `auto` routing
alias. The exact list of accepted forms lives in the
[reference](./reference/asr/) — this section explains *which* you
should use:

- **English long-form, throughput-sensitive** → `parakeet`
  (very fast, no translation, no streaming).
- **Mandarin / Cantonese / Japanese / Korean** → `sensevoice`
  (designed for CJK; routes here from `auto` when `language` is CJK).
- **Anything multilingual or English short-form, want streaming or
  translation** → `whisper-{size}`. Pick the size by the
  speed/quality trade-off your use case needs.
- **Don't want to choose** → `auto` (or omit `model`). Aistack
  picks the best installed backend by `language`.

Backend coverage at a glance:

| Backend                              | Languages                          | Streaming | Translate (`task=translate`) |
|--------------------------------------|------------------------------------|-----------|------------------------------|
| faster-whisper (Whisper sizes)       | All 99 Whisper-supported           | yes       | yes (X → English only)       |
| Parakeet TDT v3                      | English + 24 European              | no¹       | **no**                       |
| SenseVoice Small                     | zh / yue / en / ja / ko            | yes       | **no**                       |

¹ The gateway accepts `stream=true` on Parakeet but emits a
single-event SSE downgrade with a `warning` event; it will not
silently chunk a non-streaming model and pay the WER cost.

## Segment granularity

The `segments` array in `verbose_json` (and the per-segment SSE deltas
under `stream=true`) can be grouped two ways:

- `"sentence"` (default) — each segment is a complete sentence; cue
  length unbounded within a 30 s safety cap. Right input for
  line-by-line LLM translation, semantic search, agent reasoning,
  alignment with another transcription.
- `"subtitle"` — each segment is ≤ 70 chars and 1–7 s, following the
  stable-ts / subtitle-localisation industry standard. **Cuts
  mid-sentence when the sentence is too long.**

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

- A pipeline that **only** generates SRT/VTT and doesn't do any
  further transformation. Saves implementing your own cue-sizing.
- A pipeline that already has good downstream cue-sizing and just
  wants aistack to provide the final formatting for an ASR-only path.

### When to stick with `"sentence"`

- **Any LLM translation pipeline.** Even if the ultimate output is
  SRT, translate first at sentence granularity, then cue-size the
  translated text — translation quality dominates everything else.
- Semantic search, alignment, summarisation, agent reasoning over
  the transcript.
- Any consumer that does its own sentence detection and would have
  to glue subtitle cues back together.

For pipelines that need both (translate sentences AND emit SRT),
the recommended flow is: `segment_granularity=sentence` for
translation, then run a downstream cue-sizer (your own, or
stable-ts) on the translated sentence-level segments.

`segment_granularity` only affects Parakeet. faster-whisper and
SenseVoice produce VAD-driven sentence-level segments natively.

## Backend-specific quirks worth knowing

### faster-whisper (Whisper sizes)

- Default `device="auto"` selects CUDA when `torch.cuda.is_available()`,
  otherwise CPU (int8 quantization).
- Smaller sizes are faster but less accurate; pick by the
  speed/quality trade-off your use case needs.
- Supports `translate=true` to convert non-English audio **to
  English** (X → en only — Whisper's training data does not include
  any other target language).

### Parakeet (NVIDIA NeMo)

- Auto-detects across 25 European languages plus English. The
  `language` hint is recorded in the response but does not constrain
  the model — the model decides.
- `translate=true` is **not supported**; returns 400 `malformed`.
- Best for European-language ASR; do not route Mandarin / Cantonese /
  Japanese / Korean here.

### SenseVoice (Alibaba FunASR)

- Strongest for Mandarin and Cantonese; also covers Japanese, Korean,
  and English (English output omits inter-word spaces — a quirk of
  the CJK-centric tokenizer; post-process if needed).
- `language` accepts `"auto"`, `"zh"`, `"en"`, `"yue"`, `"ja"`,
  `"ko"`.
- `translate=true` is **not supported**; returns 400 `malformed`.
- VAD + main model are loaded as a pair; on first use both load
  sequentially.

## Cold start

The first request to a given backend pays the model load cost. On
subsequent requests within the cache idle window (default 5 min,
configurable via `AISTACK_MODEL_KEEP_ALIVE_SEC`), the model is hot
and only inference latency applies.

Approximate first-call latencies on RTX 4060 Laptop (8 GB VRAM):

| Backend                | Cold (load + first inference) | Warm (200 s audio) |
|------------------------|------------------------------:|-------------------:|
| faster-whisper-small   |                          ~14s |              ~10 s |
| Parakeet TDT 0.6B      |                          ~25s |               ~2 s |
| SenseVoice Small       |                          ~20s |               ~6 s |

CPU-mode latencies are several times higher. See the
`docs/research-note/parakeet-on-consumer-gpu.md` and
`consumer-gpu-asr-baseline.md` files in the repo for the full
benchmarks and the rationale behind GPU-only deployment on tight
VRAM.

## Concurrency

aistack serializes ASR inference: at most one transcription at a
time on this worker (the same global GPU slot is shared with TTS
and LLM). Concurrent requests get an immediate **HTTP 503** with
`Retry-After: 5` (no queueing). See [errors](./errors/) for the
slot-busy envelope shape and the consumer retry pattern. Callers
should respect the retry hint and back off rather than hammering.

## Stability

OpenAI-spec request fields (`file`, `model`, `language`,
`response_format`, `translate`) and response shapes (`text`,
`verbose_json`) are stable within `/v1`.

The set of accepted `model` values **changes** as backends are added
or removed; treat it as configuration, not contract. Use
`GET /v1/models` to discover the current set per deployment.
