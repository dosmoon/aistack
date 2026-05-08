---
title: aistack
description: Local-first ASR / TTS / LLM-proxy gateway with an OpenAI-compatible API. Research-shaped open source for developers and researchers studying local AI capabilities.
---

# aistack

A self-hosted, OpenAI-API-compatible gateway that wraps open-source ASR (faster-whisper, NVIDIA Parakeet, Alibaba SenseVoice), TTS (Qwen3-TTS via vLLM), and a local LLM proxy (to Ollama) behind a single HTTP endpoint.

## Positioning

aistack is **research-shaped, not productized**. It is built for developers and researchers who want to compare backends, measure quality/latency trade-offs, and integrate local ASR/TTS into their own tools — not for end users who expect a single-installer "download and run" experience. The emphasis is on observability, multi-backend comparison, and unsentimental benchmarks.

If you want a turnkey local-AI experience, use [Ollama](https://ollama.com) directly for LLMs and wait for a downstream productized tool for ASR/TTS. If you want to *study* local ASR/TTS, aistack is built for you.

## What's published here

| Section | What it is |
|---|---|
| [HTTP API](api/) | The contract aistack publishes to consumers. Start at the [Integration Guide](api/integration/). |

More sections (configuration, deployment notes, research findings) will be progressively published. The full repo lives at [github.com/dosmoon/aistack](https://github.com/dosmoon/aistack).

## Quick start

```bash
git clone https://github.com/dosmoon/aistack
cd aistack
pip install -e .[asr-fasterwhisper]
python -m uvicorn aistack.main:app --port 11500
curl http://127.0.0.1:11500/health
```

The full install + extras layout (per-ASR-backend opt-in, TTS Docker container, Ollama side-by-side) is documented in the repo's `README.md`.

## API at a glance

```
GET  /health                        liveness
GET  /v1/models                     capability inventory
POST /v1/audio/transcriptions       speech-to-text  (Whisper / Parakeet / SenseVoice)
POST /v1/audio/speech               text-to-speech  (Qwen3-TTS)
POST /v1/chat/completions           chat completion (proxy to Ollama)
```

Default base URL: `http://127.0.0.1:11500`. For the per-endpoint reference, head to the [HTTP API section](api/).
