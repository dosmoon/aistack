---
title: Reference
description: Auto-generated HTTP API reference for aistack — endpoints, schemas, error codes.
sidebar:
  order: 9
---

<!-- AUTO-GENERATED: do not edit. Source: aistack/api/* docstrings + Pydantic models, rendered by scripts/gen_api_reference.py. -->

# HTTP API reference

These pages are generated from the live FastAPI app's OpenAPI spec on every build. The source of truth is the docstrings and Pydantic models in `aistack/api/`; editing the rendered markdown has no effect.

For the design rationale and integration journey (the *why*, not the *what*), start with the [Integration Guide](../integration/).

## Sections

- [Inventory & health](./models/) — Auto-generated reference for GET /health and GET /v1/models — what the gateway can serve right now.
- [ASR — speech to text](./asr/) — Auto-generated reference for POST /v1/audio/transcriptions. OpenAI Whisper API compatible.
- [TTS — text to speech](./tts/) — Auto-generated reference for the /v1/audio/{path} reverse proxy to Qwen3-TTS.
- [LLM — chat completion](./llm/) — Auto-generated reference for POST /v1/chat/completions, reverse-proxied to local Ollama.
- [Admin runtime controls](./admin/) — Auto-generated reference for the runtime control endpoints under /admin/api/.
