# aistack

> Localhost AI service for ASR and TTS — designed to fill the gap left by Ollama (which already covers LLMs perfectly).

`aistack` is a self-hosted, OpenAI-API-compatible service that wraps open-source ASR (faster-whisper, Parakeet, SenseVoice) and TTS (Qwen3-TTS via vLLM-Omni) models behind a single HTTP endpoint. Other tools — including [VideoCraft](https://github.com/dosmoon/videocraft) — consume it as a pure HTTP client and never deal with model lifecycle directly.

## Positioning

**aistack is an open, exploratory, research-grade open-source project.** It is built for developers and researchers who want to compare backends, measure quality/latency trade-offs, and integrate local ASR/TTS into their own tools — not for end users who expect a single-installer "download and run" experience.

What this means in practice:

- **No release artifact, no installer.** The supported install path is `git clone` + `pip install -e .[asr-...]`. Heavy ML deps are opt-in via extras so you only pay for the backends you actually use.
- **Heavy / cutting-edge dependencies are welcomed**, not avoided. NeMo, vLLM, Docker-bound TTS — if it gives a meaningful research signal, it stays. We do not optimize for PyInstaller-friendliness or for users without a working CUDA stack.
- **Observability is first-class**, deeper than what a productized tool would expose: rolling p50/p95/p99 per capability, per-request payload capture, JSONL access logs, bench harness for cross-backend WER/RTF comparison. These exist because the project's value is in *understanding* the backends, not just running them.
- **The admin UI is a runtime control surface, not a package manager.** It manages live state (cache, GPU slot, Ollama models, observability toggles). It does not mutate the venv, persist env vars, or manage external services like Docker — those belong to the developer's own toolchain.
- **aistack is a pathfinder for future productization.** When dosmoon eventually packages a consumer-grade tool, the cross-backend benchmarks, latency data, and quality measurements gathered through aistack inform what that product looks like. aistack itself stays research-shaped.

If you want a turnkey local-AI experience, use [Ollama](https://ollama.com) directly for LLMs and wait for a downstream productized tool for ASR/TTS. If you want to *study* local ASR/TTS — pick a backend, measure trade-offs, integrate into your own pipeline — aistack is built for you.

## Status

**Working gateway, actively evolving.** ASR + TTS + LLM-proxy capabilities are all live; observability is built in.

| Capability | State |
|---|---|
| ASR — `POST /v1/audio/transcriptions` | faster-whisper / Parakeet / SenseVoice; SSE streaming on whisper + sensevoice; `model="auto"` language-aware routing; long-audio chunked mode for Parakeet |
| TTS — `POST /v1/audio/speech` | Qwen3-TTS via vLLM-Omni Docker container |
| LLM — `POST /v1/chat/completions` | reverse-proxy to local Ollama with ASR-cache eviction before forwarding |
| Models metadata — `GET /v1/models` | per-entry `languages`, `supports_streaming`, `auto` routing alias |
| Admin UI — `/admin` | read-only dashboard (lock / GPU mem / models / cache / logs / metrics); runtime controls for cache reset and observability toggles |
| Observability | rolling p50/p95/p99 per capability, JSONL access log, per-request payload capture, `X-Request-ID` correlation |
| Bench harness | LibriSpeech WER, long-audio RTF, cross-backend comparison aggregator |

See `docs/progress/` for the day-by-day chronicle and `docs/backlog.md` for what's queued next.

## Scope

| | |
|---|---|
| ✅ In scope | ASR (faster-whisper / Parakeet / SenseVoice), TTS (Qwen3-TTS) |
| ❌ Out of scope | LLM — use [Ollama](https://ollama.com) directly; aistack will not duplicate it |
| 🔮 Future | Local video / vision-language models, where no Ollama-equivalent exists |

## Default endpoint

```
http://127.0.0.1:11500
```

OpenAI-compatible:
- `POST /v1/audio/transcriptions` — ASR
- `POST /v1/audio/speech` — TTS
- `GET  /v1/models` — list installed models
- `GET  /health` — readiness probe

## Quick start (dev)

```bash
pip install -e .
python -m uvicorn aistack.main:app --port 11500 --reload
curl http://127.0.0.1:11500/health
```

Or on Windows:

```
scripts\dev.bat
```

## Naming convention

- Repository / package / docs / CLI / files: `aistack`
- UI title bar (when shown to end users): `dosmoon-aistack`

## License

MIT
