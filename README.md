# aistack

> Localhost AI service for ASR and TTS — designed to fill the gap left by Ollama (which already covers LLMs perfectly).

`aistack` is a self-hosted, OpenAI-API-compatible service that wraps open-source ASR (faster-whisper, Parakeet, SenseVoice) and TTS (Qwen3-TTS via vLLM-Omni) models behind a single HTTP endpoint. Other tools — including [VideoCraft](https://github.com/dosmoon/videocraft) — consume it as a pure HTTP client and never deal with model lifecycle directly.

## Status

🚧 **Early development.** D1 phase: bare skeleton, no models loaded yet.

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
