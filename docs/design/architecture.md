# aistack architecture

> Status: D1 skeleton phase
> Last updated: 2026-05-06

## Mission

Fill the gap left by Ollama: provide a localhost, OpenAI-API-compatible HTTP
service that wraps open-source ASR and TTS models. Other tools (VideoCraft and
beyond) consume `aistack` as a pure HTTP client and never deal with model
lifecycle directly.

## Scope

| | |
|---|---|
| ✅ In | ASR (faster-whisper, Parakeet, SenseVoice), TTS (Qwen3-TTS via vLLM-Omni) |
| ❌ Out | LLM — Ollama already covers it perfectly; aistack will not duplicate |
| 🔮 Future | Local video / VLM models, where no Ollama-equivalent exists |

## Service shape

- Single FastAPI process — no gateway in phase 1 (YAGNI)
- Default bind: `127.0.0.1:11500`
- Protocol: OpenAI API compatible
  - `POST /v1/audio/transcriptions` — ASR
  - `POST /v1/audio/speech` — TTS
  - `GET  /v1/models` — list installed models
  - `GET  /health` — readiness probe
  - `GET  /admin` — Web UI (HTMX/Jinja, planned phase 1)
- Streaming responses use SSE (OpenAI standard)

## Naming convention

- Repo / package / docs / CLI / files / logs: bare `aistack`
- UI title bar (only when shown to end users): `dosmoon-aistack`
- The `dosmoon-` prefix is a runtime brand marker, never a code identifier

## Phased roadmap

| Phase | Scope |
|---|---|
| **D1** | Bare FastAPI skeleton: `/health`, `/v1/models` returning empty list. No models. |
| **D2** | Migrate L3 Qwen3-TTS + vLLM-Omni from VideoCraft; wire `/v1/audio/speech`. Run Tier 1/2/3 perf tests here. |
| **D3** | Migrate ASR providers (faster-whisper / Parakeet / SenseVoice) from VideoCraft; wire `/v1/audio/transcriptions`. |
| **D4** | Web UI at `/admin` (HTMX): models list, install/remove, GPU mem, live RTF, log tail. |
| **D5** | VideoCraft client switch: replace in-process `*_local.py` with HTTP calls. Delete migrated source from VideoCraft. |

## Cross-process concerns (deferred design)

- **Cancellation semantics**: VideoCraft's `cancellation.py` is in-process. Once HTTP boundary is added, we need a cancel propagation contract (HTTP request abort vs. dedicated cancel endpoint).
- **Streaming latency**: in-process call has zero overhead; HTTP+SSE adds small overhead — measure on real-time ASR scenarios before committing.
- **Error type passthrough**: `AIError(Kind.X)` in VideoCraft must serialize across HTTP and reconstitute on client side. Define error envelope schema in D2.

## Out of repo

- LLM service — use Ollama's own installer at `localhost:11434/v1`
- VideoCraft business logic (translate, rewrite, prompt hub) — stays in VideoCraft
- Anything specific to a single consumer's UX
