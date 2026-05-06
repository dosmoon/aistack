# aistack architecture

> Status: design refresh 2026-05-06 — repositioned from "local AI service" to "AI capability gateway"

## Mission

aistack is the **local AI capability gateway** for dosmoon-system clients
(VideoCraft today; future tools tomorrow). Clients see a single OpenAI-
API-compatible endpoint at `127.0.0.1:11500` and request capabilities
(`/v1/audio/transcriptions`, `/v1/audio/speech`, `/v1/chat/completions`)
without any awareness of which backend serves the request, where the
backend physically runs, or how local resources are coordinated.

aistack's job is to:

1. **Abstract** — present a stable, vendor-neutral HTTP API to clients.
2. **Route** — pick which backend handles a given request based on
   capability (ASR / TTS / LLM), model selector, language hints, and
   resource availability.
3. **Schedule** — coordinate shared local resources (chiefly the GPU)
   across all backends so multi-stage workflows (ASR → LLM → TTS) do
   not OOM the worker.

This positioning replaces the earlier framing ("a thin wrapper for ASR
and TTS only; Ollama covers LLM"). The wrapper part is still real, but
the gateway part is now the primary identity.

## What the gateway does — and does not — do

| | |
|---|---|
| ✅ Owns | API surface; request routing; local-resource scheduling; model lifecycle on the host; per-backend health + readiness; future: backend pool across LAN/cloud |
| ✅ Hosts | ASR (faster-whisper / Parakeet / SenseVoice in-process), TTS (Qwen3-TTS via Docker sidecar) |
| ✅ Proxies | LLM (Ollama today; future: more local LLMs, LAN GPU peers, rented cloud GPU instances) |
| ❌ Reimplements | LLM inference — Ollama remains the actual runtime; aistack is a thin proxy in front for scheduling |
| ❌ Hosts | Cloud-only LLMs (DeepSeek API, Claude API, Gemini API). Clients call those directly — no value in aistack proxying remote endpoints that have no local resource cost |
| ❌ Owns | Client-side business logic (prompt templates, translation pipelines, multi-step orchestration) — those stay in the consumer (e.g. VideoCraft) |

## How clients see it

Every client of aistack is a **dumb HTTP consumer**: it sends an
OpenAI-style request and gets an OpenAI-style response. It does **not**:

- know which backend handled the request,
- know where the backend ran (this host / LAN peer / cloud),
- need to call any cache / unload / coordination endpoint,
- pass any aistack-internal scheduling hints.

A client wanting "transcribe English audio" sends `POST /v1/audio/transcriptions`
with `model=whisper-small`. aistack picks faster-whisper, schedules it
against whatever else is on the GPU, runs it, returns. Tomorrow that
same request might be served by a LAN peer on a 4090 — the client does
not need to change.

## Topology evolution

The gateway architecture is designed to absorb topology changes
behind the API.

| Stage | Backends aistack manages | Client-visible change |
|---|---|---|
| **2026-05-06 (today)** | This host's GPU (ASR + TTS), this host's Ollama (LLM proxy) | none — clients already have the stable API |
| **Multi-model on host** | Same as today, but with hot-swap + GPU-aware scheduling between backends | none |
| **LAN GPU peer** | + a sibling machine running aistack workers, joined as compute pool | none |
| **Rented cloud GPU** | + RunPod / Lambda Labs / similar GPU rentals provisioned and torn down on demand | none |

Each step expands aistack's internal capability without touching the
client API. This is the load-bearing reason for putting Ollama behind
aistack today rather than letting clients call Ollama directly: it
locks in the abstraction now, before the topology grows.

## Service shape

- Single FastAPI process per host. **Note**: a single host's process is
  the unit of vertical scale; horizontal scale comes from peer
  aistack instances on other hosts, joined by the scheduler in a
  later phase.
- Default bind: `127.0.0.1:11500`
- Protocol: **OpenAI API compatible**:
  - `POST /v1/audio/transcriptions` — ASR (in-process backends)
  - `POST /v1/audio/speech` — TTS (proxies to Qwen3-TTS Docker)
  - `POST /v1/chat/completions` — LLM (proxies to Ollama; planned Phase 2 work)
  - `GET  /v1/models` — aggregate list across all backends
  - `GET  /health` — readiness probe (worker liveness)
  - `GET  /admin` — Web UI (HTMX/Jinja, planned)

Streaming responses use SSE (OpenAI standard) where the underlying
backend supports it.

## Naming convention

- Repo / package / docs / CLI / files / logs: bare `aistack`
- UI title bar (only when shown to end users): `dosmoon-aistack`
- The `dosmoon-` prefix is a runtime brand marker, never a code identifier

## Phased roadmap

| Phase | Scope |
|---|---|
| **D1** | Bare FastAPI skeleton: `/health`, `/v1/models` returning empty list. No models. *(done)* |
| **D2** | Migrate Qwen3-TTS + vLLM-Omni from VideoCraft; wire `/v1/audio/speech`. *(done)* |
| **D3** | Migrate ASR providers (faster-whisper / Parakeet / SenseVoice); wire `/v1/audio/transcriptions`. *(done)* |
| **D4** | Web UI at `/admin` (HTMX): models list, install/remove, GPU mem, live RTF, log tail. *(planned)* |
| **D5** | VideoCraft client switch: replace in-process `*_local.py` with HTTP calls. Delete migrated source. *(done)* |
| **D6** | **LLM gateway**: `/v1/chat/completions` proxy to Ollama. GPU-aware scheduling: ASR/LLM evict each other so neither OOMs. VideoCraft Ollama provider's `base_url` switches from `localhost:11434/v1` to `localhost:11500/v1`. *(in progress)* |
| **D7+** | Backend abstraction: refactor in-process ASR / TTS-Docker / Ollama-proxy as plug-in backends behind a scheduler. Adds room for LAN peers and cloud GPU rentals later without touching the client API. |

## Code structure (D6+ target)

```
aistack/
├── api/                  HTTP route handlers (one per OpenAI surface)
│   ├── asr.py            POST /v1/audio/transcriptions
│   ├── tts.py            POST /v1/audio/speech
│   └── llm.py            POST /v1/chat/completions   ← new in D6
├── backends/             Each capability has a registry of backends
│   ├── llm/
│   │   └── ollama.py     The first LLM backend (D6)
│   ├── asr/              (Phase D7 refactor target — in-process today)
│   └── tts/              (Phase D7 refactor target — Docker proxy today)
├── scheduler.py          Where this request goes; which to evict first
├── _model_cache.py       In-process model cache + idle eviction
├── _gpu_lock.py          Single-task GPU mutex for in-process backends
└── ...
```

D6 lands the new `api/llm.py` and `backends/llm/ollama.py`. The deeper
restructuring of ASR/TTS into `backends/` (D7) is deferred until either
a second backend per category exists or the directory naming starts
hurting more than it helps.

## Cross-process concerns

- **GPU contention**: in-process ASR shares the local GPU with Ollama
  (also on this host) and with the Qwen3-TTS Docker sidecar. The
  scheduler is responsible for evicting hot models from one capability
  before invoking another. The Docker sidecar reserves VRAM at startup
  and is the hardest to coordinate; lower its `gpu_memory_utilization`
  (default 0.80) when running on tighter hardware.

- **Cancellation**: clients that abort an HTTP request should see the
  underlying inference cancelled where supported. ASR backends already
  accept a `cancel_token` (cooperative); the HTTP layer needs to wire
  `Request.is_disconnected()` into that token. Tracked as a follow-up.

- **Error envelopes**: all backends serialize structured `AIError`
  (kind / provider / message) so clients can branch on failure mode
  without parsing free-form text. Defined in `aistack/errors.py`.

## Out of scope (forever)

- Cloud-only LLMs as a proxied target. DeepSeek / Claude / Gemini API
  calls have no local resource cost; routing them through aistack
  adds latency and a single point of failure for zero benefit. Clients
  call those directly.

- Client-side concerns: prompt templates, translation orchestration,
  user UX, persistence. Those belong to the client (e.g. VideoCraft).
  aistack is a transport, not an application.
