# Qwen3-TTS (vLLM-Omni) — internal container

Backend container for aistack's TTS endpoint. Runs `Qwen3-TTS-12Hz-0.6B-CustomVoice`
on top of the official `vllm/vllm-omni` image.

> **Do not query this container directly.** It binds to `127.0.0.1:17860`
> (host-only) and sits behind the aistack FastAPI front door at port `11500`.
> End users should call `http://127.0.0.1:11500/v1/audio/speech`.

## Why vLLM-Omni (not the qwen-tts pip package)

The reference `qwen-tts` Python package is a thin wrapper that runs vanilla
HuggingFace inference with no optimizations — measured RTF ~2.2 on RTX 4060
Laptop, vs. the paper's ~0.29 on enterprise hardware.

The paper's optimization stack (vLLM engine + torch.compile + CUDA Graph +
triton kernels for the codec decoder) is shipped as **vLLM-Omni**, the
official Qwen team-supported deployment path. Switching to it brings
RTF down to **0.72 - 0.86** on the same hardware (~3x speedup, faster than
real-time).

## Prerequisites

- Docker Desktop with WSL2 backend
- NVIDIA GPU + recent host driver (CUDA 12.x runtime in container)
- Model already cached at
  `D:\AI_Models\hf\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice`
  (standard HuggingFace cache layout, ~2.4 GB)

## Run

```powershell
cd D:\My_Prjs\dosmoon-aistack\docker\tts_qwen3
docker compose up -d                    # first run: pulls vllm/vllm-omni:v0.18.0 (~9 GB)
docker compose logs -f qwen3-tts        # watch engine init (model load, compile, CUDA graphs)
```

First HTTP request triggers torch.compile + CUDA Graph capture (~60 s).
All subsequent requests run in steady state.

## Verify (internal)

These calls hit the container directly. Used only for plumbing checks —
production traffic should always go through aistack at 11500.

```powershell
curl http://127.0.0.1:17860/health

curl -X POST http://127.0.0.1:17860/v1/audio/speech `
  -H "Content-Type: application/json" `
  -d '{"input":"hello world","voice":"vivian","task_type":"CustomVoice","language":"English"}' `
  --output out.wav
```

## End-to-end via aistack

```powershell
curl -X POST http://127.0.0.1:11500/v1/audio/speech `
  -H "Content-Type: application/json" `
  -d '{"input":"hello world","voice":"vivian","task_type":"CustomVoice","language":"English"}' `
  --output out.wav
```

aistack transparently proxies to the container. Same request body, just
a different port.

## Endpoints (provided by vLLM-Omni, all proxied by aistack)

| Path | Method | Purpose |
|---|---|---|
| `/v1/audio/speech` | POST | Standard / cloned / instructed synthesis |
| `/v1/audio/speech/stream` | POST | Streaming synthesis |
| `/v1/audio/speech/batch` | POST | Batched synthesis |
| `/v1/audio/voices` | GET / POST | List / register voices |
| `/v1/audio/voices/{name}` | DELETE | Remove a registered voice |

Request body fields beyond OpenAI standard: `task_type` (`CustomVoice` /
`VoiceClone` / `VoiceDesign`), `language`, `instructions`, `ref_audio`
(URL / base64 / `file://`), `ref_text`, `max_new_tokens`.

## Performance (RTX 4060 Laptop, 8 GB VRAM)

| Stack | Steady RTF |
|---|---|
| qwen-tts pip + sdpa | 2.08 - 2.22 |
| qwen-tts pip + flash_attention_2 | 2.42 - 2.45 |
| **vLLM-Omni v0.18.0** | **0.72 - 0.86** |

For reference, the paper reports RTF 0.288 at concurrency=1 on undisclosed
"typical computational resource" hardware (likely H100 / A100). RTF ~0.85
is the documented target for the 4060 Ti class.

Comprehensive Tier 1/2/3 testing (concurrency, streaming TTFP, multi-language,
long input, VRAM ceiling, voice clone, voice design, sustained load,
1.7B upgrade, Piper fallback) is pending and will be tracked under aistack's
own roadmap.

## Troubleshooting

- **OOM on startup** — lower `--gpu-memory-utilization` (currently 0.80);
  the 4060 Laptop shares VRAM with the Windows desktop (~440 MB baseline)
- **First request takes 60 s** — torch.compile + CUDA Graph capture, one-time
- **Image not found** — pin a specific tag (`v0.18.0` known good); `:latest`
  does not exist on Docker Hub
- **`restart: unless-stopped` hides crash logs** — temporarily set
  `restart: "no"` to capture root cause during init failures
