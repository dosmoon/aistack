# ML Runtimes per Provider

> Why each ASR/TTS provider in aistack ships its own runtime, what those
> runtimes cost, and where the model weights live on disk.

## Runtime per provider

Every open-source ASR/TTS model is bound to a specific inference runtime by
its release ecosystem. There is no single runtime that covers all of them,
so aistack treats each provider as a self-contained module with its own
heavy import path.

| Provider | Model | Runtime | Weight location | Approx. install footprint |
|---|---|---|---|---|
| **faster-whisper** | Systran/faster-whisper-{tiny..large-v3} | CTranslate2 (C++ inference engine, Python bindings) | `HF_HOME/hub/` | ~150 MB Python deps; lightest |
| **Parakeet** | nvidia/parakeet-tdt-0.6b-v3 | NVIDIA NeMo full stack | `HF_HOME/hub/` (loaded via HuggingFace `from_pretrained`) | ~3-5 GB Python deps; heaviest |
| **SenseVoice** | iic/SenseVoiceSmall | Alibaba FunASR + ModelScope | `MODELSCOPE_CACHE/models/iic/` | ~1 GB Python deps |
| **Qwen3-TTS** | Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice | vLLM-Omni inside a Docker container | `HF_HOME/hub/` (mounted into the container) | ~9 GB container image |

## Cache directory mapping

Set as environment variables before any ML library is imported. `scripts/dev.bat`
exports these for the dev server; production deployments should do the
equivalent.

```text
HF_HOME              D:\AI_Models\hf            # HuggingFace Hub cache.
                                                # Used by: faster-whisper,
                                                # Parakeet (via NeMo
                                                # `from_pretrained`),
                                                # Qwen3-TTS container mount.
MODELSCOPE_CACHE     D:\AI_Models\modelscope    # Alibaba ModelScope cache.
                                                # Used by: SenseVoice (FunASR
                                                # downloads from ModelScope,
                                                # not HuggingFace).
NEMO_CACHE_DIR       D:\AI_Models\nemo          # NeMo's own cache for
                                                # checkpoints not on HF;
                                                # currently unused for
                                                # Parakeet but exported for
                                                # forward compatibility.
```

## NVIDIA NeMo — what it is and why it is heavy

NeMo (Neural Modules) is NVIDIA's open-source full-stack AI toolkit — Apache 2.0,
GitHub `NVIDIA/NeMo`. It covers training, fine-tuning, and inference across:

- ASR (Parakeet, Canary, Conformer, Citrinet)
- TTS (FastPitch, HiFi-GAN, Mixer-TTS)
- LLM (Megatron, Nemotron)
- Multimodal experiments

aistack uses the `nemo_toolkit[asr]` extra **only for inference** of one model
(Parakeet TDT 0.6B v3). The pip install drag, however, brings the whole training
stack: Apex, Megatron-core, PyTorch Lightning, wandb, dataset loaders, metrics
libraries. There is no "inference-only" install path.

### Why we tolerate the size

Parakeet inference at RTF 0.065 on CPU (≈15× faster than faster-whisper-small
on the same hardware) is the only open-source path that hits this performance
band for European-language ASR. The runtime cost is paid once at install time;
the speed dividend pays back on every transcription.

### Windows install pitfalls (locked in pyproject extras)

`pip install nemo_toolkit[asr]` resolves to recent versions of `pyarrow`, `pandas`,
and `datasets` that crash on Windows with a C-extension access violation
(observed: `pyarrow 24.x` faulthandler stack trace at first import). The
`asr-parakeet` extra in `pyproject.toml` pins these to known-good versions:

```toml
asr-parakeet = [
    "nemo_toolkit[asr]>=2.0",
    "pyarrow==19.0.1",
    "pandas==2.3.3",
    "datasets==3.6.0",
]
```

When bumping NeMo, **re-validate these three pins**. The dependency resolver
may relax them to incompatible versions on a major NeMo upgrade.

## FunASR — the SenseVoice runtime

FunASR is Alibaba's open-source speech toolkit, MIT licensed. SenseVoice is one
of its hosted models. FunASR pulls its weights from ModelScope (Alibaba's
HuggingFace equivalent), not from HuggingFace Hub — hence the separate
`MODELSCOPE_CACHE` env var.

Two install pitfalls discovered during the VideoCraft Phase L2.2 work and now
codified in the `asr-sensevoice` extra:

1. `pip install funasr` does **not** pull `torchaudio`, but
   `funasr/utils/load_utils.py` line 1 imports it. ImportError on first use.
2. `funasr 1.3.x` only works with `torchaudio 2.11.x`. Newer pairings break the
   FunASR audio loader silently (returns garbage tensors).

```toml
asr-sensevoice = [
    "funasr==1.3.1",
    "torchaudio==2.11.0",
    "soundfile",
]
```

## CTranslate2 — the faster-whisper runtime

The lightest of the four. CTranslate2 is a small C++ inference engine with
Python bindings. The `faster-whisper` package is just a thin wrapper that
loads CT2-format Whisper weights and exposes the OpenAI Whisper API.

CUDA path requires CUDA 12 Toolkit + cuDNN installed on the host; otherwise
ctranslate2 reports `cublas64_12.dll not found` at first inference. Default
`device="auto"` falls back to CPU when those DLLs are missing — safer.

## vLLM-Omni — the Qwen3-TTS runtime

Different shape from the others: ships as a Docker image
(`vllm/vllm-omni:v0.18.0`), not a Python package. The container exposes an
OpenAI-compatible HTTP endpoint and bundles the paper's full optimization
stack: vLLM engine, torch.compile, CUDA Graph capture, triton kernels for
the codec decoder.

aistack proxies to it transparently; from a client's perspective, calling
`POST /v1/audio/speech` on aistack at port 11500 forwards to the container
at port 17860 (host-only) and streams the response back.

## Could we replace any of these?

In principle yes, in practice no.

| Provider | Replacement candidate | Why we don't |
|---|---|---|
| Parakeet (NeMo) | CTranslate2-converted Parakeet, ONNX, vLLM | Community ports exist but lag NeMo features (e.g. word timestamps, CUDA Graph capture). Maintenance burden falls on us. |
| SenseVoice (FunASR) | Direct PyTorch load of the model weights | Skips ModelScope auth quirks but loses VAD integration, language tags, ITN — all built into FunASR's pipeline. |
| Qwen3-TTS (vLLM-Omni) | The official `qwen-tts` pip package | Measured RTF 2.2 vs vLLM-Omni's 0.78 on the same hardware. Reference implementation, no production optimizations. |
| faster-whisper (CT2) | OpenAI's `openai-whisper` package | Slower (~3-5×) and heavier on GPU memory. CT2 is strictly better. |

The lock-in is the price of using state-of-the-art open-source models. We
mitigate it by treating each runtime as an isolated provider module with a
lazy import — installing `aistack[asr-fasterwhisper]` does not pull NeMo, and
running TTS-only does not pull FunASR.
