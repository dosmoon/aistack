# aistack Documentation

## api/  (public API reference — what consumers depend on)

- **[README.md](api/README.md)** — API overview: versioning policy, base URL, authentication, OpenAI-compat philosophy, `/health`.
- **[errors.md](api/errors.md)** — Error envelope schema. The set of `kind` values is part of the contract.
- **[models.md](api/models.md)** — `GET /v1/models` — capability inventory. Use for picker UIs.
- **[asr.md](api/asr.md)** — `POST /v1/audio/transcriptions` — Whisper-compatible ASR.
- **[tts.md](api/tts.md)** — `POST /v1/audio/speech` — TTS via Qwen3-TTS proxy + extension fields.
- **[llm.md](api/llm.md)** — `POST /v1/chat/completions` — *(D6 deliverable, not yet wired)*.

## design/  (how aistack works internally)

- **[architecture.md](design/architecture.md)** — Service shape as an AI capability gateway, scope, naming convention, topology evolution, phased roadmap (D1~D7+). The authoritative description of *what aistack is*.
- **[decoupling.md](design/decoupling.md)** — Why aistack was split out from VideoCraft, the boundary discussion, and the 6 cross-cutting decisions made on 2026-05-06, plus §11 D6-era revision (gateway upgrade). *Migrated from VideoCraft.*

## selection/

Model selection rationales — research notes that informed *which* models aistack ships.

- **[asr.md](selection/asr.md)** — Why faster-whisper + Parakeet + SenseVoice form the ASR triplet. *Migrated from VideoCraft.*
- **[tts.md](selection/tts.md)** — Why Qwen3-TTS-12Hz-0.6B-CustomVoice via vLLM-Omni won the TTS slot. *Migrated from VideoCraft.*
- **[llm-rationale.md](selection/llm-rationale.md)** — Historical record of the local LLM evaluation that led to the conclusion *"don't build LLM into aistack — Ollama already nails it"*. *Migrated from VideoCraft.*
- **[runtimes.md](selection/runtimes.md)** — Per-provider ML runtime breakdown (CTranslate2 / NeMo / FunASR / vLLM-Omni), cache-directory mapping, install pitfalls, and why we tolerate NeMo's heavy dependency stack.

## progress/

- **[local_ai.md](progress/local_ai.md)** — L1~L3 chronicle of every install, debug, and tuning session that happened in VideoCraft before the decoupling. Captures the why-this-config-not-that-one knowledge that would otherwise rot. *Migrated from VideoCraft.*

## research-note/

实证研究笔记。沉淀**经过本地实测且上游官方文档没有完整说明**的发现。这一类资料本身有公开发布价值（计划进 dosmoon Pages），所以独立成 first-class 目录而不是塞在 selection/ 或 design/ 之下。准入门槛、风格约定、与其他文档类别的边界划分见 [research-note/README.md](research-note/README.md)。

- **[parakeet-on-consumer-gpu.md](research-note/parakeet-on-consumer-gpu.md)** — NVIDIA Parakeet TDT 在 8 GB 消费卡上跑长音频的可工作配置 + 三个 NVIDIA 文档没说的连带效应（含实测数据：50min mp3 / VRAM / shared RAM / 推理时长对照）。

---

**On migrated documents.** Files marked *Migrated from VideoCraft* are
copies taken on 2026-05-06. The VideoCraft-side originals are temporarily
retained as snapshots but are no longer the source of truth — future
edits land here. They will be removed from VideoCraft in phase D5 once
the client-side migration is complete.
