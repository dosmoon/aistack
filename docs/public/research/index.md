---
title: Research Notes
description: Stable conclusions about the technology and decisions around aistack — empirical measurements, landscape surveys, capability boundaries, positioning calls.
sidebar:
  order: 5
---

# Research Notes

Stable conclusions about the technology and decisions around aistack. Forms include but are not limited to:

- **Empirical notes** — numbers, configurations, reproducible phenomena from our own hardware
- **Landscape surveys** — industry overviews + selection recommendations
- **Capability boundaries** — clarifying what a tool can / cannot actually do
- **Positioning calls** — what the project does and explicitly does not do, and why

Common thread: **they keep being referenced later** — they remain valid across project evolution cycles.

## Notes

| Note | Subject |
|---|---|
| [Consumer-GPU local ASR performance baseline](consumer-gpu-asr-baseline/) | 50-min English audio / RTX 4060 8GB / 62s end-to-end / RTF 0.021. The data you need to decide whether self-hosted ASR fits your situation. |
| [NVIDIA Parakeet TDT on consumer GPU for long audio](parakeet-on-consumer-gpu/) | The working configuration for running Parakeet long audio on an 8 GB card: which knobs to turn on, which not to touch, why NVIDIA's docs do not say it all in one place. |
| [The real boundary of Whisper translation](whisper-translation-capability/) | Whisper `task=translate` is X→English only, cannot do EN→ZH or any non-English target translation. Comparison of three viable paths when you need EN→ZH subtitles. |
| [Chinese ASR engine selection baseline](chinese-asr-engine-survey/) | Landscape comparison of FireRedASR-AED / Paraformer / SenseVoice / Whisper-large-v3 / Fun-ASR / FireRedASR2S. AISHELL/WenetSpeech CER + design intent + evaluation pitfalls + 8GB-card integration cost. **Desk research; read before benching.** |
| [The Whisper ecosystem](whisper-ecosystem-tools/) | 6 categories, 25+ projects organized: inference engines / distilled models / enhancement layers / streaming / Whisper-style retraining / cross-domain inversions. What aistack should and should not integrate, and what a future product form might use. |

## Style conventions

- Each note opens with a TL;DR of three lines or fewer answering "what do I take away from reading this".
- Quantities carry units and the machine ("8 GB VRAM @ RTX 4060 Laptop", not just "twice the memory").
- Every reference to upstream docs and community posts must be a link.
- Each note ends with an **Open questions** section listing what is not yet confirmed — research notes are not textbooks; admitting unknowns is more credible than pretending omniscience.
- These notes are authored in English. The Chinese versions under `/zh-cn/research/` are translations.
