---
title: The real boundary of Whisper translation
date: 2026-05-08
tags: [asr, whisper, translation, capability-boundary]
sidebar:
  order: 3
---

# The real boundary of Whisper translation

> TL;DR
>
> Whisper's `task=translate` is **X → English only** — a hard limit set by training data, not a configuration switch.
> It **cannot** translate English audio into Chinese, nor produce any non-English target.
> If you want EN→ZH subtitles you must cascade: Whisper transcribe + LLM translate; or switch models (SeamlessM4T / Qwen2-Audio).

---

## 1. Whisper's translation direction is one-way

Whisper's transcribe API takes a `task` field. The OpenAI docs list two values:

| task | Behavior | Output language |
|---|---|---|
| `transcribe` | Speech-to-text | Source language |
| `translate` | Speech translation | **Always English** |

This limit is stated in the [Whisper paper](https://arxiv.org/abs/2212.04356) §2.1 ("Data Processing"):
> "We then construct a dataset by combining audio with transcripts in the same language, as well as audio with English translations of the speech in the audio."

The training data has only two pair types: ① same-language audio + text, ② any-language audio + English translation. **No X→Y (non-English target) samples exist.** The model has therefore never been taught to translate speech into anything other than English.

Empirical behavior:

- Feed Whisper English audio + `task=translate` → output is still English (equivalent to transcribe; no translation happens)
- Feed Whisper Chinese audio + `task=translate` + initial_prompt="output in Spanish" → still outputs English; the prompt is ignored
- Community issues confirm this repeatedly: [openai/whisper#649](https://github.com/openai/whisper/issues/649), [#2046](https://github.com/openai/whisper/issues/2046)

**This is a hard model-capability boundary**, not an API restriction, and cannot be worked around with prompt engineering.

## 2. Quality of the X → English direction (the one Whisper actually does)

Community and official-eval consensus:

| Source language | Quality tier | Notes |
|---|---|---|
| FR / DE / ES / PT / IT → EN | Subtitle-grade usable | BLEU around 25–32 on large-v3 (CoVoST-2); below modern frontier LLMs doing translation directly |
| ZH / JA / KO → EN | Medium | Accurate but heavy on paraphrasing; long sentences get anglicized |
| Arabic / Hindi / Vietnamese → EN | Weaker | Hallucination risk rises |
| Low-resource (Swahili, Bengali, ...) → EN | Unreliable | Whole-segment fabrication is common |

Known issues (some shared with transcribe, some translation-specific):

- **Paraphrasing wins over literal translation** — the "stick to the source" property professional subtitling needs is not Whisper's strength
- **Long-segment truncation** — content gets dropped at the 30-second window boundary
- **Numbers and proper nouns get lost** — "$3.5 million in 2023" can become "millions of dollars"
- **VAD-failure passages hallucinate** — silent segments produce whole fabricated translated sentences

## 3. Three viable paths for EN → ZH subtitles

| Option | Shape | Controllability | Offline-friendly | Quality ceiling | Deployment cost |
|---|---|---|---|---|---|
| **A. Whisper(EN→EN) + LLM(EN→ZH)** | Cascade | High (two stages tuned independently) | ✅ Fully local | High (depends on LLM) | Low (aistack already has the pieces) |
| **B. SeamlessM4T-v2** (Meta) | One-shot speech→text translation | Medium | ✅ | Medium-high (direct EN→ZH) | Medium (model 2–9 GB, PyTorch deploy) |
| **C. Qwen2-Audio Instruct** | One-shot, prompt-driven | High (natural-language instruction) | ✅ | High | Medium-high (vLLM deploy, ~14 GB) |

**Option A is the most natural path on aistack:**

- Whisper runs `transcribe` (not `translate`), produces English source text
- The English text is fed to a mid-size-or-larger LLM on Ollama (qwen2.5:14b, qwen2.5:32b, deepseek-v2, ...) for translation
- Both pieces — ASR and the LLM proxy — already exist in aistack today; only the convenience wrapper is missing

**Why Option B is interesting:** one fewer LLM call; potentially lower end-to-end latency. But SeamlessM4T's model is large, vocabulary control is less flexible than an LLM, and proper-noun handling / prompt injection is awkward. **We have not yet measured its long-form transcription stability ourselves.**

**Why Option C is interesting:** highest quality ceiling (a multimodal LLM that genuinely "listens" and then translates), but heaviest to deploy. It is also the best fit for aistack's research-shaped form — if dosmoon ever wants to study the ceiling of "audio understanding + translation as one piece", C is the path to take.

## 4. Implications for aistack and future product forms

- **Short-term aistack value:** add a "transcribe-and-translate" convenience endpoint (Option A cascade). No new dependencies, immediately usable by downstream consumers (VideoCraft, etc.).
- **Aistack research value:** measured comparison of A vs B vs C on the same English podcast: BLEU / human preference / end-to-end latency / VRAM peak. Exactly the kind of experiment aistack exists to host.
- **Offline-product form judgment:** if a future product positioning is "offline English-to-Chinese subtitle tool", **Option A (whisper.cpp + a local small LLM) is the only clean path**. The size and complexity of SeamlessM4T and Qwen2-Audio exceed what an "install and run" form factor can carry.

---

## Open questions (to measure)

1. **Real-world quality of Option A:** run Whisper-large-v3 + qwen2.5:14b on a 17-minute English speech, then blind-compare against a frontier LLM (Claude / GPT-4 class) doing speech-to-Chinese-text directly. How big is the gap? The distance between fully-offline local and the closed-source ceiling is the key data point for any product decision here.

   **No comparison against DeepL / NLLB / traditional NMT.** dosmoon's content of interest (news, politics, economics, opinion podcasts) is knowledge-heavy. DeepL's strength is narrow-domain professional vocabulary (legal / medical / technical manuals); it **has no world-knowledge model**. Faced with content that needs grounding in entities, events, and geopolitical context, NMT systems produce output that is "word-for-word correct but reads like a machine". LLMs have already won this lane; further benchmarking has no research value. Only revisit NMT if a narrow-domain professional translation need shows up later (e.g., legal documents).

2. **Long-audio stability of Option B:** SeamlessM4T's official demos are all < 30-second clips. Does long audio need chunking? Same chunking strategy as Whisper, or a different one?

3. **Prompt control in Option C:** can a prompt make Qwen2-Audio produce a "literal" or "localized" rendering on demand? This is a possible product-differentiation knob.

4. **Pseudo EN→ZH Whisper variants:** are there community-fine-tuned Whisper checkpoints that directly do EN→ZH? (Some attempts exist, e.g., whisper-large-v2-cn-translate, but data scale and quality are unclear.)
