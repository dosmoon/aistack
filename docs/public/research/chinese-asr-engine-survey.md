---
title: Chinese ASR engine selection baseline
date: 2026-05-08
tags: [asr, chinese, mandarin, benchmark, sensevoice, paraformer, fireredasr]
sidebar:
  order: 4
---

# Chinese ASR engine selection baseline (2026 perspective)

> **TL;DR**
>
> 1. aistack currently routes `lang=zh*` to SenseVoiceSmall by default, **but this choice has never been measurement-validated** — we have not run a Chinese bench, and `bench/audio/` contains only English audio
> 2. Public benchmark data (AISHELL-1/2 + WenetSpeech net/meeting) 2024–2025 ranking: FireRedASR2S-LLM (2.89%) > FireRedASR-AED (3.18%) ≈ FireRedASR-LLM (3.05%) ≫ SenseVoice-L (4.47%) ≈ Paraformer-large (4.56%). **SenseVoice has the worst Chinese CER among mainstream engines**
> 3. But SenseVoice is not a wrong choice — its design point is "lightweight + multilingual + emotion/event tagging", not "Chinese quality ceiling". "Best Chinese quality" is a different question
> 4. **On 8 GB VRAM consumer cards, FireRedASR-AED (1.1B) is currently the highest-value Chinese ASR candidate** — Apache-2.0 license + acceptable integration cost
> 5. **Foundational work that must come first:** stand up a Chinese bench (real audio + ground truth + CER evaluation script). Without this any "swap engine" decision is empty talk

---

## 1. Problem statement: why redo this

aistack's `auto` router (`aistack/api/asr.py:_select_for_auto`) already routes `lang=zh*` to SenseVoice, but this choice rests only on "it supports Chinese + the FunASR runtime is already installed" — **with no quality data backing it**. Specifically:

- `bench/audio/` is all English audio (`perf-12min.mp3` ~ `perf-97min.mp3`); the `*_zh.srt` files are Chinese translations of English audio, not ground truth for Chinese ASR
- `bench/asr_eval.py` only runs LibriSpeech; it is an English WER bench with no Chinese path
- The backlog item "Mandarin ASR dataset (Common Voice zh-CN or AISHELL dev)" has been ready for a long time but never actioned

A research-shaped aistack **should be quantifying this**. This note is desk research before the action — extract the existing prior knowledge clean before measurement, to avoid detours.

## 2. Currently viable Chinese ASR engines (2024–2026)

Sorted by "fit with 8 GB VRAM consumer hardware + aistack's research-shaped form":

| Engine | Source | Params | Public avg CER¹ | 8GB | License | Design intent |
|---|---|---|---|---|---|---|
| **SenseVoiceSmall** | Alibaba / FunASR | ~234 M | not on mainstream benchmark² | ✅ trivial | MIT (FunASR) | Lightweight multilingual + emotion/event tagging |
| **SenseVoice-Large** | Alibaba / FunASR | ~1.6 B | 4.47% | ✅ easy | MIT | Multilingual general-purpose large variant |
| **Paraformer-large** | Alibaba / FunASR | ~220 M | 4.56% | ✅ trivial | MIT | Alibaba's Chinese-specialist SOTA elder |
| **Paraformer-large-vad-punc** | Alibaba / FunASR | ~220 M + VAD/punc | same + bundled punctuation | ✅ trivial | MIT | Paraformer + punctuation restoration + VAD, plug-and-play |
| **FireRedASR-AED** | XiaoHongShu / 2025-01 | **1.1 B** | **3.18%** | ✅ tight but workable³ | Apache-2.0 | **Current value champion** — 30% lower CER than Paraformer-large at 5× the parameters |
| **FireRedASR-LLM** | XiaoHongShu / 2025-01 | 8.3 B | 3.05% | ❌ | Apache-2.0 | Quality ceiling, but does not fit on 8 GB |
| **FireRedASR2S-AED** | XiaoHongShu / 2025 | undisclosed | undisclosed | ✅ presumed | Apache-2.0 | v2 update with VAD/LID/Punc |
| **FireRedASR2S-LLM** | XiaoHongShu / 2025 | undisclosed | **2.89%** | ❌ | Apache-2.0 | Current public SOTA, but too large |
| **Fun-ASR-Nano** | Alibaba / Tongyi 2025-12 | undisclosed⁴ | undisclosed | presumed ✅ | Mostly commercial SaaS⁵ | Multi-dialect (7 dialects + 26 accents), low hallucination (78.5%→10.7%) |
| **Whisper-large-v3** | OpenAI | ~1.5 B | clearly behind specialists on Chinese⁶ | ✅ | MIT | Multilingual general baseline |
| **Belle-whisper-large-v3-zh** | BELLE / 2024 | ~1.5 B | 24–65% improvement over Whisper-v3 | ✅ | Whisper-derived | Chinese-fine-tuned, community-built |
| **Qwen2-Audio** | Alibaba | 7 B+ | — | ⚠ tight | Commercial license | Multimodal LLM, strong on "understanding" not pure transcription efficiency |

¹ avg CER = average across AISHELL-1 + AISHELL-2-ios + WenetSpeech-net + WenetSpeech-meeting. Source: [FireRedASR paper](https://arxiv.org/abs/2501.14350)
² SenseVoiceSmall does not appear in the FunASR paper's mainstream four-set comparison; only the SenseVoice paper itself reports Chinese CER under different settings
³ FireRedASR-AED 1.1B in fp16 is ~2.2 GB VRAM; with activations and KV ~4–5 GB total — fits on 8 GB but tighter than Paraformer
⁴ Fun-ASR is Alibaba's new-generation speech recognition large model launched 2025-09; the Nano variant launched 2025-12 emphasizes real-time and multi-dialect
⁵ Fun-ASR primarily ships through Tongyi DashScope's commercial API; the GitHub repo is open but training data scale makes it product-direction not research-direction
⁶ Whisper-large-v3 has no official AISHELL/WenetSpeech CER for Chinese, but the community consensus is "clearly behind specialist models" — Belle's "24–65% relative improvement" is reverse evidence

## 3. Each engine's design intent (avoid misreading the pitch)

### SenseVoice family (small + large)
- **Real selling point:** a 234M model handling ZH/EN/JP/KO + emotion recognition + acoustic event detection — **one model for "speech understanding", not just transcription**
- **Why it is not the ceiling:** Alibaba's own "pure Chinese SOTA" track is Paraformer and Fun-ASR; SenseVoice is on a different product axis
- **Good for:** scenarios needing language switching + emotion tagging + real-time captions (live streams, meetings)
- **Bad for:** high-quality Chinese transcription — its Chinese CER is at the bottom of the mainstream comparison

### Paraformer-large
- **Design point:** non-autoregressive (NAR) end-to-end, **no Whisper-style 30s padding limit**, single-step parallel decoding hence fast
- **Quality position:** long considered a strong baseline for open-source Chinese ASR SOTA; was the first choice before FireRedASR appeared
- **Variants:** `speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` (raw) + `speech_paraformer-large-vad-punc_asr_*` (with VAD and punctuation). **The VAD-punc variant is more production-ready out of the box**
- **Integration cost:** FunASR is already installed in aistack (same runtime SenseVoice uses); adding Paraformer is the smallest possible increment

### FireRedASR (v1 + v2S)
- **Design point:** XiaoHongShu's industrial-grade in-house model, AED and LLM dual track. AED is "the practical version", LLM is "the quality ceiling"
- **Public SOTA:** in the paper, FireRedASR-LLM cuts Paraformer-large's 5.80% to 3.48% (38.6% relative improvement). FireRedASR2S-LLM goes further to 2.89%
- **AED's practical value:** 1.1B parameters, public CER 3.18%, **nearly 30% lower than Paraformer-large at only 5× the parameter cost**. This is the engine "that should be integrated immediately"
- **Risks:** ① not the FunASR runtime, requires a separate dependency install (its own inference package) ② **40–60s input hard limit**, long audio requires application-layer segmentation ③ smaller community than FunASR, "issue response time" uncertain
- **Code license:** Apache-2.0, clean

### Fun-ASR (Tongyi 2025)
- **Design point:** Alibaba Tongyi Lab's 2025 new-generation ASR large model, emphasizing multi-dialect, low hallucination, real-time
- **Key number:** in noisy scenarios cuts hallucination rate from 78.5% to 10.7% (one of the most noteworthy improvement directions since the Whisper era)
- **Risks:** primarily ships via Alibaba DashScope commercial API; the open version (GitHub `FunAudioLLM/Fun-ASR`) is usable, but training data scale, weights completeness, and long-term maintenance commitment are all less certain than the traditional Paraformer line
- **When to consider:** wait until the Nano variant (end-to-end real-time ASR) has a clearer open-source story, then evaluate

### Whisper-large-v3 (baseline reference)
- **Significance:** as a multilingual general model, its performance on Chinese is the "reference frame", not a product option
- **Fine-tune path:** BELLE-2's `Belle-whisper-large-v3-zh` improves Chinese CER 24–65% — much better than vanilla Whisper but still behind FireRedASR
- **When to use:** ① when needing simultaneous ZH/EN/JP/KO support and model storage is tight ② when running cross-language general baseline tests

### Out of scope
- **FireRedASR-LLM (8.3B)** / **FireRedASR2S-LLM** / **Qwen2-Audio (7B+)** — does not fit on 8 GB or is tight to the point of impractical; research-shaped form notes existence but does not measure
- **Conformer/Branchformer academic variants** — no public production-grade weights; research-shaped form can cite papers, no measurement effort

## 4. Public benchmark data (prior conclusions)

CER on the four standard test sets (lower is better), sourced from [FireRedASR paper Table 1](https://arxiv.org/abs/2501.14350), [FunASR paper](https://arxiv.org/abs/2305.11013), Alibaba ModelScope model cards:

| Model | AISHELL-1 | AISHELL-2-ios | WenetSpeech-net | WenetSpeech-meeting | avg |
|---|---|---|---|---|---|
| FireRedASR-LLM (8.3B) | 0.76 | 2.15 | 4.60 | 4.67 | **3.05** |
| FireRedASR-AED (1.1B) | **0.55** | 2.52 | 4.88 | 4.76 | 3.18 |
| FireRedASR2S-LLM (?) | — | — | — | — | **2.89** (Mandarin avg) |
| SenseVoice-L (~1.6B) | — | — | — | — | 4.47 |
| Paraformer-large (~220M) | 1.95 | 2.85 | — | 6.97 | 4.56 (Fire paper same setup)¹ |
| Whisper-large-v3 (~1.5B) | — | — | — | — | far above specialist models² |

¹ Differences between FireRedASR paper's Paraformer-large 5.80% and FunASR's self-reported 1.95% reflect: different papers use different Paraformer variants (vad-punc or not), different normalization strategies, different WenetSpeech subset groupings — **single numbers must be compared with setup awareness**
² Whisper's Chinese CER is typically reported as "clearly behind" without a standard number. Belle-whisper-large-v3-zh's 24–65% relative improvement implies vanilla Whisper Chinese CER is in the 10%+ range (specifics await our own measurement)

**Key observations:**
- AISHELL-1 (read aloud, quiet, professional broadcast voice) has been pushed to CER < 1% — **extremely poor representativeness for dosmoon's actual use cases (news / podcasts / commentary)**, this is not real-world difficulty
- WenetSpeech is the better real-world proxy: net is web audio (Bilibili, podcasts, video bloggers), meeting is meeting recordings — **CER generally 4–7%** is the true difficulty
- Real model gaps show up on WenetSpeech; on AISHELL-1 almost all models are < 1% (low discrimination)

## 5. Pitfalls in CER evaluation methodology

Chinese ASR evaluation must use **CER (Character Error Rate)**, not WER. Reasons and considerations:

### Why not WER
- **Chinese has no natural word boundaries** ("我去北京" is sliced as 4 / 2 / 3 words depending on the segmenter)
- Different segmentation tools (jieba / pkuseg / hanlp) produce different word counts and boundaries — unstable as a metric unit
- Character-level measure is "unambiguous" minimum granularity, no segmenter dependency

### Required normalization
Both ASR output and ground truth must be normalized to a consistent representation before measurement, or numbers will be incomparable:

| Dimension | Issue | Treatment |
|---|---|---|
| Numbers | "5000" vs "五千" | Unify to Chinese numerals (or all to Arabic) |
| Punctuation | Some models emit punctuation, others do not; "。" vs "." | When measuring text only, strip all punctuation; for punctuated versions, unify punctuation form |
| Full-width / half-width | "ＡＢＣ" vs "ABC" | Convert all to half-width |
| English case | "OK" vs "ok" | Unify to lower or upper |
| Spaces | Space placement in mixed CJK + Latin text | Strip all spaces before computing CER (or use consistent tokenization) |
| Traditional / simplified | "雲端" vs "云端" | Unify to simplified (or keep original if matching ground truth) |
| Homophone errors are real errors | "在" vs "再" is a real error and should not be normalized | Do not treat |

### Tools
- **`jiwer`** (Python lib) — most common, supports custom transformations. Compute CER directly with `jiwer.cer(reference, hypothesis)`
- **ESPnet `sclite` wrapper** — common in academia, most consistent with paper-reported numbers
- **`zhconv`** — traditional/simplified conversion
- Recommended: use jiwer + zhconv + a custom number-normalization function (reference FunASR's `funasr/utils/postprocess_utils.py` for an off-the-shelf implementation)

### Known traps
- CER numbers in papers **do not necessarily share normalization**; cross-paper comparisons must verify setup
- Almost all models on AISHELL-1 are < 1%, so **using it alone for ranking is essentially useless**; must look at WenetSpeech
- If an engine's output has punctuation but the ground truth does not, **computing CER directly inflates from punctuation mismatch**

## 6. Dataset selection strategy

### Standard public test sets
| Set | Duration | Content | Difficulty | dosmoon scenario representativeness |
|---|---|---|---|---|
| AISHELL-1 test | 5h | Professional broadcaster reading | Very low | Very poor |
| AISHELL-2 test-ios | 5h | iOS recording, home environment | Medium | Medium |
| WenetSpeech test-net | 24h | Bilibili / podcast / video blogger | Medium-high | **High** ✓ |
| WenetSpeech test-meeting | 15h | Meeting recordings | High | Medium (dosmoon doesn't focus on meetings) |
| SpeechIO test sets | Various | Real multi-scene (YouTube/TV/podcast) | Medium-high | **High** ✓ |
| Common Voice zh-CN | Various | Crowdsourced reading | Low | Low |

### Recommended dosmoon Chinese bench composition

Given dosmoon's actual content shape (news, politics, economics, podcast commentary), three layers are recommended:

**L1: Standard reference baseline** (a few minutes, quick to run)
- 5–10 clips from AISHELL-1 dev (each ≤ 30s) — "sanity check" matching paper numbers
- 5–10 clips from WenetSpeech test-net — real-audio representative

**L2: Long audio (reflects real use)**
- Self-collected or public-source 1 segment of 17–25 min Chinese news/interview, with manual or semi-automatic ground truth
- This layer is the proxy for dosmoon's real use; cross-engine stability differences on long audio surface here

**L3: Hard samples (catch the failure modes)**
- Podcast clips with background music
- Multi-speaker dialogue clips
- Political/economic commentary dense with proper nouns, foreign words, numbers
- This layer is not for scoring, but for "see which engine breaks on the hard cases"

**First version does L1+L2 only**, L3 is later research.

### Not recommended
- ❌ Evaluate using only AISHELL-1 — low discrimination, conclusions meaningless
- ❌ Use synthetic or TTS audio as ground truth — disconnected from real scenarios
- ❌ Run the full 6+ hour test sets — measurement cost is high; research-shaped form does not need paper-grade reproducibility

## 7. Long audio handling strategy

dosmoon's actual content (17–50 min news, podcasts) far exceeds every engine's single-input limit:

| Engine | Single-input limit | Recommended long-audio strategy |
|---|---|---|
| Whisper / faster-whisper | 30s internal window (auto-segmented) | Engine handles segmentation, but use VAD filter |
| Paraformer | No hard limit (memory-bound) | Feed directly, or VAD into 30s slices |
| SenseVoice | ~30s recommended per call | Pre-VAD (FunASR pipeline includes fsmn-vad) |
| FireRedASR-AED | **60s hard limit** | Application-layer VAD segmentation + post-processing concatenation |
| FireRedASR-LLM | **40s hard limit** | Same, tighter |
| FireRedASR2S | Same as v1 | Same |

aistack already implements application-layer 12-min chunking + word-LCS stitching on the Parakeet path (see the addendum in `parakeet-on-consumer-gpu`). **This mechanism can be reused for FireRedASR-AED**, but segmentation granularity must shrink from 12min to ~50s (close to the 60s ceiling with margin), and stitching strategy may need to switch from word-LCS to character-LCS (Chinese has no word boundaries).

## 8. Runtime considerations (aistack 8 GB card view)

### Integration cost comparison
| Engine | Already installed deps | New deps | Model size | Effort |
|---|---|---|---|---|
| Paraformer-large(-vad-punc) | ✅ FunASR + funasr installed | 0 | ~880 MB download | **~2 hours** (write paraformer.py copying sensevoice.py) |
| SenseVoiceSmall | ✅ installed + integrated | 0 | already downloaded | 0 (already in production path) |
| FireRedASR-AED | ❌ | FireRedASR's inference package + PyTorch | ~4.4 GB download | **~half a day** (new aistack/asr/fireredasr.py + write chunked-mode adapter for the 60s limit) |
| FireRedASR2S-AED | ❌ | Same but newer | unknown | Half day to 1 day, wait for community to settle a bit before doing |
| Whisper-large-v3 | ✅ faster-whisper installed | 0 | ~3 GB download | 0 (API-layer `model=large-v3`) |
| Fun-ASR (open version) | partial | TBD | TBD | Wait for weights/inference to stabilize |

### VRAM coexistence budget (key constraint)
When multiple models are resident on an 8 GB card:
- Paraformer + SenseVoice + Whisper-small coexisting ≈ < 4 GB total, comfortable
- FireRedASR-AED resident alone ~5 GB, **cannot coexist with a large Ollama model**, must rely on the `_model_cache` `evict_category` mechanism for hot swap
- aistack's existing hot-swap mechanism (asr-main mutual-exclusive residence) already handles this; FireRedASR additions just continue under `category="asr-main"`

### License overview
All permissive, no copyleft contamination:

| Engine | License | Commercial | Integrate into aistack / product form |
|---|---|---|---|
| Paraformer / SenseVoice (FunASR) | MIT | ✅ | ✅ |
| FireRedASR / FireRedASR2S | Apache-2.0 | ✅ | ✅ |
| Whisper / faster-whisper / CTranslate2 | MIT | ✅ | ✅ |
| Belle-whisper-large-v3-zh | Inherits Whisper upstream MIT | ✅ | ✅ |
| Fun-ASR | TBD (per GitHub repo terms) | TBD | Hold |
| Qwen2-Audio | Commercial license (Alibaba) | ⚠ read terms | Not currently evaluated |

## 9. aistack's Chinese ASR measurement plan (phased when implementing)

> This document is desk research; this section only lists the plan, not the action. Real implementation follows backlog scheduling.

### Phase 0 (infrastructure, must come first)
- Add L1+L2 Chinese audio to `bench/audio/` (AISHELL sample + WenetSpeech sample + 1 long-audio piece)
- Write `bench/zh_eval.py`: jiwer + zhconv + number normalization, output CER
- Keep existing `bench/asr_eval.py` English WER eval; parallel, no conflict

### Phase 1 (smallest incremental value)
- Add `aistack/asr/paraformer.py` (mirror sensevoice.py structure, reuse FunASR runtime)
- Add a Paraformer-priority branch for zh in `_select_for_auto`
- Run SenseVoice vs Paraformer-large CER comparison on L1+L2, write a daily log

### Phase 2 (integrate FireRedASR-AED)
- Add `aistack/asr/fireredasr.py`, integrate FireRedASR's inference package
- Write 60s segmentation + concatenation logic, reference Parakeet chunked mode
- Compare Paraformer / SenseVoice / FireRedASR-AED on L1+L2

### Phase 3 (decide the final zh routing choice)
- Look at Phase 2 data, decide whom `_select_for_auto` defaults to for zh
- Write the decision rationale into the daily log and the aistack product-path design document (internal)

### What we will never do in aistack
- FireRedASR-LLM (8.3B+) integration — too large for aistack's research hardware configuration
- Qwen2-Audio integration — commercial license + deployment complexity, beyond aistack's boundaries
- Fun-ASR DashScope API integration — that is a cloud service, not a local engine, conflicts with aistack's positioning

---

## Open questions (to validate by measurement)

1. **Real Chinese CER of SenseVoiceSmall:** public benchmark tables only list SenseVoice-L's 4.47%; the Small variant has no standard number. We need to measure one ourselves
2. **FireRedASR-AED's 60s segmentation stability on dosmoon long audio:** does segmentation strategy variance drift CER? Does word-LCS work on Chinese (no word boundaries), or must it switch to character-level?
3. **WenetSpeech-net subset's representativeness for dosmoon real scenarios:** how close is Bilibili/podcast content to the international news commentary dosmoon cares about? May need self-collected small samples
4. **Real improvement of Whisper-large-v3 + Chinese fine-tune (Belle family):** the reported 24–65% relative improvement, how much is retained on long audio?
5. **`speech_paraformer-large-vad-punc_*` bundled VAD vs aistack's existing fsmn-vad — do they conflict?** May the FunASR pipeline configuration layer have path collisions
6. **Real VRAM usage of multiple FunASR models resident simultaneously** (SenseVoice + Paraformer coexisting): the model card's "234 M" + "220 M" does not add to ~500 MB; runtime buffers will behave how?

## References

### Papers
- [FireRedASR: Open-Source Industrial-Grade Mandarin Speech Recognition Models from Encoder-Decoder to LLM Integration](https://arxiv.org/abs/2501.14350) (2025-01)
- [FireRedASR2S: A State-of-the-Art Industrial-Grade All-in-One Automatic Speech Recognition System](https://arxiv.org/html/2603.10420v1) (2025)
- [Fun-ASR / FunAudio-ASR Technical Report](https://arxiv.org/abs/2509.12508) (2025)
- [FunASR: A Fundamental End-to-End Speech Recognition Toolkit](https://www.isca-archive.org/interspeech_2023/gao23g_interspeech.pdf) (Interspeech 2023)
- [Advocating Character Error Rate for Multilingual ASR Evaluation](https://aclanthology.org/2025.findings-naacl.277.pdf) (NAACL 2025)
- [What is lost in Normalization? Exploring Pitfalls in Multilingual ASR Model Evaluations](https://arxiv.org/html/2409.02449v3)
- [Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual and Long-Form Speech Recognition Evaluation](https://arxiv.org/abs/2510.06961)

### Repos and model cards
- [FireRedTeam/FireRedASR (GitHub)](https://github.com/FireRedTeam/FireRedASR)
- [FireRedTeam/FireRedASR2S (GitHub)](https://github.com/FireRedTeam/FireRedASR2S)
- [FireRedTeam/FireRedASR-AED-L (HuggingFace model card)](https://huggingface.co/FireRedTeam/FireRedASR-AED-L)
- [modelscope/FunASR (GitHub)](https://github.com/modelscope/FunASR)
- [FunAudioLLM/Fun-ASR (GitHub)](https://github.com/FunAudioLLM/Fun-ASR)
- [BELLE-2/Belle-whisper-large-v3-zh (HuggingFace)](https://huggingface.co/BELLE-2/Belle-whisper-large-v3-zh)
- [SpeechColab/Leaderboard (GitHub)](https://github.com/SpeechColab/Leaderboard)

### Evaluation tools
- [jiwer (Python WER/CER library)](https://github.com/jitsi/jiwer)
- [zhconv (Traditional/Simplified conversion)](https://github.com/gumblex/zhconv)
- [Open ASR Leaderboard (HuggingFace Space)](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
