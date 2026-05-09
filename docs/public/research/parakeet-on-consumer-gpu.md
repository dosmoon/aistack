---
title: NVIDIA Parakeet TDT for long audio on consumer GPU
date: 2026-05-07
updated: 2026-05-08
tags: [asr, parakeet, nemo, nvidia, 8gb-vram, long-audio, chunked-transcription]
sidebar:
  order: 2
---

# NVIDIA Parakeet TDT for long audio on consumer GPU

> **TL;DR** Running Parakeet TDT on an 8 GB consumer card (RTX 4060 Laptop and similar) for 50+ minute audio requires three configurations to land together: local attention, subsampling chunking, **do not** touch the decoding strategy's `preserve_alignments`. No single NVIDIA document spells out this combination.
>
> Even with the right configuration, single `transcribe()` calls on long audio still exhibit 2–4× wall-time drift, reserved VRAM spiking to 13 GB, and occasional tail timeouts. **2026-05-08 addendum:** layering an application-side **12-minute window + 2-minute overlap** chunker with word-LCS stitching on top of the NeMo call eliminates all of these "unpredictable" side effects — recall actually goes up (98.1% / 95.5% on 25/50 min), wall stays at RTF ≈ 0.008 across any duration, and VRAM locks at 7–8 GB. See "Application-layer chunking" below.

## Who should read this

- Self-deploying Parakeet TDT 0.6B v2 / v3 on an 8–12 GB consumer GPU
- Long audio (over 30 min) is OOMing, absurdly slow, or returning empty segment-timestamp arrays
- Wanting to understand why `aistack/asr/parakeet.py` flips the switches it does

## Context

NVIDIA Parakeet TDT is one of the best consumer-grade ASR options today — a 50-minute English political speech runs in 62 seconds on an RTX 4060 Laptop (8 GB VRAM) on a cache hit, RTF ≈ 0.021, about **80× real time**. But the default configuration does not run; three independent layers of knobs have to cooperate, and each layer has a trap.

What follows is the working combination, validated on **a real 50-minute audio measurement** (a Rubio May 5 press conference on Iran, 47.8 MB mp3).

## The working configuration

```python
from nemo.collections.asr.models import ASRModel

model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")

# (1) Attention stage: avoid the O(N²) attention-matrix blowup
model.change_attention_model("rel_pos_local_attn", [256, 256])

# (2) Subsampling stage: avoid downsampling consuming the full audio at once
model.change_subsampling_conv_chunking_factor(1)   # 1 = auto-pick chunk size

# Note: DO NOT call change_decoding_strategy with preserve_alignments=True.
# See "Trap #3" below.

model.eval()

# Transcribe
results = model.transcribe([wav_path], timestamps=True, num_workers=0, batch_size=1)
```

`num_workers=0` avoids file-lock contention between NeMo's internal manifest.json and DataLoader subprocesses on Windows — this is the default in NeMo's own `examples/asr/transcribe_speech.py`, not something we invented.

## What each of the three knobs solves

### Knob #1: `change_attention_model("rel_pos_local_attn", [256, 256])`

**Default problem:** Parakeet TDT defaults to full self-attention (rel_pos), which is **O(N²)** in audio token count N. On an 8 GB card, 2–3 minutes of audio already OOMs. NVIDIA's official ceiling: A100 80 GB + full attention = 24-min audio. Scaled to an 8 GB card ≈ a few minutes.

**Fix:** switch to Longformer-style local attention — each position only sees ±256 frames (≈ ±20 s of context at 80 ms/frame), bringing compute down to **O(N × 256)**, linear in audio length.

**Cost:** about 1–3% WER (the model loses long-range context, so cross-sentence anaphora / coreference suffers slightly). On an 8 GB card, this trade is worth it.

**Official references:**
- Parakeet HF model card: `change_attention_model("rel_pos_local_attn", att_context_size=...)` is the officially recommended path
- FastConformer paper + NVIDIA blog: local attention's design goal is exactly to support long audio on small cards

### Knob #2: `change_subsampling_conv_chunking_factor(1)`

**Default problem:** the FastConformer encoder's first layer is subsampling (4× downsampling, turning raw acoustic frames into tokens that attention can process). This step **by default loads the entire audio's intermediate activations into VRAM at once** — a 50-minute audio blows past 8 GB at this layer alone, before attention even runs. NVIDIA's own FastConformer research blog states it directly: "the downsampling module at the earliest stage can take more memory than the actual forward pass since it directly operates on the audio sequence which may not fit in memory for very long audio files".

**Fix:** call `change_subsampling_conv_chunking_factor(1)` to switch subsampling to chunked processing — each chunk's activations are computed, released, then the next chunk runs. `1` means auto-pick chunk size (recommended).

**Explicit cost:** essentially none — only memory allocation patterns change; computation results do not, so WER is unaffected.

**Implicit benefit (this is the key finding **not stated in the official docs**):** in the long-audio + local-attention combination, enabling chunking **also fixes the bug where NeMo's long-audio path returns an empty segment-timestamp array** — this is in the family of NeMo open issues (see discussion around [#14714](https://github.com/NVIDIA-NeMo/NeMo/issues/14714)).

**Measured comparison** (same 50-min Rubio audio, aistack `/v1/audio/transcriptions`):

| Configuration | NeMo-returned segment count | Notes |
|---|---|---|
| Local attention only (no chunking) | **1** (a single 0–2986s segment) | Long-audio segment-timestamp path triggers the bug |
| Local attention + chunking | **788** (sentence-level, with correct punctuation boundaries) | NeMo's punctuation-aware splitting works again |

So Knob #2 fixes two problems at once, **but no NVIDIA document states** the second effect. This is one of the most valuable findings in this note.

**Official references** (what they say):
- [NeMo ASR API Reference](https://docs.nvidia.com/nemo-framework/user-guide/24.07/nemotoolkit/asr/api.html): `subsampling_conv_chunking_factor` is an optional parameter; values can be powers of 2 / 1 (auto) / -1 (disable); only depthwise-separable conv subsampling models support it
- [Parakeet HF model card #15](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/discussions/15): the recommended 8 GB configuration uses both knobs together

**What official does not say:** turning it on also fixes the segment-timestamps path. This requires measurement to find.

### Knob #3: `change_decoding_strategy(preserve_alignments=True)` — **do not touch**

While reading NeMo docs you will hit a recommendation along the lines of "to make NeMo emit segment timestamps natively, explicitly call `change_decoding_strategy` with `preserve_alignments=True / compute_timestamps=True / segment_seperators=['.','?','!']`". Sounds reasonable. **It is a trap.**

**Measured cost** (same RTX 4060 Laptop + same 50-minute mp3):

| Configuration | VRAM | Windows shared system memory | Total | 50min cache-hit inference time |
|---|---|---|---|---|
| Without `change_decoding_strategy` | 8 GB | 10 GB | 18 GB | **62 s** |
| With `preserve_alignments=True` | 8 GB | **30 GB** | **38 GB** | **≥120 s (client timeout)** |

The extra 20 GB working set **spills from VRAM into Windows shared system memory (PCIe path)**. PCIe bandwidth ≈ 1/30 of GDDR6, so every GPU kernel that touches that region pays a bus traversal — half the working set becomes IO-bound. Compute time is unchanged but IO doubles, ending in 2× slowdown.

**Mechanism (reverse-engineered from source after measurement):**

1. NeMo source comments are explicit: `preserve_alignments is not implemented for Frame-Looping + CUDA graphs` — the RNNT/TDT decoder **falls back** from the CUDA-graph fast path to the Python-loop path, and per-step workspace tensors no longer get reused
2. `preserve_alignments=True` makes the hypothesis retain per-frame alignment-logit tensors, length equal to acoustic frames T. 50 min ≈ 75,000 frames × vocab size V+1 ≈ 1000 × 4 bytes = hundreds of MB of alignment data alone
3. (1) and (2) compound; with the CUDA graph disabled, intermediate tensors no longer recycle, and the working set inflates by +20 GB

**Relation to NeMo issue #14714:** that OPEN issue reports a "Boolean value of Tensor" runtime error on `preserve_alignments=True` + parakeet-tdt-0.6b-v3 + timestamps. We did not hit the crash but we hit the same-root-cause performance collapse — same bug class.

**Correct usage:** do not call `change_decoding_strategy`. `transcribe(timestamps=True)` is enough to get word-level timestamps; segment timestamps are already fixed by Knob #2. If NeMo ever spits empty segments on long audio (it should not), a post-processing word→sentence splitter (taking word timestamps and merging on sentence-final punctuation + silence gaps) is far safer fallback than enabling `preserve_alignments`.

## Composite performance baseline

Machine: RTX 4060 Laptop (8 GB VRAM) + i9 13th gen + 64 GB DDR5 + Windows shared GPU memory ceiling 31 GB, torch 2.7.1+cu126, cuDNN 9.7.1, NeMo 2.7+.

Full end-to-end baseline (includes ffmpeg transcode, model inference, word/segment timestamp computation, verbose_json serialization, HTTP response):

| Audio length | Cold start | Cache hit (good state) | RTF (best) |
|---|---|---|---|
| ~80 s | ~25 s | ~3 s | ~0.04 |
| 4 min | ~37 s | ~13 s | 0.05 |
| 12 min | ~30 s | **5.7 s** | **0.008** |
| 17 min | ~12 s | ~10 s | 0.010 |
| 25 min | ~52 s | 13–20 s | 0.009 |
| 50 min | ~120 s | 60–80 s | 0.021 |
| 99 min | — | ~490 s | 0.082 (shared memory ceiling hit) |

Short-audio RTF is high (4 min → 0.05) because fixed overhead (ffmpeg transcode + JSON serialization + HTTP transfer) cannot amortize, not because the GPU is slow. The medium-to-long range (12–25 min) reveals real GPU speed, **RTF 0.008 ≈ 125× real time** is the steady-state best on this machine.

> **2026-05-08 addendum:** the table above shows single `transcribe()` call performance, **not** what aistack actually sees today. After the application-layer chunker (next section), any-length audio is split into 12-minute windows that run one by one, wall time matches the single-block baseline, **and the wall drift disappears**. Updated steady-state numbers in the comparison table at the end of "Application-layer chunking".

## Application-layer chunking: 12-minute window + 2-minute LCS stitch

> **Background:** the previous section shows the "theoretical ceiling" under correct configuration, but in production three problems do not go away: (a) wall time drifts 2–4× depending on prior request history (see "Request-to-request memory dynamics" below); (b) 50+ minute audio occasionally trips client-side 120s timeouts with no reliable predictor; (c) reserved VRAM on long audio can climb to 13 GB (occupying Windows shared memory), only forced back down by reducing input length.
>
> All three share a root cause: "long inputs + cuDNN workspace + caching allocator + tensor shape changes" — non-determinism beneath the NeMo call, inside PyTorch. We cannot fix it, but we can **route around it**: split any long audio into equal-length segments and call independently, with each segment's length sitting in the "short-input steady-state" zone.

### Algorithm

The chunking rule is simple: **the last block must be ≥ 5 minutes**.

```python
def plan_chunks(T, window=720, overlap=120, min_last=300):
    if T <= window:
        return [(0, T)]
    stride = window - overlap        # 600s
    chunks = []
    start = 0
    while start + window < T:
        chunks.append((start, start + window))
        start += stride
    tail = T - start
    if tail < min_last and chunks:
        s, _ = chunks[-1]
        chunks[-1] = (s, T)            # short tail merges into previous block
    else:
        chunks.append((start, T))
    return chunks
```

Parameter choice rationale follows in "Experimental data" below. Worst case is a final block of stride + min_last - eps ≈ 15 minutes, still inside Parakeet's safe zone.

Each block is fed independently to `model.transcribe()`, returning a relative-timestamped word list. Add the block's offset back to get absolute time, then **at each adjacent chunk's overlap region run word-level LCS (Longest Common Subsequence)** and split at the LCS midpoint to stitch the two sides together:

```python
def stitch_words(a, b, seam_start, seam_end):
    # a is kept entirely for [< seam_start]; b is kept entirely for [≥ seam_end];
    # in the overlap [seam_start, seam_end], LCS finds words both sides agree on,
    # and the cut lands at the LCS midpoint — never on a word that exists on
    # only one side.
```

The final segment list is regenerated from the merged word list using sentence-final punctuation + silence gaps — so seam points never cut sentences in half.

### Why LCS

The simple approach is "split the overlap at the time midpoint", but the risk is that the cut lands inside a word recognized on one side, breaking a segment / sentence. LCS finds the word sequence both chunks agreed on in the overlap and cuts at its midpoint — the seam always lands on a word both sides identified, never one missing from one side.

Across 13 measured seams (25 + 50 + 97 min audio totaling 13 stitch points), **all clean**: no duplicate words, no LCS-omitted words, segment boundaries tidy. See `aistack/asr/_chunking.py` in commit `d8cbe56`.

### Experimental data: att_context_size choice

NVIDIA's model card recommends `[256, 256]`. HF discussion #15 uses `[128, 128]`. Compared on real 25 + 50 min audio (driven by aistack's bundled `bench/run_experiments.py` automation):

| `att_context_size` | 25min wall | 25min recall | 50min wall | 50min recall | VRAM peak |
|---|---|---|---|---|---|
| `128, 128` | 12.8s | 95.98% | 21.4s | 94.00% | ~7.8 GB |
| `256, 256` (recommended) | 15.6s | **97.19%** | 24.0s | **95.16%** | ~7.8 GB |
| `512, 512` | 16.5s | 97.49% | 26.1s | 95.68% | ~7.8 GB |

- VRAM peak is **essentially constant** across the three (< 15 MB difference) — context window is not the memory bottleneck
- Recall return curve is clearly concave: 128 → 256 +1.2pp (significant), 256 → 512 only +0.3pp (half the marginal return)
- Wall cost roughly linear, ~+2–3 s per doubling
- **256 is the economic sweet spot**, agreeing with the model card

> **Closes the previous open question** — "is `[128, 128]` more VRAM-friendly on a 4060 Laptop?" The answer is **no** (difference within measurement noise), and the 1.2pp recall loss is not worth it. Stay on 256.

### Experimental data: overlap choice

Fixed at `[256, 256]`, sweep overlap:

| overlap | 25min recall | 50min recall | 50min reserved VRAM | Notes |
|---|---|---|---|---|
| 60 s (old default) | 97.34% | 95.39% | 7.8 GB | 25min tail-merge stretches the last block to 14 min |
| **120 s (new default)** | **98.11%** | **95.50%** | **7.6 GB** | 25min splits cleanly into 3 blocks; last block exactly 5 min |
| 180 s | 98.13% | 94.71% | **13.0 GB** ⚠ | 50min last-block merges to 13.77 min, exceeding the 12-min safe zone |

**120 s wins** — highest recall, no wall increase, VRAM actually drops. The reason is tied to the `min_last` merge mechanism:

- overlap=60s (stride=11min): 25min audio's tail = 1502s − 1320s = 3min < 5min → merges into previous block → last block 14 min (over window)
- overlap=120s (stride=10min): tail = 1502s − 1200s = 5min (exactly the threshold) → independent block → last block 5 min (safe)
- overlap=180s (stride=9min): 50min audio's tail = 2986s − 2700s = 4.77min < 5min → merges → last block 13.77 min (over window) → VRAM jumps to 13 GB

So the answer to "is more overlap always better?" is **no** — beyond a certain point the tail-merge mechanism stretches a single block past the 12-min threshold, making things worse.

### What chunking does **not** fix: Parakeet's own occasional word drops

The 13 seams stitch cleanly, **but** Parakeet in chunked mode occasionally fails to recognize short crosstalk inside a chunk. Example: at 720s in the 50min audio, ground truth has "Do they get two questions for these? Two questions" (11 words); sweeping all 9 configurations (ctx-128/256/512 × overlap-60/120/180), **no configuration** recovers all 11. 

Mechanism: this passage is an interjecting voice, possibly with crosstalk. Unrelated to chunking — in chunked mode, chunk B captured the segment (absolute time 720.9s sits inside chunk B's [600, 1320], 120s past chunk B's start with full warm-up), **Parakeet itself misses it**. Changing the context window / increasing overlap / changing the LCS cut strategy — none works.

Engineering conclusion: this is a Parakeet model limitation (poor short crosstalk recognition). If audio quality on this kind of content ever becomes critical, the direction is to switch to Whisper (more robust to crosstalk), not to keep tuning Parakeet parameters.

### Composite steady state (chunked mode)

Same machine as above, application-layer chunking + default parameters (`window=720, overlap=120, min_last=300`):

| Audio length | Chunks | wall (warm) | RTF | reserved VRAM peak |
|---|---|---|---|---|
| 12 min | 1 (no split) | 5.3 s | 0.007 | 7.8 GB |
| 25 min | 3 | 12.6 s | 0.008 | 7.8 GB |
| 50 min | 5 | 26.0 s | 0.009 | 7.8 GB |
| 97 min | 9 | 44.5 s | 0.007 | 7.8 GB |

Across an 8× duration spread, VRAM peak is **completely flat**. This is exactly what the chunker exists to solve — fold "long input → unpredictable memory behavior" into "equal-length blocks → short-input steady state".

> This is what aistack actually runs today. The numbers in the previous section's "Composite performance baseline" table (especially 50min 60–80s wall, 99min 490s wall) reflect the pre-chunking single-call version, now superseded. That table is kept for comparison.

## Request-to-request memory dynamics

> This section is an engineering phenomenon discovered by the aistack team on 2026-05-07, written up as a mechanism explanation. No NVIDIA / PyTorch / NeMo official document discusses these pieces together.

### Phenomenon

**The same audio drifts 2–4× in wall time depending on prior request history. In the worst case, slower than a cold start.**

Measured (same 25-min audio, aistack `/v1/audio/transcriptions`):

| Warm-up path | wall #1 | wall #2 | reserved peak |
|---|---|---|---|
| Cold → 50min × 2 → 25min | 20 s | 20 s | 24.7 GB |
| Cold → 12min × 2 → 25min | 15 s | **13 s** | 14.5 GB |
| Cold → 25min × 3 (no warm-up) | 52 s | 27 s | 13.4 GB |

Measured (50-min audio under different contexts):

| Scenario | wall |
|---|---|
| 50 min cache-hit baseline | 69 s |
| 25 min × 1 then 50 min | **175 s** (56 s slower than even a cold start) |
| aistack restart cold-start 50 min | 119 s (includes ~25 s model load) |

### Mechanism

From PyTorch official docs + Z. DeVito's caching-allocator deep dive + NeMo / Parakeet TDT architecture analysis:

**1. PyTorch maintains a GPU memory pool (caching allocator)**

Freed tensors leave their **blocks in the pool for reuse**, not returned to the CUDA driver. The design intent is to avoid `cudaFree` (a synchronous device call that breaks the CUDA-CPU pipeline).

**2. The pool's shape is dictated by the previous request's workload**

- Last 50min Rubio request leaves a 24 GB pool full of "50min-shape" free blocks
- Next 25min request needs new shapes: cuDNN re-selects conv algorithms and requests new-size workspaces
- Old blocks must be split or new blocks allocated → bookkeeping + sync overhead

**3. Different shapes vary widely in compatibility**

- 12min → 25min: cuDNN algorithms are essentially the same family, pool extends directly, lowest wall (13s)
- 50min → 25min: cuDNN re-selects, pool fragments, medium wall (20s)
- Cold → 25min: rebuild pool + load model, highest wall (52s including 25s load)

**4. Parakeet TDT's architecture makes it more sensitive**

- The TDT decoder runs two networks per timestep (prediction + joint), with duration deciding how many frames to skip → frequent Python control-flow synchronization with the GPU
- FastConformer 8x subsampling + local attention has large cuDNN workspaces strongly correlated with input length
- Higher stream-event density than simple CTC models, with more pool-shape variation

### Practical impact

**Positive:** running same-shape (same length, same model, same language) audio in sequence yields strong cache reuse — best observed RTF 0.008 (12min Trump audio, repeated runs).

**Negative:** mixed-length workloads have unpredictable wall drift; the worst case can exceed cold start, because "polluted state" + "pool reorganization overhead" > "model load overhead".

### What does not work

We tried `torch.cuda.empty_cache()` to "fix fragmentation" automatically — it failed:

- Calling empty_cache after the first request causes the second request to hang inside a worker-thread CUDA call
- The GPU lock never releases; the entire ASR endpoint becomes unavailable; only killing the process recovers
- Mechanism: `empty_cache()` internally calls `cudaFree` on each free block; `cudaFree` is a synchronous call (per NVIDIA docs), waiting for all streams to complete
- If NeMo / cuDNN have unfinished events on internal streams, the synchronous wait can deadlock against cuDNN descriptor lifecycle
- Not Windows-specific, this is the empty_cache antipattern PyTorch maintainers explicitly warn against (see [zdevito's deep dive](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html))

### What works

1. **Batch same-length, same-language workloads** — best and most stable
2. **Mixed workloads, batched by class** — avoid frequent shape changes
3. **Kill aistack and restart between workload-class switches** (25–30 s model load cost is much faster than the "polluted state" penalty)
4. **Accept wall drift as an engineering reality of local ASR** — document it for clients; do not promise strict SLA

aistack **does not auto-detect-and-fix** this phenomenon — measurement convinced us no reliable heuristic exists (too many coupled variables; the rule does not exist), and the auto-fix path has been falsified.

> **2026-05-08 addendum:** the wall-drift phenomenon described here **largely disappears** after the application-layer chunker — chunking feeds equal-length 12-min blocks regardless of input audio, and cuDNN algorithm choice + caching allocator pool shape stay stable across requests. 9 chunks for a 97-min audio and 1 chunk for a 12-min audio show identical VRAM peaks (see steady-state table above). This section is kept because: (a) when chunking is disabled (`AISTACK_PARAKEET_CHUNK_DISABLE=1`, e.g., for someone with an 80 GB card) the phenomenon returns; (b) it is the underlying mechanism, valuable to understand for future debugging.

## Why the documentation reads like scripture

NVIDIA's information about this configuration is spread across 7 layers:

| Layer | Source | Says | Missing |
|---|---|---|---|
| 1 | NeMo User Guide | "long audio inference is supported" | how to configure |
| 2 | NeMo API Reference | parameter semantics, valid ranges | side effects, combination rules |
| 3 | FastConformer research blog | design motivation (subsampling memory issue) | operational details |
| 4 | NeMo source docstrings | key warnings like "preserve_alignments not implemented for CUDA graphs" | not quantified |
| 5 | Parakeet HF model card | A100 + full attention ceiling, recommended API | consumer-card scenario |
| 6 | HF model card discussion | community-measured 8 GB recipe | unofficial, no guarantees |
| 7 | GitHub issue tracker | open bug reports (e.g., #14714) | not fixed |

**No layer is complete on its own.** For example:
- Reading Layer 1's "default 1" suggests no call is needed; in reality you must call it explicitly
- Layer 4 tells you preserve_alignments disables CUDA graphs, but does not say it blows up to 30 GB
- Layer 6 tells you which two switches to flip, but not why

This note's goal is to **cut horizontally across the 7 layers and fill in the missing connections** — especially Layer 5 (Knob #2 fixes segment timestamps) and Layer 7 (preserve_alignments costs 20 GB).

## Open questions

- **Mechanism for chunking → segment-timestamp fix:** measurement confirms the two are linked, but we have not stepped through NeMo source line-by-line to identify which logic gets enabled by chunking. Hypothesis: chunking causes subsampling output to insert some marker at boundaries, allowing the downstream segment detector to split correctly; but only a hypothesis.
- **Boundary on audio longer than 97 min:** this note's baseline measures up to 97 minutes (chunked mode 9 blocks, 44s wall, steady-state VRAM). NVIDIA's official position is "local attention + chunking on 8 GB can reach 11 hours", but application-layer chunking has already "deconstructed" the ceiling problem into "any length = N independent runs", so theoretically there is no ceiling — worth confirming on some very long audio.
- **Why "default 1" still needs an explicit call:** NeMo API Reference says `subsampling_conv_chunking_factor` defaults to 1 (auto), but our measurement shows **without the explicit call the segment-timestamp bug triggers**, with the call it works. Hypothesis: `change_attention_model` internally resets subsampling state, forcing a re-set. Worth someone digging through NeMo source to confirm.

> **Resolved** (formerly open, closed 2026-05-08):
> - **Optimal att_context_size:** swept 128/256/512 on 25/50 min real audio; conclusion is 256 (matching the model card). See "Application-layer chunking: experimental data: att_context_size choice" above.
> - **Wall-drift controllability:** the phenomenon described in the previous section is PyTorch-internal behavior and cannot be eliminated; but the application-layer chunker "routes around" it — all requests run equal-length blocks, shapes no longer vary, the pool no longer reorganizes, wall stays consistent across requests (measured: 9 separate 50min requests all in 24–26 s).

## Companion code

aistack's implementation of this configuration:

[`aistack/asr/parakeet.py`](https://github.com/dosmoon/aistack/blob/main/aistack/asr/parakeet.py) — the NeMo call layer:

- `_get_model()` calls `_maybe_switch_to_local_attention()` and `_maybe_enable_subsampling_chunking()`
- `_configure_timestamp_decoding()` is kept as a future opt-in path, but `_get_model` **does not** call it (with detailed comments explaining the measured 20 GB cost)
- `transcribe()` computes duration → `plan_chunks()` decides how many chunks → single chunk goes through original `_run_one_pass`, multi-chunk goes through `_run_chunked` invoking ffmpeg to slice wav + NeMo + `stitch_words` to merge

[`aistack/asr/_chunking.py`](https://github.com/dosmoon/aistack/blob/main/aistack/asr/_chunking.py) — the application-layer chunker:

- `plan_chunks()` chunking rules (including `min_last` merge)
- `stitch_words()` LCS stitching + time-midpoint fallback
- `shift_words()` chunk-relative → absolute time offset

All parameters are centrally managed by `ParakeetConfig` in `aistack/config.py`; env names + defaults are in the configuration reference page on the published site.

## References

- [NeMo ASR Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html)
- [NeMo ASR API Reference (24.07)](https://docs.nvidia.com/nemo-framework/user-guide/24.07/nemotoolkit/asr/api.html)
- [Parakeet TDT 0.6B v3 — HuggingFace model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [Parakeet TDT 0.6B v2 — HF Discussion #15 (working 8 GB long-audio recipe)](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/discussions/15)
- [NVIDIA Research — Fast Conformer with Linearly Scalable Attention](https://research.nvidia.com/labs/conv-ai/blogs/2023/2023-06-07-fast-conformer/)
- [NeMo Issue #14714 (OPEN)](https://github.com/NVIDIA-NeMo/NeMo/issues/14714) — `preserve_alignments=True` failure on parakeet-tdt-0.6b-v3 timestamps path
- [NeMo PR #10950](https://github.com/NVIDIA-NeMo/NeMo/pull/10950) — Timestamps to transcribe (origin of segment_seperators design)

## Acknowledgments

The measured comparison (especially the 8 GB VRAM + 30 GB shared RAM pair) by aistack team members turned the real cost of `preserve_alignments=True` from "incompatibility hinted at in a docstring" into "memory overflow quantified to GB". Reading source comments alone, we would have underestimated the cost by two orders of magnitude.

---

*This is one of the dosmoon aistack project's research notes. See [the research index](./) for others.*
