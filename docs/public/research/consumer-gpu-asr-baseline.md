---
title: Consumer-GPU local ASR performance baseline
date: 2026-05-07
updated: 2026-05-08
tags: [asr, parakeet, benchmark, consumer-gpu, self-hosting]
sidebar:
  order: 1
---

# Consumer-GPU local ASR performance baseline

> **TL;DR** RTX 4060 Laptop (8 GB VRAM, consumer-grade laptop dGPU) running NVIDIA Parakeet TDT 0.6B v3: best steady-state RTF 0.008 (≈ 125× real time). In practice, single-request wall time can drift 2–4× depending on the shape of the previous request's GPU memory state. **In the worst case, wall time exceeds a fresh cold start by 30%+.** This note gives an independently reproducible hardware / software / workload / performance bundle plus the request-to-request memory-dynamics measurements, so the reader can decide whether local ASR fits their own situation.

> **2026-05-08 addendum:** the wall-time figures in this note (especially the 60–490 s and 2–4× drift on 50/99-min audio) are aistack's **single-call NeMo** numbers, reflecting "raw Parakeet on an 8 GB card with no application-level wrapping." Aistack's current production path now uses **application-layer 12-minute chunking + LCS stitching**: RTF stays at 0.007–0.009 across any duration, reserved VRAM locks at 7.8 GB, wall time no longer drifts (97 min audio → 44 s). See [parakeet-on-consumer-gpu](parakeet-on-consumer-gpu/) — "Application-layer chunking" — for the mechanism and the data. The numbers in this note are kept as the "raw NeMo" reference baseline.

## What this note is for

aistack's product position is: **for AI tasks where local GPU performance is sufficient, there is no need to go to a cloud API.** But "sufficient" is a concrete engineering question, not a slogan. This note breaks "sufficient" into verifiable numbers — what the hardware looks like, what workload runs, what performance comes out — so readers can compare those numbers against their own situation (monthly audio volume, latency tolerance, data-compliance requirements, etc.) and judge for themselves.

The note **does no horizontal comparison**: no commercial-ASR pricing, no cost-per-minute charts, no "aistack is cheaper than X" claims. Comparisons like that go stale within a month and have to be re-explained for every use case anyway. Take our numbers, plug in the current pricing of whichever service you care about, and do the multiplication yourself — that arithmetic is more reliable than anything we could publish.

## Test hardware

| Item | Spec |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop (8 GB VRAM, consumer-grade laptop dGPU, entry-tier SKU) |
| CPU | Intel Core i9 13th gen |
| System RAM | 64 GB DDR5 |
| Windows shared GPU memory ceiling | 31 GB (manually configured, not the default) |
| OS | Windows 11 |
| Driver | NVIDIA Studio Driver 5xx series |
| Whole-machine power (peak inference) | ~70 W (measured; CPU + GPU + memory combined; not GPU-only) |

A mid-range gaming / creator laptop configuration — the dGPU is the entry-tier 8 GB part (not a 4090 / 4080 high-VRAM model); CPU and RAM are the i9 + 64 GB combination common to creator workstations.

**Windows shared GPU memory** is the WDDM driver's "spillover area" for the GPU — when VRAM fills up, the GPU reaches into a portion of host RAM over PCIe as extended VRAM. This machine allows up to 31 GB (default is half the system RAM; raised manually here). This is important context for the performance data below.

## Software stack

| Component | Version |
|---|---|
| Python | 3.12.13 |
| torch | 2.7.1+cu126 |
| torchaudio | 2.7.1+cu126 |
| System cuDNN (bundled with torch) | 9.7.1 |
| CUDA runtime | 12.6 |
| NeMo toolkit | 2.7+ (`[asr,cu12]` extras) |
| ASR model | `nvidia/parakeet-tdt-0.6b-v3` (HuggingFace public weights) |

Model weights are a one-time ~1.2 GB download, cached locally in NEMO_CACHE_DIR; reuse does not re-download. Zero API fees, zero account signup.

Key runtime configuration (full reasoning in [parakeet-on-consumer-gpu](parakeet-on-consumer-gpu/)):

```python
model.change_attention_model("rel_pos_local_attn", [256, 256])
model.change_subsampling_conv_chunking_factor(1)
```

WER cost relative to full attention is ~1–3% (NVIDIA states this in the model card).

## Test workload

| Item | Description |
|---|---|
| Audio | 50-minute English political speech (a Rubio diplomatic press conference, 47.8 MB mp3) |
| Language | English, with diplomatic proper nouns, place names, numbers |
| Noise | Clean indoor capture, mild room reverb |
| Speakers | One main speaker plus occasional reporter questions |

Picked because it represents **a realistic medium-difficulty scenario**: long-form audio + heavy proper-noun load + natural pace variation. Not a clean read-aloud sample (à la Common Voice), and not an extreme noisy environment (open street).

## Performance data

End-to-end through aistack's `/v1/audio/transcriptions` (includes ffmpeg transcode, model inference, word/segment timestamp computation, verbose_json serialization, HTTP response).

> RTF (real-time factor) = wall time ÷ audio duration. RTF < 1 means processing is faster than real time; RTF 0.01 ≈ 100× real time.

### Steady-state reference points by duration

"Good-state" cache-hit data at each duration (specific meaning of "good state" comes in the memory-dynamics section below):

| Audio length | wall (s) | RTF | Speedup | Notes |
|---|---|---|---|---|
| 4.4 min | ~13 s | 0.05 | 20× | Fixed overhead dominates |
| 12 min | ~6–8 s | 0.008–0.011 | 90–125× | **GPU's true cruise band** |
| 17 min | ~10 s | 0.010 | 100× | Still cruising |
| 25 min | 13–20 s | 0.009–0.013 | 75–110× | Top of cruise band |
| 50 min | 60–80 s | 0.020–0.027 | 35–50× | Working set starts spilling to PCIe shared memory |
| 99 min | ~490 s | 0.082 | 12× | Shared GPU memory ceiling hit; pagefile kicks in |

Short-audio RTF is high (4.4 min → 0.05) because fixed overhead (ffmpeg transcode, JSON serialization, HTTP transfer) cannot amortize over a short duration — it is not a GPU-throughput problem. The medium-to-long range (12–25 min) shows the GPU's actual speed.

### Cold-start fixed cost

Loading the model from disk into VRAM takes about **25–30 seconds** — the "entry ticket" the first request pays. Aistack defaults to releasing after 5 minutes idle (configurable via `AISTACK_MODEL_KEEP_ALIVE_SEC`). A second request within 5 minutes skips the reload, saving the 25–30 s.

### The long-audio cliff

99-minute audio takes 490 s (RTF 0.082) — disproportionately slower than 50 min. Mechanism: once the working set exceeds the Windows shared GPU memory ceiling (31 GB on this machine), the spillover gets paged to the SSD pagefile, and every GPU kernel access of that region traverses SSD → PCIe → GDDR6. That is roughly 30× slower than GDDR6.

90–100 minutes is the upper edge of this machine's "comfortable audio length". Longer still works (Parakeet does not crash), but RTF degrades to 0.08–0.15+. A qualitative jump requires a 24+ GB VRAM card.

## Request-to-request memory dynamics (important)

**The same audio can have wall time drift 2–4× depending on the prior request history.** This is an engineering fact worth taking seriously, not measurement noise.

### Measured comparison: same 25-min audio, three warm-up paths

| Warm-up path | 25 min #1 wall | 25 min #2 wall | Reserved peak |
|---|---|---|---|
| Cold start → 50 min × 2 → 25 min | 20 s | 20 s | 24.7 GB |
| Cold start → 12 min × 2 → 25 min | 15 s | **13 s** | 14.5 GB |
| Cold start → 25 min × 3 (no warm-up) | 52 s (cold) | 27 s (warm) | 13.4 GB |

**4× wall-time gap** — same machine, same code, same audio, only the prior-request shape differs.

### Measured comparison: same 50-min audio, after being "polluted" by 25-min work

| Scenario | wall | Notes |
|---|---|---|
| 50 min cache-hit (warm baseline) | 69 s | Steady state of repeated same-shape runs |
| 25 min × 1 then 50 min | **175 s** | **56 s slower than even a cold start** |
| aistack restart cold-start 50 min | 119 s | Includes 25–30 s model load |

**In the worst case, "polluted warm state" is slower than killing the process and starting from zero.**

### Mechanism

From PyTorch official docs and maintainer Z. DeVito's caching-allocator deep-dive:

- PyTorch maintains a GPU memory pool (caching allocator); freed tensors leave their **blocks in the pool for reuse**, not returned to the CUDA driver
- Reuse requires **shape match** — cuDNN picks different conv algorithms for different input shapes and requests workspaces of different sizes
- A pool that has been populated with "50min-shape" blocks does not match the request when 25min comes in next, so the pool has to split / rearrange
- Requests that go through the split path have higher wall time; same-shape lineage (e.g., 12min → 25min) reuses the pool directly and gets the lowest wall

For the full mechanism and seven-layer documentation trace, see [parakeet-on-consumer-gpu](parakeet-on-consumer-gpu/) — "Request-to-request memory dynamics".

### Recommended usage modes (in order of performance predictability)

1. **Batched same-length, same-language workloads** (e.g., transcribe a batch of 5–15 min videos for subtitles, or a batch of 30–60 min podcasts) → best and most stable performance
2. **Mixed workloads, batched by category** (finish all short videos first, then all long audio) → good
3. **Random-length, freely interleaved** → accept 2–4× wall drift; or kill aistack and restart between workload-class switches (25–30 s cost, much less than the "pollution" penalty)

aistack **does not auto-detect memory state and trigger cleanup** — this is a measured design decision, not a missing feature. We tried `torch.cuda.empty_cache()` for automatic pool clearing; it caused the next request to hang (mechanism: cudaFree is a synchronous device call and deadlocks against NeMo's internal stream events). Reverted. Heuristics for "should I restart now" couple too many variables for any reliable implementation to exist.

## Resource usage

| Resource | Usage during 50-min audio | Notes |
|---|---|---|
| VRAM | 8 GB (full card) | Parakeet preallocates workspace by design; VRAM is always full regardless of audio length |
| Windows shared GPU memory | 14 GB | This **scales roughly linearly with audio length**: 14 GB at 50min, 22 GB at 120min, 31 GB hit at 99min |
| System RAM | ~20 GB | Includes Python process + audio buffers + the overlap mapping for Windows shared GPU memory |
| Whole-machine power | ~70 W during inference / ~25 W with model resident but idle | |

**Parakeet is VRAM-hungry but scales gracefully**: it does not crash when the working set exceeds VRAM — the spillover automatically goes to Windows shared memory (PCIe path). The cost is that PCIe is ~30× slower than GDDR6, so the closer the working set is to the ceiling the slower it gets. This is Parakeet TDT's real engineering profile: trade VRAM for compute parallelism by laying the entire audio out for the GPU at once (FastConformer's global working set), so 8 GB VRAM is always full. See [parakeet-on-consumer-gpu](parakeet-on-consumer-gpu/) for more.

## What this data means

Take these numbers back to your own situation and do the math:

**Q1: How many hours of audio per month do you transcribe?**
Estimate using "good state" 30–50× real time (not the best 100×, but the predictable median) — 1 hour of audio takes 1–2 minutes. 100 hours of audio fits in one daytime shift on one consumer card.

**Q2: Can you accept a 100–120 second cold start?**
- Yes → service can start on demand and keep the model resident between requests
- No → keep aistack running with cache TTL extended (to hours), so subsequent requests skip the cold start

**Q3: Electricity cost?**
Peak inference at 70 W; transcribing 100 hours of audio ≈ 2 hours × 70 W = 140 Wh = **0.14 kWh**. Plug your residential rate in.

**Q4: How does hardware cost amortize?**
A 6,000–8,000 RMB gaming laptop runs this workload. A desktop with a discrete 8 GB card (4060 / used 3060 12 GB / older 2080 Ti, ...) is cheaper. The amortization depends on how many years of useful life the hardware has left.

**Q5: Network and data-compliance handling?**
With local inference, neither audio nor text leaves the machine. No vendor data-retention policy, no cross-border transfer, no logging-compliance overhead.

Stitch the answers together, compare against the total bill of any cloud service you are using or considering (base rate + any feature surcharges + unused minimum commit + data-compliance / privacy review labor), and that is your decision basis. **aistack does not do this calculation for you.**

## When local is not the right choice

Honest limitations:

1. **Very low usage** (under 10 hours/month) — hardware cost cannot amortize; pay-as-you-go cloud may be more economical
2. **High peak concurrency** — a single 8 GB card runs one ASR task at a time (aistack's design choice, to avoid OOM). For scenarios with dozens of parallel streams, multi-machine local deployment or cloud elasticity fits better
3. **Strict wall-time predictability needed** — the memory-dynamics section above shows wall time drifting 2–4× under unpredictable prior history. If your scenario requires "5-min audio must return within X seconds" as a hard SLA, local is the wrong tool; use elastic cloud
4. **Zero ops requirement** — local needs GPU drivers, Python environment, model-weight maintenance. If you do not want to touch any of that, a cloud API is one HTTP call
5. **No GPU** — Parakeet/Whisper run on CPU but much slower (RTF near 1, near-real-time but not accelerated). aistack supports CPU mode but does not recommend it as the primary path
6. **Extremely low-latency streaming** (< 500ms end-to-end) — Parakeet does not currently support native streaming; aistack falls back. For genuine real-time captions (call centers / live broadcast), pick a streaming-trained model with dedicated optimization
7. **Single audio segment over ~90 minutes and wall-time-sensitive** — 90–100 min is the cliff on this machine; longer still runs but degrades to RTF 0.08–0.15+. For sustained long-audio throughput, chunk-preprocess or use a 24+ GB VRAM card

If your scenario hits any of the above, local may not be the better choice.

## When local wins

- **Bulk offline transcription** (video subtitle generation, podcast archiving, interview cleanup, legal evidence) — keep an aistack process resident locally and drop files in
- **Private-domain data** (internal company meetings, medical records, unpublished interviews) — data never leaves the machine
- **Price-sensitive high-volume use** (tens to hundreds of hours per month) — hardware cost amortizes once; electricity is essentially negligible
- **Self-controlled stack** (no vendor deprecation policy in the path; e.g., when OpenAI retires Whisper-1 your pipeline is unaffected)

## How to reproduce

1. Install aistack: `uv pip install -e .[asr-parakeet]` (per `pyproject.toml` extras)
2. Start the service: `scripts/dev.bat`
3. Hit the API:

```bash
curl -X POST http://127.0.0.1:11500/v1/audio/transcriptions \
  -F "file=@your-50min-audio.mp3" \
  -F "model=parakeet" \
  -F "language=en" \
  -F "response_format=verbose_json"
```

4. Read `/admin` or `/admin/api/metrics` for RTF / latency / resource usage.

Full configuration rationale in [parakeet-on-consumer-gpu](parakeet-on-consumer-gpu/) — including why those two `change_*` calls are needed, why `preserve_alignments` must not be touched, and the seven-layer documentation trace built from measurement.

## Open questions

- **Boundary on longer audio** (11 hours / 8 GB card) — NVIDIA's official position is that local attention + chunking on 8 GB can reach 11 hours; we have not measured that ceiling. Stable up to 50 minutes confirmed.
- **Lower edge of cheaper cards** (4 GB / 6 GB VRAM) — current baseline is 8 GB. Parakeet 0.6B itself is ~2.4 GB; with KV cache and workspace, 4 GB should not be enough; 6 GB is borderline. Measurement needed.
- **CPU-only viability** — not recommended as primary, but it runs. Specific RTF unmeasured.

If you have measured any of the above, PRs welcome — this folder exists for findings like that.

## Acknowledgments

All numbers in this note come from real aistack development. The 50-minute baseline tests were run by the project maintainer (@OldApeTalk) on an RTX 4060 Laptop, including controlled toggle experiments (the +20 GB shared-memory cost of `preserve_alignments` was discovered exactly this way). Without the measurement, this note would at most be documentation transcription.

---

*This is one of the dosmoon aistack project's research notes. See [the research index](./) for others.*
