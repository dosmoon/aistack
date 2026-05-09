---
title: Configuration
description: Why aistack's configuration knobs default the way they do — and when you should deviate. Companion to the auto-generated env-var reference.
sidebar:
  order: 2
---

# aistack Configuration

aistack reads configuration from environment variables. There is no
config file and no `.env` support — `aistack/config.py` reads
`os.environ` directly. The admin UI may toggle a few session-level
switches at runtime but never persists them: restart returns to env
defaults.

This page covers the *why* — the design rationale behind each default,
the sweep results that produced the numbers, and when to deviate. For
the field-by-field list of every variable (name, type, default,
one-line effect), see the auto-generated
[Configuration Reference](./reference/configuration/), which is
re-rendered from `aistack/config.py` on every build.

## Where to set these

The launch script is the canonical place to set env variables. The
only launcher that ships with the project today is
`scripts/dev.bat` (Windows). It already contains commented
`set KEY=VALUE` templates for the most-tweaked variables — uncomment
the lines you need, or add new ones:

```bat
REM Inside scripts/dev.bat, before the `uvicorn` line:
set HF_HOME=D:\AI_Models\hf
set AISTACK_OBS_PAYLOAD=on
```

The `if "%KEY%"==""` guards in `dev.bat` mean that any variable
already exported in your shell wins over the script's default — useful
for one-off overrides (`set HF_HOME=E:\... && scripts\dev.bat`).

For other platforms or production deployments, you supply your own
launcher: a shell script that exports the variables before invoking
`python -m uvicorn aistack.main:app`, a systemd unit's `Environment=`
lines, a `docker run -e` flag, etc. aistack does not ship those today,
but the variable names below work identically wherever they are set.

## Conventions

- Boolean toggles accept `1 | 0` or `on | off`.
- Time values end in `_SEC` (seconds) or `_MIN` (minutes); size values
  end in `_MB` or `_GB`.
- Variables are read once at process start. Changing an env after the
  server is up requires a restart — the three observability toggles
  are the exceptions (see below).

## Model homes (third-party SDKs)

Tell each SDK where to find pre-downloaded weights. Without these,
Hugging Face / ModelScope / NeMo will pull GBs into the user profile
on first run.

- `HF_HOME` — faster-whisper / Parakeet / Qwen3-TTS weight cache.
- `MODELSCOPE_CACHE` — SenseVoice + FunASR VAD cache.
- `NEMO_CACHE_DIR` — Parakeet `.nemo` archive cache.

`scripts/dev.bat` points all three at `D:\AI_Models\<vendor>` — one
shared tree across every backend. These are upstream SDK conventions,
not aistack-defined env vars, so they don't appear in the
auto-generated reference table.

## Model lifecycle

aistack keeps each loaded model resident for a grace period after the
last request, then evicts it to free VRAM. The two relevant knobs are
`AISTACK_MODEL_KEEP_ALIVE_SEC` (default 300 s) and
`AISTACK_MODEL_SCAN_INTERVAL_SEC` (default 60 s).

A higher keep-alive is the right trade for "interactive bursts on the
same model" (pay the load cost once); a lower one is the right trade
for "rotating between several models on tight VRAM."

## Parakeet ASR

Two groups of knobs: **attention mode** (memory strategy) and
**chunking** (how long audio is split for inference).

### Attention mode — why `local` + `256,256`

- `local` attention is O(N) memory linear in audio length; `full` is
  O(N²) and OOMs on 8 GB cards past ~2-3 min. Cards with 12+ GB can
  set `full` for the small WER win, but for the consumer hardware
  profile aistack targets, `local` is the only viable mode for any
  audio over a few minutes.
- The `256,256` window is the sweet spot from a 128/256/512 sweep on
  25-min audio: 128 trades 1.2 pp recall to save 3 s wall time; 512
  buys ≤ 0.3 pp the other way. 256 is also what Parakeet's HF model
  card recommends, so the training-inference distribution matches.

### Chunking — why `window=720`, `overlap=120`

aistack splits anything longer than `WINDOW_SEC` into windows with
`OVERLAP_SEC` shared between adjacent chunks, runs each independently,
and stitches results via word-LCS in the overlap zone. This keeps
each pass inside the short-input VRAM regime instead of letting cuDNN
workspace + caching-allocator interactions push usage past 8 GB on
long audio.

Defaults come from a sweep on 25-min and 50-min real-world recordings:

- `overlap=60` produced an unexpected 14-min last chunk on 25-min
  audio (tail merge).
- `overlap=120` redistributes into three balanced chunks and gives
  the highest recall: 98.1% on 25-min, 95.5% on 50-min.
- `overlap=180` regresses on 50-min by inflating the last chunk to
  13.8 min and pushing reserved VRAM to 13 GB.

`AISTACK_PARAKEET_CHUNK_DISABLE=1` is the right knob on a 24+ GB
card where the whole-audio path no longer OOMs and you'd rather pay
linear cuDNN workspace cost than pay the 15-30 s LCS-stitch overhead
on each call.

## Backend upstreams

aistack proxies LLM and TTS to local servers. Override only when the
upstream lives on a different port or host:

- `AISTACK_OLLAMA_URL` — Ollama base URL for `/v1/chat/completions`.
- `AISTACK_QWEN3_TTS_UPSTREAM` — vLLM-Omni base URL for `/v1/audio/*`.

The defaults match the standard Ollama and Qwen3-TTS Docker bindings,
so a fresh `ollama serve` + `docker compose up -d` on the same host
needs no override.

## Observability

Three independent toggles for the metrics / access-log / payload
capture layer. **Each may also be flipped at runtime via the admin
UI; runtime flips do not persist past restart** — env vars are the
durable knobs.

- `AISTACK_OBS_METRICS=on` (default) — rolling histogram. ~5 µs per
  request; **always wanted**.
- `AISTACK_OBS_ACCESS_LOG=on` (default) — daily-rolling JSONL.
  ~10 µs per request; **almost always wanted**.
- `AISTACK_OBS_PAYLOAD=off` (default) — request and response bytes
  to disk. Disk-IO bound; **almost never wanted in steady state**.
  Turn on only when a specific bug needs the bytes.

Why three switches and not one master toggle: the three streams have
radically different cost profiles, so bundling them would force "all
or nothing" choices that fit no real workflow. The narrative side of
this is documented in the
[observability narrative](./api/observability/); the wire formats
live in the [admin reference](./api/reference/admin/) (auto-generated
from code).

Disk-budget knobs that interact:

- `AISTACK_OBS_PAYLOAD_MAX_GB=5` — total disk budget. Sweeper
  deletes oldest first when exceeded.
- `AISTACK_OBS_PAYLOAD_MAX_DAYS=7` — age budget; older trees are
  deleted first, before the size cut runs. Order matters: a
  1-day-old big request is preferred over a 6-day-old small one
  when budgeting.
- `AISTACK_OBS_PAYLOAD_RESP_MAX_MB=50` — per-response cap. Over-cap
  bodies save metadata only (the request body is always saved).

## Complete-defaults snapshot

Drop-in starting point for `scripts/dev.bat`. Every line below matches
the default already baked into the code, so deleting a line falls back
to the same behaviour. Keep what you change, delete the rest.

```bat
REM Model homes (point at your shared cache)
set HF_HOME=D:\AI_Models\hf
set MODELSCOPE_CACHE=D:\AI_Models\modelscope
set NEMO_CACHE_DIR=D:\AI_Models\nemo

REM Model lifecycle
set AISTACK_MODEL_KEEP_ALIVE_SEC=300
set AISTACK_MODEL_SCAN_INTERVAL_SEC=60

REM Parakeet — attention
set AISTACK_PARAKEET_ATTENTION_MODE=local
set AISTACK_PARAKEET_ATT_CONTEXT_SIZE=256,256

REM Parakeet — chunking
set AISTACK_PARAKEET_CHUNK_DISABLE=0
set AISTACK_PARAKEET_CHUNK_WINDOW_SEC=720
set AISTACK_PARAKEET_CHUNK_OVERLAP_SEC=120
set AISTACK_PARAKEET_CHUNK_MIN_LAST_SEC=300

REM Upstreams
set AISTACK_OLLAMA_URL=http://127.0.0.1:11434
set AISTACK_QWEN3_TTS_UPSTREAM=http://127.0.0.1:17860

REM Observability
set AISTACK_OBS_METRICS=on
set AISTACK_OBS_ACCESS_LOG=on
set AISTACK_OBS_PAYLOAD=off
set AISTACK_OBS_LOG_DIR=.\logs
set AISTACK_OBS_PAYLOAD_MAX_GB=5
set AISTACK_OBS_PAYLOAD_MAX_DAYS=7
set AISTACK_OBS_PAYLOAD_RESP_MAX_MB=50
set AISTACK_OBS_METRICS_WINDOW_MIN=60
```

For POSIX shells, swap `set KEY=VALUE` for `export KEY=VALUE` and
flip the path separators.
