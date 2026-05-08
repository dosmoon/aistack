---
title: Configuration
description: "Auto-generated reference for every AISTACK_* environment variable, with types, defaults, and effects. Source — aistack/config.py."
sidebar:
  order: 0
---

<!-- AUTO-GENERATED: do not edit. Source: aistack/config.py dataclasses,
rendered by scripts/gen_config_reference.py. -->

# Configuration reference

Every aistack environment variable, organised by feature domain. The
**Default** column shows the value applied when the variable is unset
(or set to a value the parser cannot decode). The **Effect** column
is the inline comment from the source dataclass field — concise by
design; for design rationale (why the default is what it is, when to
deviate), see [the configuration narrative](../../configuration/).

aistack reads configuration once at process start. Changing an env
variable after the worker is up requires a restart. The three
observability toggles (metrics / access_log / payload) are the only
exception — they can also be flipped live from the `/admin` dashboard,
but that is session-only and not persisted.


## Model lifecycle

Idle eviction policy for the global model cache.

| Env variable | Type | Default | Effect |
|---|---|---|---|
| `AISTACK_MODEL_KEEP_ALIVE_SEC` | float | `300.0` | idle seconds before a loaded model is unloaded |
| `AISTACK_MODEL_SCAN_INTERVAL_SEC` | float | `60.0` | how often the eviction sweeper runs |

Source: `aistack/config.py` → `ModelCacheConfig.from_env()` (line 76)

## Parakeet ASR

Attention mode + long-audio chunking knobs for Parakeet TDT.

| Env variable | Type | Default | Effect |
|---|---|---|---|
| `AISTACK_PARAKEET_ATTENTION_MODE` | string | `"local"` | 'local' = O(N) memory; 'full' = O(N²), OOMs on 8 GB cards past ~2-3 min |
| `AISTACK_PARAKEET_ATT_CONTEXT_SIZE` | string | `"256,256"` | local-attention left/right context, '<left>,<right>' frames at 80 ms each |
| `AISTACK_PARAKEET_CHUNK_DISABLE` | bool | `false` | set true to feed audio whole — useful on big GPUs that don't need chunking |
| `AISTACK_PARAKEET_CHUNK_WINDOW_SEC` | float | `720.0` | length of each chunk in seconds (default 720 = 12 min) |
| `AISTACK_PARAKEET_CHUNK_OVERLAP_SEC` | float | `120.0` | seconds shared between adjacent chunks (default 120 = 2 min) |
| `AISTACK_PARAKEET_CHUNK_MIN_LAST_SEC` | float | `300.0` | if the natural last chunk is shorter than this, merge it into the previous |

Source: `aistack/config.py` → `ParakeetConfig.from_env()` (line 92)

## Backend upstream URLs

Where local LLM / TTS daemons live.

| Env variable | Type | Default | Effect |
|---|---|---|---|
| `AISTACK_OLLAMA_URL` | string | `"http://127.0.0.1:11434"` | base URL of the local Ollama daemon for /v1/chat/completions |
| `AISTACK_QWEN3_TTS_UPSTREAM` | string | `"http://127.0.0.1:17860"` | base URL of the vLLM-Omni Qwen3-TTS container for /v1/audio/* |

Source: `aistack/config.py` → `BackendsConfig.from_env()` (line 116)

## Observability

Static observability config — paths and budgets.

Boot-time defaults for the three runtime-mutable toggles
(metrics / access_log / payload) live here too; the actual mutable
state is in `aistack.observability.config` because admin POST flips
it at runtime.

| Env variable | Type | Default | Effect |
|---|---|---|---|
| `AISTACK_OBS_METRICS` | bool | `true` | boot-time default for the metrics toggle (rolling histograms; ~5 µs/req) |
| `AISTACK_OBS_ACCESS_LOG` | bool | `true` | boot-time default for the access_log toggle (one JSONL line per request) |
| `AISTACK_OBS_PAYLOAD` | bool | `false` | boot-time default for the payload toggle (request/response bytes to disk; off by default — bytes are large + may be sensitive) |
| `AISTACK_OBS_PAYLOAD_MAX_GB` | float | `5.0` | total disk budget for payload capture (oldest-first sweep when exceeded) |
| `AISTACK_OBS_PAYLOAD_MAX_DAYS` | int | `7` | age budget for payload capture in days (older trees deleted) |
| `AISTACK_OBS_PAYLOAD_RESP_MAX_MB` | float | `50.0` | per-response cap for streaming bodies; over-cap responses save metadata only |
| `AISTACK_OBS_METRICS_WINDOW_MIN` | int | `60` | rolling window the metrics percentiles cover (in seconds) |

Source: `aistack/config.py` → `ObservabilityConfig.from_env()` (line 146)

## Path overrides (resolved by helpers)

These env vars are read inside helper functions in `aistack/config.py`, not directly inside a `from_env` classmethod, because their default behaviour cascades over multiple env vars. The **Default** column shows the literal default; the **Resolver** column points at the helper function that decides the cascade order.

| Env variable | Type | Default | Resolver |
|---|---|---|---|
| `AISTACK_OBS_PAYLOAD_DIR` | string | `<see _default_payload_dir>` | `_default_payload_dir` |
| `HF_HOME` | string | `<see _default_payload_dir>` | `_default_payload_dir` |
| `AISTACK_OBS_LOG_DIR` | string | `"./logs"` | `_default_log_dir` |
