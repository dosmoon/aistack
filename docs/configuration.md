# aistack Configuration Reference

aistack reads configuration from environment variables. There is no
config file — env keeps deployment topology (cache paths, GPU choice,
upstream URLs) in the launch script (`scripts/dev.bat`,
`docker run -e ...`, systemd units, etc.) where it belongs. The
admin UI may toggle a few session-level switches at runtime but never
persists them: restart returns to env defaults.

**Code is the source of truth.** Every env variable below is parsed
in [`aistack/config.py`](../aistack/config.py); each section below
points at the dataclass that owns it. Module-level aliases
(`_CHUNK_OVERLAP_SEC`, `UPSTREAM`, etc.) all read from this single
source — there is no other place an env is consumed.

## Conventions

- Boolean toggles accept `1 | 0` or `on | off`.
- Time values end in `_SEC` (seconds) or `_MIN` (minutes); size values
  end in `_MB` or `_GB`.
- Variables are read once at process start. Changing an env after the
  server is up requires a restart — exceptions are called out.
- "Effect" describes what changes if you set this. If you do not set
  it, the default behavior applies and you can ignore the row.

---

## Model homes (third-party SDKs)

Tell each SDK where to find pre-downloaded weights. Without these,
Hugging Face / ModelScope / NeMo will pull GBs into the user profile
on first run.

| Variable | Default | Effect |
|---|---|---|
| `HF_HOME` | _(per HF default)_ | faster-whisper / Parakeet / Qwen3-TTS weight cache. |
| `MODELSCOPE_CACHE` | _(per ModelScope default)_ | SenseVoice + FunASR VAD cache. |
| `NEMO_CACHE_DIR` | _(per NeMo default)_ | Parakeet `.nemo` archive cache. |

`scripts/dev.bat` points all three at `D:\AI_Models\<vendor>` — one
shared tree across every backend. → `scripts/dev.bat:22-30`.

---

## Model lifecycle

aistack keeps each loaded model resident for a grace period after the
last request, then evicts it to free VRAM. Both knobs are global —
they apply to faster-whisper, Parakeet, SenseVoice, vLLM-Omni alike.

| Variable | Default | Effect |
|---|---|---|
| `AISTACK_MODEL_KEEP_ALIVE_SEC` | `300` | Idle seconds before a loaded model is unloaded. |
| `AISTACK_MODEL_SCAN_INTERVAL_SEC` | `60` | How often the eviction sweeper runs. |

→ `aistack/config.py` `ModelCacheConfig`.

A higher keep-alive is the right trade for "interactive bursts on the
same model" (pay the load cost once); a lower one is the right trade
for "rotating between several models on tight VRAM."

---

## Parakeet ASR

Parakeet TDT 0.6B v3 is the default English / 25-EU-language ASR
backend. Tunables fall into two groups: **attention mode** (memory
strategy) and **chunking** (how long audio is split for inference).

### Attention mode

| Variable | Default | Effect |
|---|---|---|
| `AISTACK_PARAKEET_ATTENTION_MODE` | `local` | `local` = O(N) memory linear in audio length; `full` = O(N²) full self-attention, OOMs on 8 GB cards past ~2-3 min. |
| `AISTACK_PARAKEET_ATT_CONTEXT_SIZE` | `256,256` | Local-attention left/right context in 80 ms frames. `256,256` ≈ ±20 s window — NVIDIA's recommended value. |

→ `aistack/config.py` `ParakeetConfig` (attention fields).

256 is the sweet spot from a 128/256/512 sweep on 25-min audio:
128 trades 1.2 pp recall to save 3 s wall; 512 buys ≤ 0.3 pp at the
same cost going the other way. 256 is also what Parakeet's HF model
card recommends, so training-inference distributions match.

### Chunking (long audio)

aistack splits anything longer than `WINDOW_SEC` into windows with
`OVERLAP_SEC` shared between adjacent chunks, runs each independently,
and stitches results via word-LCS in the overlap zone. This keeps
each pass inside the short-input VRAM regime instead of letting
cuDNN workspace + caching-allocator interactions push usage past
8 GB on long audio.

| Variable | Default | Effect |
|---|---|---|
| `AISTACK_PARAKEET_CHUNK_DISABLE` | `0` | Set `1` to feed audio whole — useful on big GPUs that don't need chunking. |
| `AISTACK_PARAKEET_CHUNK_WINDOW_SEC` | `720` | Each chunk is this long (12 min). |
| `AISTACK_PARAKEET_CHUNK_OVERLAP_SEC` | `120` | Adjacent chunks share this many seconds (2 min). |
| `AISTACK_PARAKEET_CHUNK_MIN_LAST_SEC` | `300` | If the natural last chunk is shorter than this, merge it into the previous one. Worst-case last chunk = stride + min_last - eps ≈ 15 min. |

→ `aistack/config.py` `ParakeetConfig` (chunk fields).

Defaults come from a sweep on 25-min and 50-min real-world recordings:
`overlap=60` produced an unexpected 14-min last chunk on 25-min audio
(tail merge); `overlap=120` redistributes into three balanced chunks
and gives the highest recall (98.1 % on 25-min, 95.5 % on 50-min);
`overlap=180` regresses on 50-min by inflating the last chunk to 13.8
min and pushing reserved VRAM to 13 GB.

---

## Backend upstreams

aistack proxies LLM and TTS to local servers. Override only when the
upstream lives on a different port or host.

| Variable | Default | Effect |
|---|---|---|
| `AISTACK_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama base URL for `/v1/chat/completions`. |
| `AISTACK_QWEN3_TTS_UPSTREAM` | `http://127.0.0.1:17860` | vLLM-Omni base URL for `/v1/audio/speech`. |

→ `aistack/config.py` `BackendsConfig`.

---

## Observability (D5)

Three independent toggles for the metrics / access-log / payload
capture layer. Each may also be flipped at runtime via the admin UI;
runtime flips do not persist past restart.

| Variable | Default | Effect |
|---|---|---|
| `AISTACK_OBS_METRICS` | `on` | Rolling histogram of latency / 503 rate / error rate per category. < 1 µs per request. |
| `AISTACK_OBS_ACCESS_LOG` | `on` | Daily-rolling JSONL line per request to `<LOG_DIR>/access-YYYY-MM-DD.jsonl`. |
| `AISTACK_OBS_PAYLOAD` | `off` | Captures request + response bytes to disk. Off by default because audio bytes are large and may be sensitive. |
| `AISTACK_OBS_LOG_DIR` | `./logs` | Where access-log JSONL is written. |
| `AISTACK_OBS_PAYLOAD_DIR` | `<HF_HOME>/../aistack_captures` _or_ `./captures` | Where payload bytes are written. |
| `AISTACK_OBS_PAYLOAD_MAX_GB` | `5` | Total disk budget. Sweeper deletes oldest first when exceeded. |
| `AISTACK_OBS_PAYLOAD_MAX_DAYS` | `7` | Age budget — anything older is deleted. |
| `AISTACK_OBS_PAYLOAD_RESP_MAX_MB` | `50` | Per-response cap for streaming TTS / LLM; over-cap responses save metadata only. |
| `AISTACK_OBS_METRICS_WINDOW_MIN` | `60` | Rolling window the histogram covers. |

→ `aistack/config.py` `ObservabilityConfig` (static fields);
runtime-mutable toggle dict in `aistack/observability/config.py`.

Wire formats (JSONL fields, payload directory layout, metrics JSON
schema) are documented in `docs/api/observability.md`.

---

## Complete-defaults snapshot

Copy-paste starting point; everything below matches the defaults
already baked into the code, so removing a line just falls back to
the same behavior. Keep what you change, delete the rest.

```sh
# Model homes (point at your shared cache)
HF_HOME=D:\AI_Models\hf
MODELSCOPE_CACHE=D:\AI_Models\modelscope
NEMO_CACHE_DIR=D:\AI_Models\nemo

# Model lifecycle
AISTACK_MODEL_KEEP_ALIVE_SEC=300
AISTACK_MODEL_SCAN_INTERVAL_SEC=60

# Parakeet — attention
AISTACK_PARAKEET_ATTENTION_MODE=local
AISTACK_PARAKEET_ATT_CONTEXT_SIZE=256,256

# Parakeet — chunking
AISTACK_PARAKEET_CHUNK_DISABLE=0
AISTACK_PARAKEET_CHUNK_WINDOW_SEC=720
AISTACK_PARAKEET_CHUNK_OVERLAP_SEC=120
AISTACK_PARAKEET_CHUNK_MIN_LAST_SEC=300

# Upstreams
AISTACK_OLLAMA_URL=http://127.0.0.1:11434
AISTACK_QWEN3_TTS_UPSTREAM=http://127.0.0.1:17860

# Observability
AISTACK_OBS_METRICS=on
AISTACK_OBS_ACCESS_LOG=on
AISTACK_OBS_PAYLOAD=off
AISTACK_OBS_LOG_DIR=./logs
AISTACK_OBS_PAYLOAD_MAX_GB=5
AISTACK_OBS_PAYLOAD_MAX_DAYS=7
AISTACK_OBS_PAYLOAD_RESP_MAX_MB=50
AISTACK_OBS_METRICS_WINDOW_MIN=60
```
