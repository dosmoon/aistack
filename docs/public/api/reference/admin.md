---
title: Admin runtime controls
description: Auto-generated reference for the runtime control endpoints under /admin/api/.
sidebar:
  order: 14
---

<!-- AUTO-GENERATED: do not edit. Source: aistack/api/* docstrings + Pydantic models in aistack/api/_schemas.py, rendered by scripts/gen_api_reference.py. -->

## `POST /admin/api/observability/toggle`

**Flip an observability toggle (live)**

Flip one observability toggle in process memory.

Live-only — restart returns to env-var defaults
(`AISTACK_OBS_METRICS_ENABLED`, `AISTACK_OBS_ACCESS_LOG_ENABLED`,
`AISTACK_OBS_PAYLOAD_ENABLED`). Returns the re-rendered
observability HTMX fragment so the admin dashboard swaps it in
place without a separate fetch.

### Request body

**Content type:** `application/x-www-form-urlencoded`

Schema: [`Body_api_observability_toggle_admin_api_observability_toggle_post`](#schema-body_api_observability_toggle_admin_api_observability_toggle_post)

| Field | Type | Required | Description |
|---|---|---|---|
| `key` | string | yes | Which toggle to flip. One of 'metrics' / 'access_log' / 'payload'. |
| `value` | string | yes | Truthy ('1', 'on', 'true', 'yes', 'y') turns it on; anything else turns it off. |

### Responses

#### `200`

Successful Response

- `text/html` → string

#### `422`

Validation Error

- `application/json` → [`HTTPValidationError`](#schema-httpvalidationerror)

## `POST /admin/api/observability/clear-payload`

**Delete every captured request/response payload**

Wipe every payload-capture file under the configured payload
directory.

Useful between bench runs or when the on-disk usage approaches the
configured `AISTACK_OBS_PAYLOAD_MAX_GB` cap and you want to reclaim
immediately rather than wait for the size sweep. Returns the
re-rendered observability HTMX fragment.

### Responses

#### `200`

Successful Response

- `text/html` → string

## `GET /admin/api/metrics`

**Rolling-window metrics snapshot**

Machine-readable metrics snapshot — same data the admin
dashboard's HTMX fragment renders.

Categorised per capability (asr / llm / tts) over a rolling time
window (default 60 minutes). Includes p50/p95/p99 latency, error
rates, GPU-slot wait distribution, and a tail of the last 50
samples per category for spot-checks. The full schema is the
`MetricsSnapshot` Pydantic model in `aistack.api._schemas`.

Restart loses the in-process samples; for cross-restart trend
analysis use the JSONL access log under `AISTACK_OBS_LOG_DIR`.

### Responses

#### `200`

Successful Response

- `application/json` → [`MetricsSnapshot`](#schema-metricssnapshot)

## `POST /admin/api/reset-asr-state`

**Drop loaded ASR weights from cache**

Drop every ASR weight currently resident in the model cache.

Useful between bench runs to free VRAM without restarting uvicorn.
In-flight requests keep their own reference to the loaded model and
finish normally; only the cache slot is released, so the next call
triggers a cold load.

TTS and LLM cache entries are untouched.

Returns the re-rendered cache fragment when called from HTMX (so the
admin UI swaps it in place), or JSON `{evicted, remaining}` otherwise
(so scripts and bench runners can read the count directly).

### Responses

#### `200`

Successful Response

- `application/json`

---

## Schemas

### `MetricsCategorySnapshot` {#schema-metricscategorysnapshot}

Rolling-window metrics for one capability category (asr / llm / tts).

| Field | Type | Required | Description |
|---|---|---|---|
| `total` | integer | yes | Total requests counted in this category since process start (not windowed). |
| `by_class` | object | yes | Counts per status_class. Keys: 2xx / 4xx / 5xx / 503-busy / client-disconnect. |
| `error_count` | integer | yes | 4xx + 5xx count. Excludes 503-busy and client-disconnect. |
| `error_rate` | number | yes | error_count / total. 0.0 when total is 0. |
| `slot_503` | integer | yes | Count of 503 responses caused by GPU slot contention (load-shedding). |
| `disconnected` | integer | yes | Count of requests aborted by client disconnect mid-flight. |
| `throughput_per_min` | number | yes | Approximate requests-per-minute over the rolling window. |
| `latency_ms` | [`MetricsLatencyStats`](#schema-metricslatencystats) | yes |  |
| `slot_wait_ms` | [`MetricsSlotWaitStats`](#schema-metricsslotwaitstats) | yes |  |
| `recent` | array of [`MetricsRecentSample`](#schema-metricsrecentsample) | yes | Last ≤50 requests in this category, newest last. |

### `MetricsLatencyStats` {#schema-metricslatencystats}

Latency distribution for one capability category.

| Field | Type | Required | Description |
|---|---|---|---|
| `p50` | number | yes | 50th percentile (median) latency in milliseconds. |
| `p95` | number | yes | 95th percentile latency in milliseconds. |
| `p99` | number | yes | 99th percentile latency in milliseconds. |
| `max` | number | yes | Maximum observed latency in milliseconds within the rolling window. |
| `samples` | integer | yes | Number of samples in the rolling window the percentiles were computed from. |
| `histogram` | object | yes | Coarse histogram with power-of-2 buckets (`<=10`, `<=25`, ... `<=60000`, `>60000`). Counts samples per bucket. |

### `MetricsRecentSample` {#schema-metricsrecentsample}

One entry in the per-category rolling tail of recent requests.

Capped at the last 50 samples per category. Useful for spot-checks
of "what just happened" without sifting the full access log.

| Field | Type | Required | Description |
|---|---|---|---|
| `ts` | number | yes | Unix timestamp (seconds, with sub-second precision) of when the request completed. |
| `request_id` | string \| null | no | Short hex correlation id (matches the access log entry). |
| `status` | integer | yes | HTTP status code returned to the client. |
| `class` | enum (`'2xx'`, `'4xx'`, `'5xx'`, `'503-busy'`, `'client-disconnect'`) | yes | Status class. '503-busy' is load-shedding (slot mutex rejection — counted separately so a busy gateway doesn't look broken). 'client-disconnect' is informational, not an error. |
| `latency_ms` | number | yes | End-to-end request duration in milliseconds. |
| `slot_wait_ms` | number | yes | Time waiting for the GPU slot in milliseconds. |
| `extra` | object | yes | Per-request extras (model id, audio_sec, etc.). Free-form. |

### `MetricsSlotWaitStats` {#schema-metricsslotwaitstats}

GPU-slot wait distribution for one capability category.

| Field | Type | Required | Description |
|---|---|---|---|
| `p50` | number | yes | 50th percentile slot wait in milliseconds. |
| `p95` | number | yes | 95th percentile slot wait in milliseconds. |
| `p99` | number | yes | 99th percentile slot wait in milliseconds. |
| `samples` | integer | yes | Number of recorded slot waits in the rolling window. |

### `MetricsSnapshot` {#schema-metricssnapshot}

Response shape for `GET /admin/api/metrics`.

Built by `aistack.observability.metrics.snapshot()`. Stable across
`/v1` — adding new categories or new top-level keys is allowed,
renaming or removing requires a version bump.

| Field | Type | Required | Description |
|---|---|---|---|
| `uptime_sec` | number | yes | Process uptime in seconds. |
| `window_sec` | integer | yes | Rolling window duration the percentiles are computed over. |
| `categories` | object | yes | Per-capability metrics. Keys: 'asr', 'llm', 'tts' (only those that received traffic since startup). |
