# Observability — performance & availability analysis

aistack ships a built-in observability layer for **performance analysis**
(latency distributions, throughput, RTF) and **availability analysis**
(error rates, slot-busy rejections, disconnects). Three independent
toggles, all on by default except payload capture.

| Toggle | Default | What it does |
|---|---|---|
| `metrics` | on | In-process rolling histograms + counters per capability. Read via `/admin/api/metrics` and the `/admin` dashboard. Cost: <10 µs per request. |
| `access_log` | on | One JSONL line per request appended to `<LOG_DIR>/access-YYYY-MM-DD.jsonl`. Cost: enqueue + background flush. |
| `payload` | **off** | Per-request request body and response body persisted to disk for replay/diagnosis. Cost: disk IO; size+age bounded. |

Toggle at startup with env vars (see below) or live from the
`/admin` dashboard. Live toggles are session-only — restart returns to
the env-driven defaults.

## Env vars

| Var | Default | Notes |
|---|---|---|
| `AISTACK_OBS_METRICS` | `on` | `on` / `off` |
| `AISTACK_OBS_ACCESS_LOG` | `on` | `on` / `off` |
| `AISTACK_OBS_PAYLOAD` | `off` | `on` / `off` |
| `AISTACK_OBS_LOG_DIR` | `./logs` | JSONL output dir |
| `AISTACK_OBS_PAYLOAD_DIR` | `<HF_HOME>/../aistack_captures` (or `./captures`) | per-request capture root |
| `AISTACK_OBS_PAYLOAD_MAX_GB` | `5` | total disk cap, oldest evicted |
| `AISTACK_OBS_PAYLOAD_MAX_DAYS` | `7` | max age, older evicted |
| `AISTACK_OBS_PAYLOAD_RESP_MAX_MB` | `50` | per-request response body cap; over → meta records `resp_truncated:true` |
| `AISTACK_OBS_METRICS_WINDOW_MIN` | `60` | rolling window for percentiles |

## Request IDs

Every request is tagged with `X-Request-ID`:

* If the client sends the header, aistack passes it through (so
  upstream callers like VideoCraft can stitch their own trace IDs).
* Otherwise aistack generates a 16-char hex id.
* The id is echoed back in the response header, written to access log,
  embedded in payload directory names, and attached to metrics samples.

## /admin/api/metrics — JSON shape

```json
{
  "uptime_sec": 3601.2,
  "window_sec": 3600,
  "categories": {
    "asr": {
      "total": 142,
      "by_class": {"2xx": 138, "4xx": 1, "5xx": 0,
                   "503-busy": 3, "client-disconnect": 0},
      "error_count": 1,
      "error_rate": 0.007,
      "slot_503": 3,
      "disconnected": 0,
      "throughput_per_min": 2.3,
      "latency_ms": {
        "p50": 480.0, "p95": 2100.0, "p99": 9800.0, "max": 12030.0,
        "samples": 142,
        "histogram": {"<=10":0,"<=25":0,"<=50":0,"<=100":12, ...}
      },
      "slot_wait_ms": {
        "p50": 0.0, "p95": 12.0, "p99": 340.0, "samples": 142
      },
      "recent": [
        {"ts": 1715000000.123, "request_id": "a1b2c3d4e5f6a7b8",
         "status": 200, "class": "2xx", "latency_ms": 480.4,
         "slot_wait_ms": 0.0,
         "extra": {"audio_sec": 12.5, "rtf": 0.04, "model": "..."}}
      ]
    }
  }
}
```

`status_class` taxonomy:

| Class | Meaning |
|---|---|
| `2xx` | success |
| `4xx` | client error (validation, unknown model) |
| `5xx` | server error (provider crash, upstream down) |
| `503-busy` | GPU slot rejection — load shedding, **not** counted as error |
| `client-disconnect` | client closed connection mid-request |

## Access log JSONL fields

```json
{"ts":"2026-05-08T10:23:45.123+00:00","request_id":"a1b2c3d4e5f6a7b8",
 "method":"POST","path":"/v1/audio/transcriptions","query":null,
 "status":200,"category":"asr","model":"iic/SenseVoiceSmall",
 "latency_ms":482.4,"slot_wait_ms":0.0,
 "client":"127.0.0.1:51234",
 "extra":{"audio_sec":12.5,"rtf":0.04,"language":"en",
          "provider":"sensevoice","stream":false,
          "response_format":"json","request_audio_bytes":234567,
          "detected_language":"en"}}
```

Files roll daily by UTC date. Writer is a single background thread —
overflow drops records (warned once) so disk wedging never blocks
inference.

## Payload capture layout

```
<PAYLOAD_DIR>/2026-05-08/a1b2c3d4e5f6a7b8/
    meta.json     route, headers (Authorization redacted), status, timing
    req.bin       request body (mp3 / wav / json — original bytes)
    resp.bin      response body (json / audio bytes; truncated if huge)
```

`meta.json` is the single source of truth for what those binary
files contain — it includes the `Content-Type` of both directions.

The sweeper runs at startup and every 30 min:

1. drop dirs older than `PAYLOAD_MAX_DAYS`
2. while total > `PAYLOAD_MAX_BYTES`, drop oldest

Sensitive request headers (`Authorization`, `Cookie`, `X-Api-Key`,
`Proxy-Authorization`) are replaced with `***` in `meta.json`. The
request body itself is **not** scrubbed — payload capture is opt-in
and intended for trusted-environment diagnostics only.

## Long-term analysis recipes

```bash
# slow ASR requests last 24h
jq 'select(.category=="asr" and .latency_ms > 1000)' logs/access-2026-05-08.jsonl

# error rate per model
jq -r 'select(.category=="llm") | [.model, .status] | @tsv' logs/*.jsonl \
    | sort | uniq -c

# mean RTF for SenseVoice over a day
jq -r 'select(.extra.provider=="sensevoice") | .extra.rtf' logs/access-2026-05-08.jsonl \
    | awk '{s+=$1;n++} END{print s/n}'

# replay a captured request
curl -X POST http://127.0.0.1:11500/v1/audio/transcriptions \
    -F file=@captures/2026-05-08/<rid>/req.bin \
    -F model=auto
```

## Performance overhead

Measured on a hot path with all toggles default (`metrics`+`access_log` on,
`payload` off):

* metrics: ~5 µs/req (one dict update + one deque append)
* access_log: ~10 µs/req (dict pickup + queue.put)
* request id middleware: ~3 µs/req

Total < 20 µs per request — invisible compared to ASR/LLM/TTS work
which is in the 100s of milliseconds.

With `payload` on, overhead is dominated by disk IO of the request
audio (often 100s of KB to several MB). Acceptable for diagnostics,
not recommended for sustained high-throughput.
