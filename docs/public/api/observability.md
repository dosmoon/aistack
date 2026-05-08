---
title: Observability
description: Built-in performance and availability layer — rolling p50/p95/p99, JSONL access log, per-request payload capture, X-Request-ID correlation.
sidebar:
  order: 7
---

# Observability — performance & availability analysis

aistack ships a built-in observability layer for **performance
analysis** (latency distributions, throughput, RTF) and **availability
analysis** (error rates, slot-busy rejections, disconnects). Three
independent toggles, all on by default except payload capture.

The wire-format schemas (the metrics JSON, the access-log JSONL fields,
the payload capture layout) are owned by code and rendered into the
[admin reference](./reference/admin/) on every build. This page covers
the *why* — design rationale, when to use each toggle, the optional
`X-Request-ID` story for cross-system correlation, and analysis recipes.

## Three independent toggles

Why three switches and not one master toggle: the three streams have
**radically different cost profiles**, so bundling them would force
"all or nothing" choices that fit no real workflow.

- `metrics` — in-process rolling histograms + counters per capability.
  Cost: < 10 µs per request. **Always wanted.**
- `access_log` — one JSONL line per request appended to today's
  daily-rolling file under `AISTACK_OBS_LOG_DIR`. Cost: enqueue +
  background flush. **Almost always wanted.**
- `payload` — request body and response body persisted to disk for
  replay/diagnosis. Cost: disk IO; size + age bounded. **Almost
  never wanted in steady state**; turn on only when a specific bug
  needs the bytes.

Toggle at startup with env vars (see configuration in the repo
README) or live from the `/admin` dashboard. Live toggles are
session-only — restart returns to the env-driven defaults. The
toggles' **wire effect on consumers is zero** — every observability
feature is server-side and fully backward compatible. Existing
clients (CLI scripts, VideoCraft, OpenAI-shape SDKs) work unmodified
and start showing up in metrics / access logs immediately.

## Optional: send `X-Request-ID` for cross-system correlation

If the caller has its own job id (a VideoCraft pipeline step, an agent
task id, a user-facing request id), pass it as the `X-Request-ID`
header. aistack will:

1. Use it as the `request_id` field in the access log JSONL.
2. Use it as the directory name when payload capture is on
   (`<PAYLOAD_DIR>/<date>/<X-Request-ID>/`).
3. Attach it to in-memory metrics samples (visible under `recent` in
   `/admin/api/metrics`).
4. Echo it back in the response header.

```python
import httpx

trace_id = f"videocraft-{pipeline_id}-step-{step_idx}"
resp = httpx.post(
    "http://127.0.0.1:11500/v1/audio/transcriptions",
    headers={"X-Request-ID": trace_id},
    files={"file": open("clip.mp3", "rb")},
    data={"model": "auto", "language": "en"},
)
# Echoed back, even if you didn't send one (aistack generates 16-hex):
returned_id = resp.headers["X-Request-ID"]
```

**Format constraints** when sending your own id:

- ASCII letters, digits, and `-_:.` — anything else is rejected and
  aistack generates a 16-hex id instead.
- Max 128 chars — over-length values are also rejected.
- Uniqueness is the caller's responsibility (we do not deduplicate).

**Not sending one is fine.** aistack generates a 16-hex id (e.g.
`825e53134e178cd1`) and returns it in the response header. Clients
that just want to log "the id aistack assigned" only need to read
the response header — no request-side change.

### When stitching pays off

Skip it if:

- One process talks to aistack and never correlates across systems.
- Requests are inherently distinguishable by timestamp + endpoint.

Add it when:

- A long pipeline (transcribe → LLM → TTS) needs end-to-end
  debugging.
- Slow / failing requests need to be traced from VideoCraft's logs
  into aistack's access log + payload dir without timestamp
  guessing.
- Multiple clients share one aistack instance and you need to
  attribute traffic.

## Status class taxonomy (why `503-busy` is its own bucket)

The metrics counters split each capability's traffic into five
classes:

- `2xx` — success.
- `4xx` — client error (validation, unknown model).
- `5xx` — server error (provider crash, upstream down).
- `503-busy` — GPU slot rejection. **Load shedding is not an error.**
  A healthy gateway under load returns `503-busy` *because the system
  is doing its job*. Counting it as `5xx` would make a healthy gateway
  look broken whenever traffic spikes.
- `client-disconnect` — client closed connection mid-request. Also
  not an error — the request was abandoned by the caller.

`error_rate` in the metrics snapshot is `(4xx + 5xx) / total` —
load shedding and disconnects deliberately excluded.

## Payload scrubbing — what's safe, what's not

When `payload` is on, every request and response body persists to
disk. Two things to know:

- **Headers are scrubbed** in the per-request `meta.json`:
  `Authorization`, `Cookie`, `X-Api-Key`, `Proxy-Authorization`
  are replaced with `***`.
- **Bodies are NOT scrubbed.** Payload capture is opt-in and
  intended for trusted-environment diagnostics only. If your
  request body contains PII, do not turn this on in production.

The on-disk sweeper (runs at startup and every 30 min) drops:

1. Capture dirs older than `AISTACK_OBS_PAYLOAD_MAX_DAYS` (default 7).
2. Then while total > `AISTACK_OBS_PAYLOAD_MAX_GB` (default 5),
   the oldest survivors.

Order matters — the age cut runs before the size cut, so a
1-day-old big request is preferred over a 6-day-old small one when
budgeting disk.

## Long-term analysis recipes

```bash
# Slow ASR requests in the last 24 h
jq 'select(.category=="asr" and .latency_ms > 1000)' logs/access-2026-05-08.jsonl

# Error rate per LLM model
jq -r 'select(.category=="llm") | [.model, .status] | @tsv' logs/*.jsonl \
    | sort | uniq -c

# Mean RTF for SenseVoice over a day
jq -r 'select(.extra.provider=="sensevoice") | .extra.rtf' logs/access-2026-05-08.jsonl \
    | awk '{s+=$1;n++} END{print s/n}'

# Replay a captured request
curl -X POST http://127.0.0.1:11500/v1/audio/transcriptions \
    -F file=@captures/2026-05-08/<rid>/req.bin \
    -F model=auto
```

## Performance overhead

Measured on a hot path with all toggles default (`metrics` +
`access_log` on, `payload` off):

- metrics: ~5 µs/req (one dict update + one deque append).
- access_log: ~10 µs/req (dict pickup + queue.put).
- request-id middleware: ~3 µs/req.

Total < 20 µs per request — invisible compared to ASR / LLM / TTS
work which is in the 100s of milliseconds.

With `payload` on, overhead is dominated by disk IO of the request
audio (often 100s of KB to several MB). Acceptable for diagnostics,
not recommended for sustained high-throughput.

## Why no Prometheus exporter

aistack ships a JSON `/admin/api/metrics` endpoint, not OpenMetrics.
Two reasons: (1) aistack is a single-host gateway, so a scraper +
retention policy is overhead the rest of the architecture does not
ask for; (2) the JSON shape is already what scripts and dashboards
need. If a real Grafana ever ends up on the other end, adding a
Prometheus exporter is an additive `/v1` change.
