---
title: POST /v1/chat/completions
description: Chat completion endpoint. OpenAI compatible. Reverse-proxied to a local Ollama daemon with GPU scheduling and a sensible keep_alive default.
sidebar:
  order: 5
---

# `POST /v1/chat/completions`

OpenAI-compatible chat completion endpoint. aistack reverse-proxies
the request to a local Ollama daemon (default
`http://127.0.0.1:11434`) and adds two pieces of value over a direct
call.

The full request and response schema lives in OpenAI's authoritative
[Chat Completion API reference](https://platform.openai.com/docs/api-reference/chat).
aistack proxies that schema verbatim — Ollama's own OpenAI-compat
layer mirrors it, and aistack does not transform the body. The
[auto-generated LLM reference](./reference/llm/) lists what aistack's
own envelope around the proxy (response status codes, error shapes)
looks like.

This page covers the *why* — the two value-adds, GPU scheduling,
keep_alive policy, error attribution, and how concurrency works
across the gateway.

## What aistack adds over a direct Ollama call

1. **GPU scheduling.** Before forwarding, aistack evicts its own ASR
   `asr-main` cache entries — the in-process Whisper / Parakeet /
   SenseVoice model that may be sitting in VRAM gets dropped, with
   `gc.collect()` and `torch.cuda.empty_cache()` so the LLM model
   can fit. The handler then holds the global gateway GPU slot for
   the entire upstream call.

2. **Sensible `keep_alive` default.** When the request omits
   `keep_alive`, aistack injects `"30s"`. Sequential LLM calls
   within 30 s reuse the loaded model; idle releases VRAM back for
   ASR. Pass an explicit value (`"5m"`, `"1h"`, `"-1"` for forever,
   `"0"` for unload now) to override.

Clients written against OpenAI's `/v1/chat/completions` work
unchanged — aistack does not alter `messages`, `tools`,
`response_format`, or any other OpenAI field. It only fills in
`keep_alive` when missing.

## Resource scheduling explained

When this endpoint receives a request:

1. **Acquire global GPU slot.** Concurrent requests across ASR / TTS
   / LLM get HTTP 503 with `Retry-After`. The slot represents
   "GPU is doing inference" regardless of which process owns the
   kernels.
2. **Evict ASR.** `_model_cache.evict_category("asr-main")` drops
   every currently-resident ASR main model.
3. **Inject `keep_alive`.** If the request body omits it, set it to
   `"30s"`.
4. **Forward to Ollama.** Stream the response (when `stream=true`)
   chunk-by-chunk so first-token latency reaches the client as
   quickly as possible.
5. **Release the slot.** On normal completion, on stream end, on
   client disconnect, or on upstream error.

This means: **a fresh ASR call right after an LLM call may pay a
cold-load latency** (the asr-main model was evicted to make room).
That trade-off is intentional — the alternative is OOM on tight
VRAM. For workflows that hammer the LLM many times in a row
(e.g. batch translation), the 30 s keep_alive default keeps Ollama
warm across calls. For workflows that interleave ASR and LLM
tightly (e.g. a real-time multimodal agent), expect cold-load
taxes either way on 8 GB hardware; lowering Qwen3-TTS Docker's
`gpu_memory_utilization` is the principal lever to recover VRAM.

## Streaming

Every Ollama-served LLM advertises `supports_streaming: true` in
`/v1/models`. When `stream=true`, aistack forwards Ollama's SSE
chunks through unchanged — first-token latency is Ollama's prefill
time plus a sub-millisecond proxy overhead on localhost.

Client disconnect propagates: aistack closes the upstream connection
on disconnect so Ollama's runner can abort generation rather than
running to completion on a dead socket.

## Error attribution

aistack distinguishes errors that originate **at the gateway
boundary** from errors that originate **at Ollama itself**.

**Gateway-originated errors** use the standard
`{error: {kind, provider, message}}` envelope (see
[errors](./errors/)):

```json
{
  "error": {
    "kind": "network",
    "provider": "aistack",
    "message": "Ollama is not reachable at http://127.0.0.1:11434. Start it with: ollama serve"
  }
}
```

**Backend-originated errors** (Ollama rejected the request because
of an unknown model, malformed messages, etc.) are passed through
**verbatim**. The client sees Ollama's error format directly. This
is intentional: existing OpenAI-compatible clients have their own
expectations for what an Ollama / OpenAI error looks like, and
re-wrapping them would break that.

So: branch on `error.kind` when present (it is the aistack
envelope); fall back to OpenAI / Ollama error parsing for any
non-2xx response without a `kind` field.

## Health and inventory

- `GET /v1/models` aggregates Ollama's installed models with
  `capabilities=["llm"]` and `owned_by="ollama"`. If Ollama is
  unreachable the LLM entries are silently omitted (no entry, no
  error) — the rest of `/v1/models` continues to serve.
- A direct `POST /v1/chat/completions` while Ollama is unreachable
  returns `503 network` with the actionable message above.

## Configuration

Server-side env var:

```
AISTACK_OLLAMA_URL    default http://127.0.0.1:11434
```

Set in `scripts/dev.bat` or your launcher when Ollama runs on a
non-default port or another host on the LAN.

## Stability

OpenAI-compatible request and response shapes within `/v1` follow
OpenAI's published spec. Streaming format follows OpenAI's chunk
schema. aistack-side behaviour (eviction, `keep_alive` injection)
is documented above and stable within `/v1`; if it changes
meaningfully (e.g. a different default keep_alive), it will be a
`/v1` additive change with a release note, not a `/v2` break.
