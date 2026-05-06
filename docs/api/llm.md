# `POST /v1/chat/completions`

OpenAI-compatible chat completion endpoint. aistack reverse-proxies the
request to a local Ollama daemon and adds two pieces of value over a
direct call:

1. **GPU scheduling** — before forwarding, aistack evicts its own ASR
   `asr-main` cache entries so the LLM model can fit in VRAM without
   contending with a hot Whisper / Parakeet / SenseVoice.
2. **Sensible `keep_alive` default** — when the request omits
   `keep_alive`, aistack injects `"30s"` so Ollama releases the model
   shortly after each completion. Sequential LLM calls within 30s
   reuse the loaded model; idle releases VRAM back for ASR.

Clients written against OpenAI's `/v1/chat/completions` work unchanged
— aistack mirrors the OpenAI request and response schemas verbatim
through Ollama's own OpenAI-compat layer.

## Request

`Content-Type: application/json`

OpenAI-spec required fields:

| Field | Type | Description |
|---|---|---|
| `model` | string | Model id from `GET /v1/models` (filter by `capabilities=["llm"]`). Examples: `qwen3:4b`, `llama3.1:8b`. |
| `messages` | array | List of `{role, content}` objects. Supported roles: `system`, `user`, `assistant`. |

OpenAI-spec optional fields (forwarded as-is):

| Field | Type | Description |
|---|---|---|
| `stream` | bool | If `true`, response is sent as Server-Sent Events. |
| `temperature` | number | 0.0 (deterministic) to 2.0. |
| `top_p` | number | Nucleus sampling cutoff. |
| `max_tokens` | int | Cap on generated tokens. |
| `stop` | string \| array | Stop sequence(s). |
| `presence_penalty`, `frequency_penalty` | number | Token-frequency penalties. |
| `seed` | int | Reproducibility hint where the backend honors it. |
| `response_format` | object | e.g. `{"type": "json_object"}` for JSON-mode generation (Ollama-supported on selected models). |
| `tools`, `tool_choice` | array / object | Tool calling, where the backend supports it. |

aistack-injected behavior:

| Field | Default behavior |
|---|---|
| `keep_alive` | If absent, aistack sets `"30s"`. Pass an explicit value (`"5m"`, `"1h"`, `"-1"` for forever, `"0"` for unload now) to override. |

### Example — non-streaming

```bash
curl -X POST http://127.0.0.1:11500/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "qwen3:4b",
           "messages": [
             {"role": "user", "content": "Translate to Chinese: hello world"}
           ]
         }'
```

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1714998000,
  "model": "qwen3:4b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好，世界。"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 4,
    "total_tokens": 14
  }
}
```

### Example — streaming

```bash
curl -N -X POST http://127.0.0.1:11500/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "qwen3:4b",
           "stream": true,
           "messages": [
             {"role": "user", "content": "Count from 1 to 3."}
           ]
         }'
```

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{"role":"assistant","content":"1"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{"content":", "}}]}

data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{"content":"2"}}]}

data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{"content":", "}}]}

data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{"content":"3."}}]}

data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

aistack streams chunks through unchanged — first-token latency depends
on Ollama's prefill time plus the small proxy overhead (sub-millisecond
on localhost).

## Response

Identical to Ollama's OpenAI-compat layer (which already mirrors
OpenAI). Refer to OpenAI's chat completions documentation for the
detailed response object schema; aistack does not alter it.

## Resource scheduling

When this endpoint receives a request:

1. The handler calls `_model_cache.evict_category("asr-main")` — every
   currently-resident ASR main model (faster-whisper, Parakeet,
   SenseVoice) is dropped from aistack's cache, with `gc.collect()`
   and `torch.cuda.empty_cache()` to actually release VRAM.

2. The request is forwarded to Ollama. If `keep_alive` was not
   provided in the body, aistack sets it to `"30s"`. Ollama loads the
   target model (cold-start ~2-10s for a 4B model on consumer GPU)
   and runs inference.

3. Response is streamed (or buffered) back to the client.

This means: **a fresh ASR call right after an LLM call may pay a
cold-load latency** (the asr-main model was evicted to make room).
That trade-off is intentional — the alternative is OOM on tight VRAM.

For workflows that hammer the LLM many times in a row (e.g. batch
translation), the 30s keep_alive default keeps Ollama warm across
calls. For workflows that interleave ASR and LLM tightly (e.g. a
real-time multimodal agent), expect cold-load taxes either way on 8 GB
hardware; lowering Qwen3-TTS Docker's `gpu_memory_utilization` is the
principal lever to recover VRAM.

## Health and inventory

- `GET /v1/models` aggregates Ollama's installed models with
  `capabilities=["llm"]` and `owned_by="ollama"`. If Ollama is
  unreachable the LLM entries are silently omitted (no entry, no
  error) — the rest of `/v1/models` continues to serve.

- A direct `POST /v1/chat/completions` while Ollama is unreachable
  returns `503 network` with an actionable message:
  ```json
  {
    "error": {
      "kind": "network",
      "provider": "aistack",
      "message": "Ollama is not reachable at http://127.0.0.1:11434. Start it with: ollama serve"
    }
  }
  ```

## Concurrency

aistack's single-task GPU lock (`busy_or_503`) **does not gate**
chat-completion requests. The lock protects in-process ASR inference;
LLM work happens in Ollama's own process with its own scheduler,
and Ollama supports concurrent requests on a single loaded model.
Multiple chat completions in flight are handled by Ollama directly.

ASR and LLM may still contend at the GPU level if both load large
models at the same time — that is a hardware reality, not a contract
guarantee. The eviction step above mitigates the most common case
(ASR was just used, LLM call follows).

## Configuration

Server-side env var:

```
AISTACK_OLLAMA_URL    default http://127.0.0.1:11434
```

Set in `scripts/dev.bat` or your launcher when Ollama runs on a
non-default port or another host on the LAN.

## Error scenarios

All errors use the standard envelope from [errors.md](errors.md).

| HTTP | `kind` | Cause |
|---|---|---|
| 400 | `malformed` | Body is not valid JSON; missing required fields. |
| 503 | `network` | Ollama daemon is not reachable. |
| 502 | `unknown` | Ollama returned a non-200 we could not decode. |
| 504 | `network` | Read timeout (default 600s); Ollama is stuck. |
| 400 | (Ollama-formatted) | Ollama itself rejected the request (unknown model id, malformed messages). aistack passes these through verbatim. |

When Ollama returns its own error envelope (e.g. unknown model), aistack
forwards the body untouched — the client sees Ollama's error format
directly. This is intentional: aistack only repackages errors that
originate at the gateway boundary; backend-originated errors keep
their native shape so existing OpenAI-compatible clients see what
they expect.

## Stability

OpenAI-compatible request and response shapes within `/v1` follow
OpenAI's published spec. Streaming format follows OpenAI's chunk
schema. aistack-side behavior (eviction, keep_alive injection) is
documented above and stable within `/v1`; if it changes meaningfully
(e.g. a different default keep_alive), it will be a `/v1` additive
change with a release note, not a `/v2` break.
