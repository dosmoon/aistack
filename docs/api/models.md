# `GET /v1/models`

Lists every model that is currently servable by aistack — across all
backends and all capabilities — plus any aistack-provided routing
aliases. Consumers use this to populate model pickers, filter by
language for ASR, and decide whether a given capability is available
before sending a request.

> For the integration journey (when to call this, how the result feeds
> into other endpoints, common pitfalls), read
> [`integration.md`](integration.md). This page is a field-by-field
> reference.

## Request

```bash
curl http://127.0.0.1:11500/v1/models
```

No parameters. The response reflects whatever the worker can serve
right now (which depends on which optional extras are installed and
whether external backends like the Qwen3-TTS Docker container are up).

## Response

```http
HTTP/1.1 200 OK
Content-Type: application/json
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "auto",
      "object": "model",
      "owned_by": "aistack",
      "capabilities": ["asr"],
      "is_routing_alias": true,
      "supports_streaming": false
    },
    {
      "id": "whisper-small",
      "object": "model",
      "owned_by": "openai",
      "capabilities": ["asr"],
      "languages": ["af", "am", "ar", "...", "yue", "zh"],
      "supports_streaming": true
    },
    {
      "id": "nvidia/parakeet-tdt-0.6b-v3",
      "object": "model",
      "owned_by": "nvidia",
      "capabilities": ["asr"],
      "languages": ["en", "bg", "hr", "...", "ru", "uk"],
      "supports_streaming": false
    },
    {
      "id": "iic/SenseVoiceSmall",
      "object": "model",
      "owned_by": "alibaba",
      "capabilities": ["asr"],
      "languages": ["zh", "yue", "en", "ja", "ko"],
      "supports_streaming": true
    },
    {
      "id": "qwen3-tts-12hz-0.6b-customvoice",
      "object": "model",
      "owned_by": "qwen",
      "capabilities": ["tts"],
      "supports_streaming": true
    },
    {
      "id": "qwen3:4b",
      "object": "model",
      "owned_by": "ollama",
      "capabilities": ["llm"],
      "supports_streaming": true
    }
  ]
}
```

## Field reference

### `id` (string)

Pass this verbatim as the `model` parameter on capability endpoints.
The id namespace mirrors what the underlying backend uses:

- Whisper sizes from faster-whisper: `whisper-tiny`, `whisper-small`,
  `whisper-medium`, `whisper-large-v3`, ...
- HuggingFace ids when the backend loads from HF: `nvidia/parakeet-tdt-0.6b-v3`,
  `iic/SenseVoiceSmall`.
- Ollama tags pass through as-is: `qwen3:4b`, `llama3.1:8b`.

### `object` (string)

Always `"model"`. OpenAI-spec required.

### `owned_by` (string)

Free-form attribution to the originator of the model weights. Not a
backend identifier. Examples: `"openai"` (Whisper), `"nvidia"`,
`"alibaba"`, `"qwen"`, `"ollama"`. Consumers may use this for display
grouping but should not branch dispatch on it.

### `capabilities` (array of string) — aistack extension

Indicates which task slots a model can serve. Allowed values:

- `"asr"` — usable as `model=` for `POST /v1/audio/transcriptions`
- `"tts"` — usable as `model=` for `POST /v1/audio/speech`
- `"llm"` — usable as `model=` for `POST /v1/chat/completions`

Most models have a single capability; the array shape leaves room for
future multi-modal entries.

This field is **not** in the OpenAI spec; OpenAI-only clients can
ignore it. aistack-aware clients should use it to filter the picker
shown to users (e.g. translate-task picker shows only `capabilities`
including `"llm"`).

### `languages` (array of string) — aistack extension, ASR only

ISO 639-1 codes for the languages the backend can transcribe. Present
on every real ASR entry. **Absent** on TTS / LLM entries and on
routing aliases.

Use this to filter the picker by user-requested language. For example,
if the user asks for Cantonese (`yue`), only `whisper-small` and
`iic/SenseVoiceSmall` are valid; `nvidia/parakeet-tdt-0.6b-v3` is
not. The list is the backend's full supported set, not a recommendation
— quality varies across the list (e.g. Whisper supports 99 languages
but is much stronger on top-resource ones).

Backend coverage as of `/v1`:

| Backend | Languages |
|---|---|
| `whisper-small` (faster-whisper / Whisper family) | All 99 ISO codes Whisper officially supports. |
| `nvidia/parakeet-tdt-0.6b-v3` | English plus 24 European languages (the model's published training set). |
| `iic/SenseVoiceSmall` | `zh`, `yue`, `en`, `ja`, `ko`. |

### `supports_streaming` (boolean) — aistack extension

True when the model serves the corresponding capability endpoint with
`stream=true` as a real incremental SSE stream; false when streaming
is not natively available for this model.

For ASR specifically, the gateway will still accept `stream=true`
on a model with `supports_streaming=false`, but the response is a
single-event SSE downgrade: a `warning` event explaining the
limitation followed by one `transcript.text.delta` event with the
full text and one `transcript.text.done` event. Aware clients should
filter pickers by this field rather than rely on the downgrade path.

The auto routing alias's value is the AND of the candidate pool —
True only when every installed real ASR backend supports streaming.
As of this contract version, Parakeet does not, so the alias's value
is False whenever Parakeet is installed.

For TTS and LLM entries the field is True when the upstream natively
streams the corresponding response (Qwen3-TTS via chunked transfer,
Ollama via SSE) and False otherwise. As of this contract version the
field is True for every TTS and LLM entry the gateway emits.

### `is_routing_alias` (boolean) — aistack extension

Marks an entry as a virtual id that aistack resolves internally rather
than a real backend model. The only routing alias currently defined is
`id="auto"` for ASR, which selects the best installed backend by the
request's `language` field (CJK → SenseVoice, European → Parakeet,
else → faster-whisper-small).

Routing aliases:

- Have `capabilities` so they can be filtered by task type.
- Do **not** have `languages` (the alias resolves to whichever installed
  backend best fits the request's language hint).
- Are valid `model` values on the corresponding capability endpoint —
  OpenAI-shape clients that ignore the flag still get correct routing.

aistack-aware clients can use the flag to render the alias prominently
(e.g. as the picker default) or to group it separately from real
models.

## Reachability semantics

The list is **dynamic**. A model only appears in the response if its
backend can actually serve a request right now:

| Backend | Visible in `/v1/models` only when ... |
|---|---|
| ASR providers (faster-whisper, Parakeet, SenseVoice) | the corresponding Python library is importable in the venv |
| TTS (Qwen3-TTS) | the Docker container responds to its own `/health` |
| LLM (Ollama, D6+) | aistack can reach `localhost:11434` (Ollama daemon up) |

If you start aistack but do not start the TTS Docker container, the
TTS entry is omitted from `/v1/models` and `POST /v1/audio/speech`
will return a `503 network` envelope. Consumers should treat the model
list as a capability inventory rather than a static catalog.

## When to call

- **At client startup**: cache the list, populate UI pickers.
- **When the user opens a "pick model" dialog**: refresh, in case the
  user just installed a new backend or started Ollama.
- **Not on every inference call**: there is no reason to re-fetch
  before each transcription.

The endpoint itself is cheap (a few import-probes + an HTTP head check
on the TTS container) but is not free; treat it like any model-listing
API.

## Stability

`id`, `object`, and `owned_by` are part of the OpenAI-spec contract
and stable within `/v1`. `capabilities` is an aistack extension; new
capability values may be added in `/v1` (additive), but existing
values never change meaning.

Whether a specific model id is present depends on installed backends
and is not a contract guarantee.
