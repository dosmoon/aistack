# `GET /v1/models`

Lists every model that is currently servable by aistack — across all
backends and all capabilities. Consumers use this to populate model
pickers and to decide whether a given capability is available before
sending a request.

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
      "id": "whisper-small",
      "object": "model",
      "owned_by": "openai",
      "capabilities": ["asr"]
    },
    {
      "id": "nvidia/parakeet-tdt-0.6b-v3",
      "object": "model",
      "owned_by": "nvidia",
      "capabilities": ["asr"]
    },
    {
      "id": "iic/SenseVoiceSmall",
      "object": "model",
      "owned_by": "alibaba",
      "capabilities": ["asr"]
    },
    {
      "id": "qwen3-tts-12hz-0.6b-customvoice",
      "object": "model",
      "owned_by": "qwen",
      "capabilities": ["tts"]
    }
  ]
}
```

When D6 lands, LLM models proxied from Ollama appear in the same list:

```json
{
  "id": "qwen3:4b",
  "object": "model",
  "owned_by": "ollama",
  "capabilities": ["llm"]
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
