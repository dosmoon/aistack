"""Pluggable backends.

aistack abstracts ASR / TTS / LLM behind capability-shaped HTTP routes.
Concrete backend integrations live under this package — one module per
backend, grouped by capability subdirectory.

Today's lineup:
  backends.llm.ollama      Reverse-proxy + model-listing for a local
                           or remote Ollama daemon.

Future entries (not yet implemented) will follow the same shape:
  backends.llm.lan_remote  Forward to a sibling aistack on the LAN.
  backends.llm.cloud_*     Forward to a rented GPU instance running
                           an OpenAI-compatible LLM server.

ASR and TTS providers live under aistack/asr/ and the TTS Docker
proxy is in aistack/tts/ for historical reasons; they will migrate
into backends/asr/ and backends/tts/ in a later refactor (phase D7).
"""
