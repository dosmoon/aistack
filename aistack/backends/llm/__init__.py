"""LLM backends.

Each module wraps a single LLM endpoint that aistack proxies for, and
exposes:

  - `model_entries() -> list[dict]`    aggregate-able entries for /v1/models
  - `is_healthy() -> bool`             upstream readiness check
  - whatever the backend needs to forward / transform requests

Today only `ollama` is implemented; future backends (LAN peers, cloud
GPU rentals) will plug in here.
"""
