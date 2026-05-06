"""FastAPI application entry point.

Skeleton only — D1 phase. ASR and TTS endpoints will be wired in D2/D3
once their respective providers are migrated from VideoCraft.
"""

from fastapi import FastAPI

from aistack import __version__

app = FastAPI(
    title="aistack",
    version=__version__,
    description="Localhost AI service for ASR and TTS (OpenAI-API-compatible).",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/v1/models")
def list_models() -> dict:
    # OpenAI-compatible empty list. Will be populated once providers are loaded.
    return {"object": "list", "data": []}
