"""FastAPI application entry point.

D2: TTS wired (proxy to Qwen3-TTS via vLLM-Omni).
D3: ASR wired (in-process faster-whisper / parakeet / sensevoice).
"""

from fastapi import FastAPI

from aistack import __version__
from aistack.api import asr as asr_api
from aistack.api import tts as tts_api
from aistack.tts import qwen3 as tts_qwen3

app = FastAPI(
    title="aistack",
    version=__version__,
    description="Localhost AI service for ASR and TTS (OpenAI-API-compatible).",
)

app.include_router(asr_api.router)
app.include_router(tts_api.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/v1/models")
async def list_models() -> dict:
    """OpenAI-compatible model list. Each model entry reflects an upstream
    backend that is currently reachable; unreachable backends are omitted
    so the client can detect missing services via empty / smaller listings.
    """
    data: list[dict] = []
    if await tts_qwen3.is_healthy():
        data.append(
            {
                "id": tts_qwen3.MODEL_ID,
                "object": "model",
                "owned_by": "qwen",
                "capabilities": ["tts"],
            }
        )
    return {"object": "list", "data": data}
