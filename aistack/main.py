"""FastAPI application entry point.

D2: TTS wired (proxy to Qwen3-TTS via vLLM-Omni).
D3: ASR wired (in-process faster-whisper / parakeet / sensevoice).
"""

from fastapi import FastAPI

from aistack import __version__
from aistack import asr as asr_pkg
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
    """OpenAI-compatible model list. Lists only backends that will actually
    serve a request right now:

      - ASR entries are emitted for every provider whose ML library is
        importable (pure import probe, no model load).
      - TTS entry is emitted only if the Qwen3-TTS upstream container
        responds to /health.

    Clients can read this once on startup to know which capabilities are
    available and skip a 503 round-trip.
    """
    data: list[dict] = list(asr_pkg.model_entries())
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
