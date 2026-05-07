"""FastAPI application entry point.

D2: TTS wired (proxy to Qwen3-TTS via vLLM-Omni).
D3: ASR wired (in-process faster-whisper / parakeet / sensevoice).
"""

import logging
import sys

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from aistack import __version__
from aistack import asr as asr_pkg
from aistack.admin import log_buffer as admin_log_buffer
from aistack.admin import router as admin_router
from aistack.api import asr as asr_api
from aistack.api import llm as llm_api
from aistack.api import tts as tts_api
from aistack.backends.llm import ollama as llm_ollama
from aistack.tts import qwen3 as tts_qwen3


# uvicorn's default log config only attaches handlers to its own loggers
# (uvicorn / uvicorn.access / uvicorn.error). Anything we emit under the
# "aistack.*" namespace would otherwise propagate up to a root logger that
# has no handlers and silently drop INFO-level messages. Wire up our own
# stdout handler so cache eviction, request_summary, etc. are visible
# alongside uvicorn's request log lines.
_aistack_logger = logging.getLogger("aistack")
if not _aistack_logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    _aistack_logger.addHandler(_h)
    _aistack_logger.setLevel(logging.INFO)
    _aistack_logger.propagate = False

# Mirror every aistack log line into an in-memory ring buffer so the
# /admin dashboard can tail them without reading the filesystem. Also
# capture uvicorn.access so request flow shows up — without it the
# dashboard log panel only fills on errors / cache evictions / cancels,
# which makes a working server look dead.
admin_log_buffer.install(_aistack_logger)
admin_log_buffer.install(logging.getLogger("uvicorn.access"))

app = FastAPI(
    title="aistack",
    version=__version__,
    description="Localhost AI service for ASR and TTS (OpenAI-API-compatible).",
)

app.include_router(asr_api.router)
app.include_router(tts_api.router)
app.include_router(llm_api.router)
app.include_router(admin_router.router)


@app.exception_handler(StarletteHTTPException)
async def _envelope_aware_http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Render HTTPException without wrapping envelope-shape detail.

    aistack's contract is `{"error": {"kind", "provider", "message"}}` for
    every non-2xx body. FastAPI's stock handler always wraps detail as
    `{"detail": <whatever>}`, which historically forced the slot-busy
    503 path to use a bare string and broke envelope consistency.

    Convention: when an HTTPException's `detail` is already a dict
    containing an `error` key, treat it as an already-formed envelope
    and emit it unwrapped. Otherwise fall back to FastAPI's stock
    `{"detail": ...}` shape so existing consumers of generic
    HTTPException behavior elsewhere (validation errors, 404s, etc.)
    are unaffected.
    """
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        return JSONResponse(
            status_code=exc.status_code,
            content=detail,
            headers=exc.headers,
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": detail},
        headers=exc.headers,
    )


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
      - LLM entries are aggregated from Ollama (`/api/tags`) when the
        daemon is reachable; an unreachable Ollama silently contributes
        nothing rather than failing the whole listing.

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
    data.extend(await llm_ollama.model_entries())
    return {"object": "list", "data": data}
