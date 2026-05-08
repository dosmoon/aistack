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
from aistack.api._schemas import HealthResponse, ModelsList
from aistack.backends.llm import ollama as llm_ollama
from aistack.observability import config as obs_config
from aistack.observability.middleware import ObservabilityMiddleware
from aistack.observability.request_id import RequestIdMiddleware
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

# Observability layer. ObservabilityMiddleware runs as pure ASGI so it
# can tee streaming response bodies for payload capture without buffering
# them. RequestIdMiddleware is added last so it sits on the outside and
# the request_id is already on request.state by the time observability
# starts timing. Toggles are independent — see aistack/observability/config.py.
app.add_middleware(ObservabilityMiddleware)
app.add_middleware(RequestIdMiddleware)

_obs_toggles = obs_config.snapshot()
_aistack_logger.info(
    "observability: metrics=%s access_log=%s payload=%s (payload_dir=%s log_dir=%s)",
    "on" if _obs_toggles["metrics"] else "off",
    "on" if _obs_toggles["access_log"] else "off",
    "on" if _obs_toggles["payload"] else "off",
    obs_config.PAYLOAD_DIR, obs_config.LOG_DIR,
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


@app.get(
    "/health",
    summary="Liveness probe",
    response_model=HealthResponse,
)
def health() -> dict:
    """Returns 200 with a small JSON body when the worker is ready.

    Connection refused or non-200 means aistack is down or still
    starting; consumers should surface "service unreachable" with a
    hint to start the dev server. This endpoint never blocks on
    backend health — it only confirms the FastAPI worker itself is
    alive.
    """
    return {"status": "ok", "version": __version__}


@app.get(
    "/v1/models",
    summary="List servable models",
    response_model=ModelsList,
    response_model_exclude_none=True,
    responses={
        200: {
            "description": (
                "Inventory of every model the gateway can serve right now, "
                "across ASR / TTS / LLM, plus the 'auto' routing alias when "
                "at least one ASR backend is reachable."
            ),
        },
    },
)
async def list_models() -> dict:
    """OpenAI-compatible model list. Lists only backends that will actually
    serve a request right now:

      - ASR entries are emitted for every provider whose ML library is
        importable in the running venv (pure import probe — no model
        weights are loaded).
      - The TTS entry is emitted only if the Qwen3-TTS upstream
        container responds to its `/health` probe.
      - LLM entries are aggregated from Ollama's `/api/tags` when the
        daemon is reachable; an unreachable Ollama silently contributes
        zero entries rather than failing the whole listing.

    Clients can read this once on startup to know which capabilities are
    available, build language-aware pickers, and skip a 503 round-trip
    on first call. The response shape is OpenAI-compatible plus the
    aistack extension fields `capabilities`, `languages`,
    `supports_streaming`, and `is_routing_alias` per entry — see the
    ModelEntry schema.
    """
    data: list[dict] = list(asr_pkg.model_entries())
    if await tts_qwen3.is_healthy():
        data.append(
            {
                "id": tts_qwen3.MODEL_ID,
                "object": "model",
                "owned_by": "qwen",
                "capabilities": ["tts"],
                # Qwen3-TTS upstream returns chunked audio; our proxy
                # forwards the chunks verbatim, so streaming is native.
                "supports_streaming": True,
            }
        )
    data.extend(await llm_ollama.model_entries())
    return {"object": "list", "data": data}
