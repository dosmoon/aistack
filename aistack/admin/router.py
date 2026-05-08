"""Admin /admin routes.

Layout:
    GET /admin                       full page (Jinja layout)
    GET /admin/fragments/models      models inventory fragment
    GET /admin/fragments/lock        GPU slot status fragment
    GET /admin/fragments/cache       model cache state fragment
    GET /admin/fragments/gpu         CUDA memory fragment
    GET /admin/fragments/logs        log tail fragment

The full page wires HTMX onto the fragment endpoints with
hx-trigger="every 2s" so the dashboard live-refreshes without any
client-side framework. If the network can't reach unpkg, the static
page still renders — only the auto-refresh degrades.

This router is mounted under prefix /admin in aistack/main.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from aistack import _gpu_lock, _model_cache
from aistack import asr as asr_pkg
from aistack.admin import log_buffer
from aistack.api._schemas import MetricsSnapshot
from aistack.backends.llm import ollama as llm_ollama
from aistack.observability import config as obs_config
from aistack.observability import metrics as obs_metrics
from aistack.observability import payload as obs_payload
from aistack.tts import qwen3 as tts_qwen3

router = APIRouter(prefix="/admin", tags=["admin"])

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _gpu_stats() -> dict[str, Any]:
    """Return CUDA memory snapshot, or a stub when torch / CUDA absent."""
    try:
        import torch  # type: ignore
    except ImportError:
        return {"available": False, "reason": "torch not installed"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA not available"}
    try:
        device = torch.cuda.current_device()
        name = torch.cuda.get_device_name(device)
        # mem_get_info returns (free_bytes, total_bytes) — reflects
        # what the entire GPU is using, including other processes
        # (e.g. the Qwen3-TTS Docker container on the same card).
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        return {
            "available": True,
            "device": name,
            "total_mb": total // (1024 * 1024),
            "free_mb": free // (1024 * 1024),
            "used_mb": (total - free) // (1024 * 1024),
            "allocated_mb": allocated // (1024 * 1024),
            "reserved_mb": reserved // (1024 * 1024),
        }
    except Exception as e:
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}


async def _models_inventory() -> list[dict]:
    """Reuse /v1/models logic but assemble the same way without the route."""
    data: list[dict] = list(asr_pkg.model_entries())
    if await tts_qwen3.is_healthy():
        data.append({
            "id": tts_qwen3.MODEL_ID,
            "object": "model",
            "owned_by": "qwen",
            "capabilities": ["tts"],
        })
    data.extend(await llm_ollama.model_entries())
    return data


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def admin_index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "admin.html", {})


@router.get("/fragments/models", response_class=HTMLResponse)
async def fragment_models(request: Request) -> HTMLResponse:
    models = await _models_inventory()
    return templates.TemplateResponse(request, "_models.html", {"models": models})


@router.get("/fragments/lock", response_class=HTMLResponse)
async def fragment_lock(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "_lock.html",
        {"busy": _gpu_lock.is_busy(), "holder": _gpu_lock.current_holder()},
    )


@router.get("/fragments/cache", response_class=HTMLResponse)
async def fragment_cache(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "_cache.html", {"cache": _model_cache.stats()},
    )


@router.get("/fragments/gpu", response_class=HTMLResponse)
async def fragment_gpu(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "_gpu.html", {"gpu": _gpu_stats()},
    )


@router.get("/fragments/logs", response_class=HTMLResponse)
async def fragment_logs(request: Request, n: int = 200) -> HTMLResponse:
    n = max(1, min(n, log_buffer.CAPACITY))
    return templates.TemplateResponse(
        request, "_logs.html", {"lines": log_buffer.tail(n)},
    )


@router.get("/fragments/metrics", response_class=HTMLResponse)
async def fragment_metrics(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "_metrics.html", {"snap": obs_metrics.snapshot()},
    )


def _observability_context() -> dict[str, Any]:
    return {
        "toggles": obs_config.snapshot(),
        "usage": obs_payload.usage(),
        "log_dir": str(obs_config.LOG_DIR),
    }


@router.get("/fragments/observability", response_class=HTMLResponse)
async def fragment_observability(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "_observability.html", _observability_context(),
    )


@router.post(
    "/api/observability/toggle",
    response_class=HTMLResponse,
    summary="Flip an observability toggle (live)",
)
async def api_observability_toggle(
    request: Request,
    key: str = Form(..., description="Which toggle to flip. One of 'metrics' / 'access_log' / 'payload'."),
    value: str = Form(..., description="Truthy ('1', 'on', 'true', 'yes', 'y') turns it on; anything else turns it off."),
) -> HTMLResponse:
    """Flip one observability toggle in process memory.

    Live-only — restart returns to env-var defaults
    (`AISTACK_OBS_METRICS_ENABLED`, `AISTACK_OBS_ACCESS_LOG_ENABLED`,
    `AISTACK_OBS_PAYLOAD_ENABLED`). Returns the re-rendered
    observability HTMX fragment so the admin dashboard swaps it in
    place without a separate fetch.
    """
    enabled = value.strip().lower() in {"1", "on", "true", "yes", "y"}
    try:
        obs_config.set_enabled(key, enabled)
    except ValueError as e:
        return HTMLResponse(f'<p class="empty">error: {e}</p>', status_code=400)
    return templates.TemplateResponse(
        request, "_observability.html", _observability_context(),
    )


@router.post(
    "/api/observability/clear-payload",
    response_class=HTMLResponse,
    summary="Delete every captured request/response payload",
)
async def api_observability_clear_payload(request: Request) -> HTMLResponse:
    """Wipe every payload-capture file under the configured payload
    directory.

    Useful between bench runs or when the on-disk usage approaches the
    configured `AISTACK_OBS_PAYLOAD_MAX_GB` cap and you want to reclaim
    immediately rather than wait for the size sweep. Returns the
    re-rendered observability HTMX fragment.
    """
    obs_payload.clear_all()
    return templates.TemplateResponse(
        request, "_observability.html", _observability_context(),
    )


@router.get(
    "/api/metrics",
    summary="Rolling-window metrics snapshot",
    response_model=MetricsSnapshot,
    response_model_exclude_none=True,
)
async def api_metrics() -> JSONResponse:
    """Machine-readable metrics snapshot — same data the admin
    dashboard's HTMX fragment renders.

    Categorised per capability (asr / llm / tts) over a rolling time
    window (default 60 minutes). Includes p50/p95/p99 latency, error
    rates, GPU-slot wait distribution, and a tail of the last 50
    samples per category for spot-checks. The full schema is the
    `MetricsSnapshot` Pydantic model in `aistack.api._schemas`.

    Restart loses the in-process samples; for cross-restart trend
    analysis use the JSONL access log under `AISTACK_OBS_LOG_DIR`.
    """
    return JSONResponse(obs_metrics.snapshot())


# Categories the ASR providers tag their cache entries with. Kept in sync
# with the `category=` arguments at sensevoice.py / parakeet.py /
# faster_whisper.py call sites — evicting both empties every ASR weight
# the cache currently holds without touching TTS/LLM peers.
_ASR_CACHE_CATEGORIES = ("asr-main", "asr-aux")


@router.post(
    "/api/reset-asr-state",
    response_model=None,
    summary="Drop loaded ASR weights from cache",
)
async def api_reset_asr_state(request: Request) -> HTMLResponse | JSONResponse:
    """Drop every ASR weight currently resident in the model cache.

    Useful between bench runs to free VRAM without restarting uvicorn.
    In-flight requests keep their own reference to the loaded model and
    finish normally; only the cache slot is released, so the next call
    triggers a cold load.

    TTS and LLM cache entries are untouched.

    Returns the re-rendered cache fragment when called from HTMX (so the
    admin UI swaps it in place), or JSON `{evicted, remaining}` otherwise
    (so scripts and bench runners can read the count directly).
    """
    evicted = sum(_model_cache.evict_category(c) for c in _ASR_CACHE_CATEGORIES)
    if request.headers.get("hx-request"):
        return templates.TemplateResponse(
            request, "_cache.html", {"cache": _model_cache.stats()},
        )
    return JSONResponse({"evicted": evicted, "remaining": _model_cache.stats()})
