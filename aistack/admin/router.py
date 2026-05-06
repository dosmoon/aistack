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

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from aistack import _gpu_lock, _model_cache
from aistack import asr as asr_pkg
from aistack.admin import log_buffer
from aistack.backends.llm import ollama as llm_ollama
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
