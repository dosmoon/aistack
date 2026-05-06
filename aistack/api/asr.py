"""ASR HTTP routes.

Exposes OpenAI-compatible POST /v1/audio/transcriptions. The `model` field
in the multipart form selects the provider:

    whisper-* / tiny / base / small / medium / large-v3 / large-v3-turbo
        -> faster-whisper (CTranslate2)
    parakeet*
        -> NVIDIA NeMo Parakeet TDT 0.6B v3
    sensevoice*
        -> Alibaba FunASR SenseVoice Small

aistack saves the uploaded audio to a temp file, calls the provider's
in-process transcribe() function, and returns the result. Provider modules
do lazy ML-library imports so this route loads even when none of
faster-whisper / nemo_toolkit / funasr are installed; users will see a
503 with installation instructions when they first hit a provider that
requires a missing library.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from typing import Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from aistack import _gpu_lock
from aistack.asr import faster_whisper as _fw
from aistack.asr import parakeet as _pk
from aistack.asr import sensevoice as _sv
from aistack.errors import AIError, Kind, http_status_for

logger = logging.getLogger("aistack.asr")

router = APIRouter(tags=["asr"])


# Whisper sizes that faster-whisper accepts as model_name directly.
_WHISPER_SIZES = {
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v1", "large-v2", "large-v3",
    "large-v3-turbo", "distil-large-v3",
}


def _select_provider(model: str) -> Tuple[object, dict]:
    """Pick a provider module + transcribe-kwargs from the OpenAI `model` field.

    Returns (module, kwargs). Raises AIError(MALFORMED) on unknown models.
    """
    if not model:
        raise AIError(Kind.MALFORMED, "aistack", "Missing required field: model")

    m = model.strip().lower()

    # OpenAI's only public whisper id; map to small for a reasonable default.
    if m == "whisper-1":
        return _fw, {"model_name": "small"}

    # Explicit whisper-{size}
    if m.startswith("whisper-"):
        size = m.removeprefix("whisper-")
        if size in _WHISPER_SIZES:
            return _fw, {"model_name": size}
        raise AIError(
            Kind.MALFORMED, "aistack",
            f"Unknown Whisper size: {size!r}. Supported: {sorted(_WHISPER_SIZES)}",
        )

    # Bare size, e.g. "small", "large-v3-turbo"
    if m in _WHISPER_SIZES:
        return _fw, {"model_name": m}

    if "parakeet" in m:
        # Accept either the bare alias or the full HF id.
        canonical = "nvidia/parakeet-tdt-0.6b-v3"
        return _pk, {"model_name": canonical}

    if "sensevoice" in m:
        canonical = "iic/SenseVoiceSmall"
        return _sv, {"model_name": canonical}

    raise AIError(
        Kind.MALFORMED, "aistack",
        f"Unknown model: {model!r}. "
        "Use whisper-{size}, parakeet, or sensevoice.",
    )


def _to_openai_response(result: dict, response_format: str) -> dict | str:
    """Reshape provider output (Lemonfox verbose_json shape) into the format
    requested by the client.
    """
    if response_format == "verbose_json":
        return result
    if response_format == "text":
        return result.get("text", "")
    # Default OpenAI "json" — minimal shape.
    return {"text": result.get("text", "")}


@router.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(..., description="Audio file (any ffmpeg-readable format)."),
    model: str = Form(..., description="Provider/model selector. See module docstring."),
    language: str | None = Form(None, description="ISO 639-1 code, or omit for auto-detect."),
    response_format: str = Form("json", description="json | verbose_json | text"),
    translate: bool = Form(False, description="If true, transcribe to English instead of source language. Only Whisper-family models support translation."),
):
    if response_format not in ("json", "verbose_json", "text"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format: {response_format!r}. "
                   "Use 'json', 'verbose_json', or 'text'.",
        )

    # Persist upload to a temp file — provider transcribe() takes a path
    # because all three backends prefer ffmpeg-readable files over streams.
    tmp_dir = tempfile.mkdtemp(prefix="aistack_asr_")
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    audio_path = os.path.join(tmp_dir, f"upload{suffix}")
    try:
        with open(audio_path, "wb") as fh:
            shutil.copyfileobj(file.file, fh)

        try:
            module, kwargs = _select_provider(model)
        except AIError as e:
            return JSONResponse(
                status_code=http_status_for(e.kind),
                content=e.to_envelope(),
            )

        # Single-task GPU policy: at most one ASR inference at a time on
        # this aistack worker. Concurrent requests get HTTP 503 + Retry-
        # After. The blocking transcribe() runs in a worker thread so the
        # FastAPI event loop stays responsive (e.g. /health stays live).
        try:
            with _gpu_lock.busy_or_503("asr"):
                try:
                    result = await asyncio.to_thread(
                        module.transcribe,
                        audio_path,
                        language=language,
                        translate=translate,
                        **kwargs,
                    )
                except AIError as e:
                    logger.warning("ASR provider error: %s", e)
                    return JSONResponse(
                        status_code=http_status_for(e.kind),
                        content=e.to_envelope(),
                    )
                except Exception as e:
                    logger.exception("Unexpected ASR failure")
                    err = AIError(
                        Kind.UNKNOWN, "aistack",
                        f"Internal error: {e}", raw=e,
                    )
                    return JSONResponse(
                        status_code=http_status_for(err.kind),
                        content=err.to_envelope(),
                    )
        except HTTPException:
            # 503 from busy_or_503 — let FastAPI render it; we still
            # need to clean up the temp dir, which the outer finally does.
            raise

        payload = _to_openai_response(result, response_format)
        if response_format == "text":
            return PlainTextResponse(content=payload)
        return payload
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
