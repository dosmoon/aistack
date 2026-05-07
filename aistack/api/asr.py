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
import json
import logging
import os
import shutil
import tempfile
from typing import AsyncIterator, Tuple

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from aistack import _gpu_lock
from aistack.asr import _SUPPORTS_STREAMING, faster_whisper as _fw
from aistack.asr import installed_providers
from aistack.asr import parakeet as _pk
from aistack.asr import sensevoice as _sv
from aistack.errors import AIError, Kind, http_status_for

logger = logging.getLogger("aistack.asr")


class _CancelToken:
    """Cooperative cancel signal honoured by all three ASR backends.

    They check `.cancelled` between segments / VAD chunks (see
    aistack/asr/*.py). Setting it from the handler when the HTTP client
    disconnects lets a long transcription release the GPU slot promptly
    instead of running to completion on a dead connection.
    """
    __slots__ = ("cancelled",)

    def __init__(self) -> None:
        self.cancelled: bool = False


async def _watch_disconnect(request: Request, token: _CancelToken) -> None:
    """Poll for client disconnect; set token.cancelled when it happens."""
    try:
        while not token.cancelled:
            try:
                if await request.is_disconnected():
                    token.cancelled = True
                    logger.info("ASR client disconnected; cancelling in-flight transcription")
                    return
            except Exception:
                # is_disconnected() can raise during shutdown; treat as
                # still-connected and try again next tick.
                pass
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass


# Each module maps to its provider id for _SUPPORTS_STREAMING lookup.
_MODULE_TO_PROVIDER_ID = {
    _fw: "faster-whisper",
    _pk: "parakeet",
    _sv: "sensevoice",
}


def _sse_event(payload: dict) -> bytes:
    """Encode a dict as a single SSE 'data: ...\\n\\n' frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


async def _stream_transcribe(
    *,
    module,
    audio_path: str,
    language: str | None,
    translate: bool,
    kwargs: dict,
    request: Request,
    supports_streaming: bool,
    canonical_model_id: str,
    tmp_dir: str,
) -> AsyncIterator[bytes]:
    """SSE event stream for a transcription request.

    The slot is already held by the caller (route handler), so this
    generator only needs to release it at the end. Owns the temp-dir
    cleanup and the disconnect-watcher task lifetime so they survive
    until the stream is fully consumed (the route function returns
    early once the StreamingResponse object is constructed).

    Two paths:

      * `supports_streaming=True` — runs `module.transcribe` in a
        worker thread with an on_segment callback that pushes each
        decoded segment onto an asyncio.Queue. Yields one
        `transcript.text.delta` per segment, then `transcript.text.done`.

      * `supports_streaming=False` — runs `module.transcribe` blockingly,
        then emits a `warning` event explaining the downgrade, the full
        text as a single `transcript.text.delta`, then
        `transcript.text.done`. The warning is the first event so
        aware clients can detect the downgrade before consuming any
        delta.

    Errors are emitted as in-stream `error` events with the standard
    envelope shape (HTTP layer is already 200 SSE by the time we know
    something went wrong).
    """
    cancel_token = _CancelToken()
    watcher = asyncio.create_task(_watch_disconnect(request, cancel_token))

    try:
        if not supports_streaming:
            # Downgrade path — declare the limitation up front, then
            # serve the request as if it were non-streaming, emitting
            # the result as a single delta event.
            yield _sse_event({
                "type": "warning",
                "code": "streaming_not_supported",
                "model": canonical_model_id,
                "message": (
                    f"Model {canonical_model_id!r} does not support "
                    "streaming; returning full transcription as a single "
                    "delta event. See /v1/models for streaming-capable models."
                ),
            })
            try:
                result = await asyncio.to_thread(
                    module.transcribe,
                    audio_path,
                    language=language,
                    translate=translate,
                    cancel_token=cancel_token,
                    **kwargs,
                )
            except AIError as e:
                logger.warning("ASR provider error (downgrade path): %s", e)
                yield _sse_event({
                    "type": "error",
                    "error": e.to_envelope()["error"],
                })
                return
            except Exception as e:
                logger.exception("Unexpected ASR failure (downgrade path)")
                yield _sse_event({
                    "type": "error",
                    "error": {
                        "kind": "unknown",
                        "provider": "aistack",
                        "message": f"Internal error: {e}",
                    },
                })
                return

            yield _sse_event({
                "type": "transcript.text.delta",
                "delta": result.get("text", ""),
                "segment": {
                    "start": 0.0,
                    "end": float(result.get("duration", 0.0)),
                    "words": result.get("words", []),
                },
            })
            yield _sse_event({
                "type": "transcript.text.done",
                "language": result.get("language"),
                "duration": float(result.get("duration", 0.0)),
            })
            return

        # Real streaming path — bridge a worker-thread on_segment callback
        # into an asyncio.Queue, async-iterate the queue, yield SSE events.
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _on_segment(seg: dict) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, ("segment", seg))
            except RuntimeError:
                # Event loop is closed (request torn down). Nothing to do.
                pass

        async def _run_worker() -> None:
            try:
                result = await asyncio.to_thread(
                    module.transcribe,
                    audio_path,
                    language=language,
                    translate=translate,
                    on_segment=_on_segment,
                    cancel_token=cancel_token,
                    **kwargs,
                )
                await queue.put(("done", result))
            except AIError as e:
                await queue.put(("error", e))
            except Exception as e:
                logger.exception("Unexpected ASR failure (streaming path)")
                await queue.put(("error", AIError(
                    Kind.UNKNOWN, "aistack",
                    f"Internal error: {e}", raw=e,
                )))

        worker_task = asyncio.create_task(_run_worker())

        try:
            while True:
                event_type, payload = await queue.get()
                if event_type == "segment":
                    yield _sse_event({
                        "type": "transcript.text.delta",
                        "delta": payload.get("text", ""),
                        "segment": {
                            "start": float(payload.get("start", 0.0)),
                            "end": float(payload.get("end", 0.0)),
                            "words": payload.get("words", []),
                        },
                    })
                elif event_type == "done":
                    yield _sse_event({
                        "type": "transcript.text.done",
                        "language": payload.get("language"),
                        "duration": float(payload.get("duration", 0.0)),
                    })
                    break
                elif event_type == "error":
                    err: AIError = payload
                    logger.warning("ASR streaming error: %s", err)
                    yield _sse_event({
                        "type": "error",
                        "error": err.to_envelope()["error"],
                    })
                    break
        finally:
            # Wait for the worker thread to settle so we don't leak a
            # GPU operation past the response. Cancel-token will have
            # been set by the disconnect watcher if the client bailed.
            try:
                await worker_task
            except Exception:
                pass
    finally:
        watcher.cancel()
        try:
            await watcher
        except (asyncio.CancelledError, Exception):
            pass
        _gpu_lock.release()
        shutil.rmtree(tmp_dir, ignore_errors=True)

router = APIRouter(tags=["asr"])


# Whisper sizes that faster-whisper accepts as model_name directly.
_WHISPER_SIZES = {
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v1", "large-v2", "large-v3",
    "large-v3-turbo", "distil-large-v3",
}

# Languages that SenseVoice handles best (CJK + tones).
_CJK_LANGS = {"zh", "yue", "ja", "ko"}

# Languages that Parakeet TDT v3 supports (25 European + English).
_PARAKEET_LANGS = {
    "en", "bg", "hr", "cs", "da", "nl", "et", "fi", "fr", "de",
    "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
    "sl", "es", "sv", "ru", "uk",
}


def _select_for_auto(language: str | None) -> Tuple[object, dict]:
    """Pick the best installed ASR backend for the given language hint.

    Routing policy:
      * CJK / tonal language hint -> SenseVoice (if installed).
      * European-language hint covered by Parakeet TDT v3 -> Parakeet.
      * Anything else (or unhinted) -> faster-whisper-small as the
        general-purpose fallback.

    If the preferred backend is not installed, we degrade to the
    next-best installed backend rather than failing the request — the
    `auto` mode is meant to "just work" in any deployment.
    """
    installed = set(installed_providers())
    iso = (language or "").strip().lower()

    if iso in _CJK_LANGS and "sensevoice" in installed:
        return _sv, {"model_name": "iic/SenseVoiceSmall"}
    if iso in _PARAKEET_LANGS and "parakeet" in installed:
        return _pk, {"model_name": "nvidia/parakeet-tdt-0.6b-v3"}
    if "faster-whisper" in installed:
        return _fw, {"model_name": "small"}
    raise AIError(
        Kind.NETWORK, "aistack",
        "No ASR backend installed. Add at least one extra: "
        "uv pip install -e .[asr-fasterwhisper]",
    )


def _select_provider(model: str, language: str | None) -> Tuple[object, dict]:
    """Pick a provider module + transcribe-kwargs from the OpenAI `model` field.

    Special values:
      * "" (empty) or "auto" -> _select_for_auto(language).

    Otherwise the model field is matched against the explicit selectors
    below. Raises AIError(MALFORMED) on unknown models.
    """
    if not model or model.strip().lower() == "auto":
        return _select_for_auto(language)

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
        "Use 'auto', whisper-{size}, parakeet, or sensevoice.",
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
    request: Request,
    file: UploadFile = File(..., description="Audio file (any ffmpeg-readable format)."),
    model: str = Form("", description="Provider/model selector. Empty or 'auto' = pick best installed backend for the given language. Otherwise: whisper-{size} | parakeet | sensevoice."),
    language: str | None = Form(None, description="ISO 639-1 code (e.g. 'en', 'zh'). Omit for auto-detect."),
    response_format: str = Form("json", description="json | verbose_json | text"),
    translate: bool = Form(False, description="If true, transcribe to English instead of source language. Only Whisper-family models support translation."),
    stream: bool = Form(False, description="If true, return Server-Sent Events with one transcript.text.delta per decoded segment, ending with transcript.text.done. Models with supports_streaming=false in /v1/models still accept this and emit a warning event followed by a single delta. response_format is ignored when stream=true."),
):
    if response_format not in ("json", "verbose_json", "text"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format: {response_format!r}. "
                   "Use 'json', 'verbose_json', or 'text'.",
        )

    # Persist upload to a temp file — provider transcribe() takes a path
    # because all three backends prefer ffmpeg-readable files over streams.
    # When streaming, the temp dir is cleaned up by the streaming
    # generator's finally so it survives until the worker thread is done.
    tmp_dir = tempfile.mkdtemp(prefix="aistack_asr_")
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    audio_path = os.path.join(tmp_dir, f"upload{suffix}")
    cleanup_on_exit = True

    try:
        with open(audio_path, "wb") as fh:
            shutil.copyfileobj(file.file, fh)

        try:
            module, kwargs = _select_provider(model, language)
        except AIError as e:
            return JSONResponse(
                status_code=http_status_for(e.kind),
                content=e.to_envelope(),
            )

        if stream:
            # SSE path. Acquire the slot synchronously here so a busy
            # gateway returns 503 + envelope cleanly (rather than starting
            # a streaming response and then yielding an in-stream error).
            # The streaming generator owns release + temp-dir cleanup
            # from this point on, so we tell the outer finally to skip.
            _gpu_lock.try_acquire_or_503("asr")
            cleanup_on_exit = False
            provider_id = _MODULE_TO_PROVIDER_ID[module]
            supports = _SUPPORTS_STREAMING[provider_id]
            canonical_id = kwargs.get("model_name") or model or "auto"
            return StreamingResponse(
                _stream_transcribe(
                    module=module,
                    audio_path=audio_path,
                    language=language,
                    translate=translate,
                    kwargs=kwargs,
                    request=request,
                    supports_streaming=supports,
                    canonical_model_id=canonical_id,
                    tmp_dir=tmp_dir,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        # Single-task GPU policy: at most one inference at a time across
        # the entire gateway (ASR / LLM proxy / TTS proxy share one slot).
        # Concurrent requests get HTTP 503 + Retry-After. The blocking
        # transcribe() runs in a worker thread so the FastAPI event loop
        # stays responsive (e.g. /health stays live, and the disconnect
        # watcher below can set the cancel token).
        cancel_token = _CancelToken()
        watcher = asyncio.create_task(_watch_disconnect(request, cancel_token))
        try:
            with _gpu_lock.busy_or_503("asr"):
                try:
                    result = await asyncio.to_thread(
                        module.transcribe,
                        audio_path,
                        language=language,
                        translate=translate,
                        cancel_token=cancel_token,
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
        finally:
            watcher.cancel()
            try:
                await watcher
            except (asyncio.CancelledError, Exception):
                pass

        payload = _to_openai_response(result, response_format)
        if response_format == "text":
            return PlainTextResponse(content=payload)
        return payload
    finally:
        if cleanup_on_exit:
            shutil.rmtree(tmp_dir, ignore_errors=True)
