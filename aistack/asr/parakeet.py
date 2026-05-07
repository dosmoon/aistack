"""Parakeet TDT local ASR provider (NVIDIA NeMo).

Runs Parakeet TDT 0.6B v3 (multilingual, 25 European languages) locally
via the NeMo toolkit. Tuned for the ASR router's "language=en/es/fr/..."
slot — Chinese should route to SenseVoice (separate provider, planned).

Output normalized to the same shape as faster_whisper / lemonfox:
    {language, duration, text, segments[], words[]}

Models download to NEMO_CACHE_DIR (set by core.paths.apply_cache_env at
process start — points to <models_dir>/nemo). NeMo expects 16 kHz mono
audio, so we transcode via ffmpeg into a temp WAV before inference.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from typing import Callable

from aistack import _model_cache
from aistack.errors import AIError, Kind

logger = logging.getLogger("aistack.asr.parakeet")


EventCallback = Callable[..., None]


# Attention-mode policy on consumer GPUs.
#
# Parakeet TDT v3 ships with full self-attention (rel_pos) by default, which
# is O(N^2) in audio length and treats audio under ~24 min on an A100 80GB
# as a single pass. On 8 GB consumer cards this OOMs the process around the
# 2-3 minute mark and on Windows the failure surfaces as a downstream temp-
# file race ("[WinError 32] manifest.json"), not a clean OOM.
#
# NeMo exposes `change_attention_model("rel_pos_local_attn", att_context_size)`
# which switches the encoder to a Longformer-style local attention pattern.
# Memory becomes O(N * context_size) — linear in audio length — at the cost
# of ~1-3% WER (the model loses access to global context). For an aistack
# gateway running on 8 GB hardware that is the correct trade-off: any audio
# length works, accuracy still beats faster-whisper-small on English.
#
# Power users can opt back into full attention via env var when running on
# bigger cards. att_context_size is also configurable; 256 frames on each
# side at the model's 80 ms-per-frame rate is ~20 s of context, which is
# what NVIDIA's HuggingFace model card recommends.
_ATTENTION_MODE = os.environ.get("AISTACK_PARAKEET_ATTENTION_MODE", "local").lower()
_ATT_CONTEXT_SIZE = os.environ.get("AISTACK_PARAKEET_ATT_CONTEXT_SIZE", "256,256")


# Parakeet TDT v3 supports these 25 languages (plus English) — used by the
# router's language_routing UI to know which langs are routable here.
SUPPORTED_LANGUAGES = (
    "en", "bg", "hr", "cs", "da", "nl", "et", "fi", "fr", "de",
    "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
    "sl", "es", "sv", "ru", "uk",
)


# Model caching is delegated to aistack._model_cache, which evicts entries
# idle longer than AISTACK_MODEL_KEEP_ALIVE_SEC (default 300s).

_PROVIDER_TAG = "parakeet"


def _get_model(model_name: str, emit: Callable):
    cached = _model_cache.get(_PROVIDER_TAG, model_name)
    if cached is not None:
        return cached
    emit("model_loading", model=model_name, device="auto", compute_type="auto")
    try:
        from nemo.collections.asr.models import ASRModel
    except ImportError as e:
        raise AIError(
            Kind.NETWORK, "Parakeet",
            "NeMo toolkit not installed. Run: pip install nemo_toolkit[asr]",
            raw=e,
        ) from e
    model = ASRModel.from_pretrained(model_name=model_name)
    _maybe_switch_to_local_attention(model)
    # Keep on whatever device NeMo picked (cuda if available, else cpu).
    model.eval()
    _model_cache.put(_PROVIDER_TAG, model_name, model, category="asr-main")
    device = _device_str(model)
    emit("model_loaded", model=model_name, device=device, compute_type="auto")
    return model


def _maybe_switch_to_local_attention(model) -> None:
    """Switch the encoder to local attention if AISTACK_PARAKEET_ATTENTION_MODE
    is "local" (the default).

    Best-effort: if the loaded model class lacks change_attention_model
    (older NeMo, or a checkpoint that wasn't built on FastConformer), we
    log a warning and keep going on whatever the model's default attention
    mode is. The request will still run; it just may OOM on long audio.
    """
    if _ATTENTION_MODE != "local":
        logger.info(
            "Parakeet attention mode = %r (env override); leaving full attention",
            _ATTENTION_MODE,
        )
        return
    try:
        l, r = (int(x.strip()) for x in _ATT_CONTEXT_SIZE.split(",", 1))
    except ValueError:
        logger.warning(
            "Invalid AISTACK_PARAKEET_ATT_CONTEXT_SIZE=%r; falling back to 256,256",
            _ATT_CONTEXT_SIZE,
        )
        l, r = 256, 256
    change = getattr(model, "change_attention_model", None)
    if not callable(change):
        logger.warning(
            "model.change_attention_model not available; staying on default "
            "full attention (long audio may OOM on 8 GB cards)"
        )
        return
    try:
        change(self_attention_model="rel_pos_local_attn", att_context_size=[l, r])
        logger.info(
            "Parakeet switched to local attention att_context_size=[%d,%d] "
            "(O(N) memory; ~1-3%% WER trade vs full attention)", l, r,
        )
    except Exception as e:
        logger.warning(
            "change_attention_model failed (%s: %s); staying on default attention",
            type(e).__name__, e,
        )


def _device_str(model) -> str:
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return "unknown"


# ── Audio preprocessing ──────────────────────────────────────────────────────

def _ensure_16k_mono_wav(src_path: str, tmp_dir: str) -> str:
    """Transcode arbitrary audio/video into 16 kHz mono PCM WAV via ffmpeg.
    Returns the temp WAV path. Caller owns cleanup of tmp_dir."""
    if not shutil.which("ffmpeg"):
        raise AIError(
            Kind.MALFORMED, "Parakeet",
            "ffmpeg not found on PATH — required for audio preprocessing",
        )
    out_path = os.path.join(tmp_dir, "audio_16k.wav")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src_path,
        "-ac", "1", "-ar", "16000",
        "-f", "wav", out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AIError(
            Kind.MALFORMED, "Parakeet",
            f"ffmpeg preprocessing failed: {proc.stderr.strip()[:300]}",
        )
    return out_path


def _audio_duration_sec(path: str) -> float:
    """Best-effort WAV duration via ffprobe. Returns 0.0 if unavailable."""
    if not shutil.which("ffprobe"):
        return 0.0
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            text=True, timeout=10,
        ).strip()
        return float(out) if out else 0.0
    except (subprocess.SubprocessError, ValueError):
        return 0.0


# ── Public API ───────────────────────────────────────────────────────────────

def transcribe(
    audio_path: str,
    *,
    model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
    language: str | None = None,
    translate: bool = False,
    on_event: EventCallback | None = None,
    on_segment: Callable[[dict], None] | None = None,
    cancel_token=None,
) -> dict:
    """Transcribe audio locally via NeMo Parakeet TDT.

    Args:
        audio_path:  Path to audio/video; transcoded to 16 kHz mono WAV
                     internally via ffmpeg.
        model_name:  HF model id. Default is the multilingual v3 checkpoint.
        language:    Hint kept in returned `language` field; Parakeet v3
                     itself auto-detects across its 25 supported languages.
                     None = auto.
        translate:   Not supported — Parakeet does ASR only. Raises if True.
        on_event:    Optional callback(event_type, **kwargs). Event types
                     mirror faster_whisper for UI consistency:
                       request_summary_local / model_loading / model_loaded
                       state_processing / state_done
        on_segment:  Accepted for signature parity with the other ASR
                     providers but never invoked — Parakeet's transcribe()
                     is one eager call returning all segments at once,
                     not an incremental generator. The streaming path
                     in api/asr.py handles Parakeet by emitting a
                     warning + single-delta SSE downgrade response
                     rather than driving on_segment from this layer.
        cancel_token: Cooperative cancel checked at coarse boundaries
                     (Parakeet's transcribe() is one blocking call — cannot
                     interrupt mid-inference).

    Returns:
        dict normalized to lemonfox verbose_json shape.

    Raises:
        AIError: model load / preprocessing / inference failure.
    """
    def emit(event_type: str, **kwargs):
        if on_event is None:
            return
        try:
            on_event(event_type, **kwargs)
        except Exception:
            pass

    if not os.path.exists(audio_path):
        raise AIError(Kind.MALFORMED, "Parakeet",
                      f"Audio file not found: {audio_path}")

    if translate:
        raise AIError(Kind.MALFORMED, "Parakeet",
                      "Parakeet does not support translation — disable the "
                      "translate flag or route to a Whisper provider")

    emit(
        "request_summary_local",
        filename=os.path.basename(audio_path),
        model=model_name,
        device="auto",
        compute_type="auto",
        language=language or "auto",
        translate="false",
    )

    if cancel_token is not None and cancel_token.cancelled:
        raise AIError(Kind.CANCELLED, "Parakeet", "Cancelled by user")

    try:
        model = _get_model(model_name, emit)
    except AIError:
        raise
    except Exception as e:
        raise AIError(Kind.NETWORK, "Parakeet",
                      f"Failed to load model {model_name!r}: {e}", raw=e) from e

    if cancel_token is not None and cancel_token.cancelled:
        raise AIError(Kind.CANCELLED, "Parakeet", "Cancelled by user")

    started = time.time()
    tmp_dir = tempfile.mkdtemp(prefix="parakeet_")
    try:
        wav_path = _ensure_16k_mono_wav(audio_path, tmp_dir)
        duration = _audio_duration_sec(wav_path)

        try:
            # NeMo 2.x: transcribe() returns list of Hypothesis with .text
            # and .timestamp = {'word': [...], 'segment': [...], 'char': [...]}
            #
            # num_workers=0 disables PyTorch DataLoader subprocess spawning.
            # On Windows this prevents WinError 32 races where DataLoader
            # worker subprocesses try to read NeMo's internal temp manifest
            # before its writer has flushed/closed. NeMo's own example
            # examples/asr/transcribe_speech.py defaults num_workers to 0
            # for the same reason. batch_size=1 matches our request-at-a-
            # time gateway: there is no batching benefit when each transcribe
            # call services exactly one audio file.
            results = model.transcribe(
                [wav_path], timestamps=True, num_workers=0, batch_size=1,
            )
        except TypeError:
            # Older NeMo versions may not accept all of these kwargs. Drop
            # to a plain call so the request still succeeds — the Windows
            # temp-file race risk reappears but only some versions hit it.
            results = model.transcribe([wav_path])
        except Exception as e:
            raise AIError(Kind.UNKNOWN, "Parakeet",
                          f"Inference failed: {e}", raw=e) from e

        if cancel_token is not None and cancel_token.cancelled:
            raise AIError(Kind.CANCELLED, "Parakeet", "Cancelled by user")

        hyp = results[0] if results else None
        text, segments_out, words_out = _normalize_hypothesis(hyp)

        elapsed = int(time.time() - started)
        emit("state_done", segment_count=len(segments_out), elapsed=elapsed)

        return {
            "language": language or "auto",
            "duration": duration,
            "text":     text,
            "segments": segments_out,
            "words":    words_out,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _normalize_hypothesis(hyp) -> tuple[str, list, list]:
    """Convert a NeMo Hypothesis (or plain string) into our standard shape.

    NeMo's transcribe() return shape varies by version:
      - 2.x with timestamps=True: Hypothesis(text=..., timestamp={'segment':
        [{'segment','start','end','start_offset','end_offset'}, ...], 'word':
        [{'word','start','end',...}, ...]})
      - older: list[str]
    """
    if hyp is None:
        return "", [], []

    if isinstance(hyp, str):
        return hyp.strip(), [], []

    text = (getattr(hyp, "text", "") or "").strip()
    ts = getattr(hyp, "timestamp", None) or {}
    seg_ts = ts.get("segment") or []
    word_ts = ts.get("word") or []

    segments_out = []
    for idx, s in enumerate(seg_ts):
        segments_out.append({
            "id":    idx,
            "start": float(s.get("start", 0.0)),
            "end":   float(s.get("end", 0.0)),
            "text":  (s.get("segment") or s.get("text") or "").strip(),
        })

    words_out = []
    for w in word_ts:
        words_out.append({
            "start": float(w.get("start", 0.0)),
            "end":   float(w.get("end", 0.0)),
            "word":  (w.get("word") or "").strip(),
        })

    # Fallback: NeMo sometimes returns word-level timestamps but no
    # segment-level ones — observed on long-form audio under local
    # attention. Without segments, downstream SRT generation collapses
    # the entire transcription into one giant cue, which is unusable for
    # anything past ~10 s of speech. Synthesize segments from the word
    # timestamps so SRT / segment-by-segment consumers stay functional.
    if not segments_out and words_out:
        segments_out = _segments_from_words(words_out)
    # Last-resort fallback: text but no timestamps at all (some old
    # NeMo paths). Single-segment is wrong but better than dropping.
    if text and not segments_out:
        segments_out.append({"id": 0, "start": 0.0, "end": 0.0, "text": text})

    return text, segments_out, words_out


# ── Segment synthesis from word timestamps ───────────────────────────────────

# Tunables for the word→segment regrouping. These match conventional
# subtitle pacing (≤ ~5 s per cue, break at sentence boundaries) and
# are deliberately conservative — better to over-split than to glue
# unrelated sentences into a single cue.
_SEGMENT_MAX_DURATION_SEC = 5.0
_SEGMENT_MIN_DURATION_SEC = 0.4
_SEGMENT_SILENCE_GAP_SEC = 0.6
_SEGMENT_SENTENCE_ENDERS = (".", "!", "?", "。", "！", "？")


def _segments_from_words(words: list[dict]) -> list[dict]:
    """Group word-level timestamps into subtitle-friendly segments.

    Break rules (any one closes the current segment):
      - silence gap > _SEGMENT_SILENCE_GAP_SEC between successive words
      - current segment duration ≥ _SEGMENT_MAX_DURATION_SEC AND the
        last word ends with a sentence-final punctuation mark
      - duration would exceed 1.5× _SEGMENT_MAX_DURATION_SEC even
        without punctuation (long monotone speech)

    Words shorter than _SEGMENT_MIN_DURATION_SEC don't trigger a flush
    on their own — that prevents single-syllable utterances from
    creating tiny segments that flicker on screen.
    """
    if not words:
        return []

    segments: list[dict] = []
    cur_words: list[dict] = []
    cur_start: float = 0.0
    cur_end: float = 0.0

    def flush() -> None:
        if not cur_words:
            return
        text = " ".join(w["word"] for w in cur_words if w.get("word")).strip()
        segments.append({
            "id": len(segments),
            "start": cur_start,
            "end": cur_end,
            "text": text,
        })

    hard_max = _SEGMENT_MAX_DURATION_SEC * 1.5

    for w in words:
        ws = float(w.get("start", 0.0))
        we = float(w.get("end", ws))
        wt = (w.get("word") or "").strip()
        if not wt:
            continue

        if not cur_words:
            cur_start = ws
            cur_end = we
            cur_words = [w]
            continue

        gap = ws - cur_end
        cur_dur = cur_end - cur_start
        prev_text = (cur_words[-1].get("word") or "").rstrip()
        prev_ends_sentence = prev_text.endswith(_SEGMENT_SENTENCE_ENDERS)

        should_flush = (
            gap > _SEGMENT_SILENCE_GAP_SEC
            or (cur_dur >= _SEGMENT_MAX_DURATION_SEC and prev_ends_sentence)
            or cur_dur >= hard_max
        )
        if should_flush and cur_dur >= _SEGMENT_MIN_DURATION_SEC:
            flush()
            cur_words = [w]
            cur_start = ws
            cur_end = we
            continue

        cur_words.append(w)
        cur_end = we

    flush()
    return segments
