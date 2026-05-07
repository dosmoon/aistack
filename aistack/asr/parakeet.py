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
    # NB: _maybe_enable_subsampling_chunking() and _configure_timestamp_
    # decoding() exist below but are NOT called by default. Live
    # measurement (2026-05-07) showed that turning them on doubled
    # inference time on the 50-min Rubio press-conference audio:
    # cache-hit run went from 62 s to ≥ 120 s and clients hit their
    # default httpx timeout. Suspected interaction between
    # preserve_alignments=True (the open NeMo issue #14714) and/or
    # change_decoding_strategy() resetting the local-attention switch
    # we set just above. The word-from-words segment synthesis in
    # _segments_from_words() already covers the empty-NeMo-segment
    # case in production, so these helpers are kept available for
    # future opt-in (env flag) but not on the default path.
    # Keep on whatever device NeMo picked (cuda if available, else cpu).
    model.eval()
    _model_cache.put(_PROVIDER_TAG, model_name, model, category="asr-main")
    device = _device_str(model)
    emit("model_loaded", model=model_name, device=device, compute_type="auto")
    return model


def _maybe_enable_subsampling_chunking(model) -> None:
    """Enable adaptive chunking on the FastConformer subsampling module.

    NVIDIA's HuggingFace discussions for parakeet-tdt-0.6b-v2/v3 recommend
    pairing local attention with `change_subsampling_conv_chunking_factor(1)`
    for long-form inference; without it the subsampling step still allocates
    contiguous activations across the whole audio, which OOMs on consumer
    GPUs and is implicated in the empty-segment-timestamps NeMo bug
    surfaced for >~30 min audio. `1` means "auto-pick chunking factor",
    not "no chunking".

    Best-effort — older NeMo lacks this method; we log and continue."""
    fn = getattr(model, "change_subsampling_conv_chunking_factor", None)
    if not callable(fn):
        logger.info(
            "model.change_subsampling_conv_chunking_factor not available; "
            "long audio may run without subsampling chunking"
        )
        return
    try:
        fn(1)
        logger.info("Parakeet subsampling conv chunking enabled (factor=auto)")
    except Exception as e:
        logger.warning(
            "change_subsampling_conv_chunking_factor failed (%s: %s); continuing",
            type(e).__name__, e,
        )


def _configure_timestamp_decoding(model) -> None:
    """Explicitly enable segment-level timestamps via the decoding config.

    NeMo ASR's documented contract for segment timestamps:
      decoding_cfg.preserve_alignments = True
      decoding_cfg.compute_timestamps  = True
      decoding_cfg.segment_seperators  = [".", "?", "!"]    # NB: NeMo's spelling
      decoding_cfg.word_seperator      = " "
      asr_model.change_decoding_strategy(decoding_cfg)

    Passing only `transcribe(timestamps=True)` historically gave us word-
    level timestamps but an empty `timestamp.segment` array on long-form
    audio under local attention — a confirmed NeMo bug (HF discussion
    parakeet-tdt-0.6b-v2 #15, NeMo issue #14714). Setting these flags up
    front routes through NeMo's punctuation-aware segment splitter,
    which is the model's intended path. Our word-from-words fallback in
    _normalize_hypothesis stays as defense — NeMo can still return
    empty segments on edge cases (no punctuation, very short audio).

    Best-effort — older NeMo, hybrid models, or non-RNNT decoders may
    not expose all these knobs. We swallow and log."""
    try:
        from omegaconf import open_dict  # NeMo's own dependency
    except ImportError:
        logger.info("omegaconf not available; skipping decoding-cfg tuning")
        return

    decoding_cfg = getattr(getattr(model, "cfg", None), "decoding", None)
    change_strategy = getattr(model, "change_decoding_strategy", None)
    if decoding_cfg is None or not callable(change_strategy):
        logger.info(
            "model lacks cfg.decoding / change_decoding_strategy; "
            "relying on word-from-words segment synthesis"
        )
        return
    try:
        with open_dict(decoding_cfg):
            decoding_cfg.preserve_alignments = True
            decoding_cfg.compute_timestamps = True
            # NeMo spells the key with one 'e' missing — kept verbatim.
            decoding_cfg.segment_seperators = [".", "?", "!"]
            decoding_cfg.word_seperator = " "
        change_strategy(decoding_cfg)
        logger.info(
            "Parakeet decoding strategy: timestamps=on, "
            "segment_seperators=%s",
            [".", "?", "!"],
        )
    except Exception as e:
        logger.warning(
            "change_decoding_strategy failed (%s: %s); falling back to "
            "transcribe(timestamps=True) + word-synthesis defense",
            type(e).__name__, e,
        )


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
    segment_granularity: str = "sentence",
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
        text, segments_out, words_out = _normalize_hypothesis(
            hyp, granularity=segment_granularity,
        )

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


def _normalize_hypothesis(
    hyp, *, granularity: str = "sentence",
) -> tuple[str, list, list]:
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
    # attention. Synthesize segments from the word timestamps using the
    # requested granularity so consumers downstream of verbose_json
    # (LLM translation / agents → "sentence", SRT writers → "subtitle")
    # get the right shape.
    if not segments_out and words_out:
        segments_out = _segments_from_words(words_out, granularity=granularity)
    # Last-resort fallback: text but no timestamps at all (some old
    # NeMo paths). Single-segment is wrong but better than dropping.
    if text and not segments_out:
        segments_out.append({"id": 0, "start": 0.0, "end": 0.0, "text": text})

    return text, segments_out, words_out


# ── Segment synthesis from word timestamps ───────────────────────────────────
#
# Two granularities are supported, because the right segment shape depends
# on the consumer:
#
#   "sentence"  (default) — semantic units (full sentences). What
#                            OpenAI/WhisperX/stable-ts return in their
#                            verbose_json `segments` field. Right input
#                            for LLM translation, agent reasoning,
#                            search-by-quote, summarisation, alignment
#                            with another transcription. WRONG input for
#                            SRT generation directly — sentences can be
#                            long, exceed 70 chars / 7 s.
#
#   "subtitle"            — SRT-ready cues (≤ 70 chars, 1–7 s, ≥ min_dur)
#                            for clients that want to write SRT/VTT
#                            without doing their own cue-sizing pass.
#                            WRONG input for line-by-line LLM translation
#                            because mid-sentence breaks lose context.
#
# Picking the right default matters: line-by-line LLM translation on
# subtitle-sized cues produces broken translations (incomplete clauses,
# tense/agreement errors, lost referents). The industry consensus is
# verbose_json => sentences, SRT export => downstream cue-sizing.
#
# Subtitle-mode thresholds and algorithm are modeled after stable-ts
# (jianfch/stable-ts), the de-facto OpenAI-Whisper-ecosystem standard
# for word→subtitle regrouping.
#
# Parameter sources:
#   max_chars     = 70       stable-ts default; ≈ 35 chars × 2 lines, the
#                            standard subtitle line length cap
#   min_chars     = 50       stable-ts default; below this we don't fall
#                            back to comma-splitting (avoids over-splitting)
#   max_gap_sec   = 0.5      stable-ts default; matches WhisperX's gap merge
#                            and the typical 0.4–0.6 s silence threshold
#   max_dur_sec   = 7.0      subtitle industry max cue duration
#   min_dur_sec   = 1.0      subtitle industry min cue duration (anything
#                            shorter "flickers" — viewer cannot read it)
#   sentence_ends = .!?。！？  Latin + CJK sentence-final punctuation
#   comma_ends    = ,，       Latin + CJK comma — secondary split fallback
#
# References:
#   stable-ts default regroup string:
#     "isp_cm_sp=.* /。/?/？_sg=.5_sp=,* /，++++50_sl=70_cm"
#   subtitle-localisation conventions: 17 CPS reading-speed cap, 1–7 s cue
#   duration, 32–42 chars/line × 2 lines.
# Subtitle-mode parameters (stable-ts defaults + subtitle-localisation
# industry standards: ≤ 70 chars/cue, 1–7 s cue duration).
_SEGMENT_MAX_CHARS = 70
_SEGMENT_MIN_CHARS_FOR_COMMA_SPLIT = 50
_SEGMENT_MAX_GAP_SEC = 0.5
_SEGMENT_MAX_DURATION_SEC = 7.0
_SEGMENT_MIN_DURATION_SEC = 1.0

# Sentence-mode parameters. Looser than subtitle mode by design — we
# want full semantic units. The hard duration cap is a sanity bound
# for run-on speech with no punctuation (WhisperX PR #982 reported
# 60–80 s monster segments without one), not a subtitle constraint.
_SENTENCE_MAX_GAP_SEC = 0.7
_SENTENCE_HARD_MAX_DURATION_SEC = 30.0

_SEGMENT_SENTENCE_ENDERS = (".", "!", "?", "。", "！", "？")
_SEGMENT_COMMA_ENDERS = (",", "，")

VALID_GRANULARITIES = ("sentence", "subtitle")


def _segments_from_words(
    words: list[dict],
    *,
    granularity: str = "sentence",
) -> list[dict]:
    """Group word-level timestamps into segments.

    granularity:
        "sentence" (default) — full-sentence segments. Splits on
            sentence-final punctuation, long silence (> 0.7 s), or a
            30 s safety cap. Suitable for verbose_json's `segments`
            field, line-by-line LLM translation, semantic search, etc.

        "subtitle" — SRT-cue-sized segments via stable-ts's regroup
            pipeline (sentence enders → silence gap → comma fallback
            → length split). Every cue is ≥ 1 s, ≤ 7 s, ≤ 70 chars.
            Suitable for direct SRT emission. NOT suitable for
            line-by-line LLM translation (mid-sentence cuts break
            translation context)."""
    if not words:
        return []
    if granularity not in VALID_GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {VALID_GRANULARITIES}, got {granularity!r}"
        )

    if granularity == "sentence":
        return _publish(_sentence_split(words))

    # granularity == "subtitle": stable-ts three-stage pipeline.
    raw = _stage1_primary_split(words)

    refined: list[dict] = []
    for seg in raw:
        if _seg_chars(seg) <= _SEGMENT_MAX_CHARS and _seg_dur(seg) <= _SEGMENT_MAX_DURATION_SEC:
            refined.append(seg)
            continue
        refined.extend(_stage2_comma_split(seg))

    final: list[dict] = []
    for seg in refined:
        if _seg_chars(seg) <= _SEGMENT_MAX_CHARS and _seg_dur(seg) <= _SEGMENT_MAX_DURATION_SEC:
            final.append(seg)
            continue
        final.extend(_stage3_hard_split(seg))

    return _publish(final)


def _publish(segs: list[dict]) -> list[dict]:
    """Drop internal _words and re-id; emit the public contract."""
    out: list[dict] = []
    for idx, s in enumerate(segs):
        out.append({
            "id": idx,
            "start": s["start"],
            "end": s["end"],
            "text": s["text"],
        })
    return out


def _sentence_split(words: list[dict]) -> list[dict]:
    """Sentence-level split. Closes the current segment when:
       - the previous word ends with a sentence-final punctuation, OR
       - silence gap > _SENTENCE_MAX_GAP_SEC (0.7 s — paragraph/turn
         boundary), OR
       - segment duration would exceed _SENTENCE_HARD_MAX_DURATION_SEC
         (30 s, sanity bound for run-on speech without punctuation).
    No char/duration constraint suitable for SRT cues — those are
    deliberately the consumer's job in subtitle export."""
    segs: list[dict] = []
    run: list[dict] = []
    cur_start = 0.0
    cur_end = 0.0

    def flush() -> None:
        nonlocal run, cur_start, cur_end
        if not run:
            return
        segs.append(_seg_from_word_run(run))
        run = []

    for w in words:
        wt = (w.get("word") or "").strip()
        if not wt:
            continue
        ws = float(w.get("start", 0.0))
        we = float(w.get("end", ws))

        if not run:
            run = [w]
            cur_start = ws
            cur_end = we
            continue

        gap = ws - cur_end
        prev_text = (run[-1].get("word") or "").rstrip()
        prev_ends_sentence = prev_text.endswith(_SEGMENT_SENTENCE_ENDERS)
        prospective_dur = we - cur_start

        if (prev_ends_sentence
                or gap > _SENTENCE_MAX_GAP_SEC
                or prospective_dur > _SENTENCE_HARD_MAX_DURATION_SEC):
            flush()
            run = [w]
            cur_start = ws
            cur_end = we
            continue

        run.append(w)
        cur_end = we

    flush()
    return segs


def _seg_chars(seg: dict) -> int:
    return len(seg.get("text", ""))


def _seg_dur(seg: dict) -> float:
    return float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))


def _seg_from_word_run(run: list[dict]) -> dict:
    """Build a segment from a list of word dicts {start,end,word}."""
    text = " ".join(w["word"] for w in run if w.get("word")).strip()
    return {
        "id": 0,  # set by caller
        "start": float(run[0].get("start", 0.0)),
        "end": float(run[-1].get("end", 0.0)),
        "text": text,
        "_words": list(run),  # internal — used by stage 2/3 to re-split
    }


def _stage1_primary_split(words: list[dict]) -> list[dict]:
    """Stage 1: sentence enders / silence gap / size limits.
    Honours min_dur to avoid flickering cues."""
    segs: list[dict] = []
    run: list[dict] = []
    cur_chars = 0
    cur_start = 0.0
    cur_end = 0.0

    def flush() -> None:
        nonlocal run, cur_chars, cur_start, cur_end
        if not run:
            return
        segs.append(_seg_from_word_run(run))
        run = []
        cur_chars = 0

    for w in words:
        wt = (w.get("word") or "").strip()
        if not wt:
            continue
        ws = float(w.get("start", 0.0))
        we = float(w.get("end", ws))

        if not run:
            run = [w]
            cur_start = ws
            cur_end = we
            cur_chars = len(wt)
            continue

        gap = ws - cur_end
        prev_text = (run[-1].get("word") or "").rstrip()
        prev_ends_sentence = prev_text.endswith(_SEGMENT_SENTENCE_ENDERS)
        cur_dur = cur_end - cur_start
        prospective_chars = cur_chars + 1 + len(wt)  # +1 for space
        prospective_dur = we - cur_start

        # Conditions that argue for breaking BEFORE this word.
        gap_break = gap > _SEGMENT_MAX_GAP_SEC
        sentence_break = prev_ends_sentence
        size_break = (
            prospective_chars > _SEGMENT_MAX_CHARS
            or prospective_dur > _SEGMENT_MAX_DURATION_SEC
        )

        if (gap_break or sentence_break or size_break) and cur_dur >= _SEGMENT_MIN_DURATION_SEC:
            flush()
            run = [w]
            cur_start = ws
            cur_end = we
            cur_chars = len(wt)
            continue

        run.append(w)
        cur_end = we
        cur_chars = prospective_chars

    flush()
    return segs


def _stage2_comma_split(seg: dict) -> list[dict]:
    """Stage 2: split an over-long segment at commas, but only if the
    prefix has accumulated at least min_chars_for_comma_split. This
    avoids producing a tiny "yes," prefix that flickers on screen."""
    words = seg.get("_words") or []
    if not words:
        return [seg]

    out: list[dict] = []
    run: list[dict] = []
    cur_chars = 0
    for w in words:
        wt = (w.get("word") or "").strip()
        if not wt:
            continue
        if not run:
            run = [w]
            cur_chars = len(wt)
            continue
        run.append(w)
        cur_chars += 1 + len(wt)
        ends_with_comma = wt.rstrip().endswith(_SEGMENT_COMMA_ENDERS)
        if ends_with_comma and cur_chars >= _SEGMENT_MIN_CHARS_FOR_COMMA_SPLIT:
            out.append(_seg_from_word_run(run))
            run = []
            cur_chars = 0
    if run:
        out.append(_seg_from_word_run(run))
    return out or [seg]


def _stage3_hard_split(seg: dict) -> list[dict]:
    """Stage 3: brute-force split by max_chars / max_dur. Last resort
    for run-on speech with neither punctuation nor silence."""
    words = seg.get("_words") or []
    if not words:
        return [seg]

    out: list[dict] = []
    run: list[dict] = []
    cur_chars = 0
    cur_start = 0.0
    for w in words:
        wt = (w.get("word") or "").strip()
        if not wt:
            continue
        ws = float(w.get("start", 0.0))
        we = float(w.get("end", ws))
        if not run:
            run = [w]
            cur_chars = len(wt)
            cur_start = ws
            continue
        prospective_chars = cur_chars + 1 + len(wt)
        prospective_dur = we - cur_start
        if (prospective_chars > _SEGMENT_MAX_CHARS
                or prospective_dur > _SEGMENT_MAX_DURATION_SEC):
            out.append(_seg_from_word_run(run))
            run = [w]
            cur_chars = len(wt)
            cur_start = ws
            continue
        run.append(w)
        cur_chars = prospective_chars
    if run:
        out.append(_seg_from_word_run(run))
    return out
