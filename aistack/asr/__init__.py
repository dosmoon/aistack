"""ASR providers.

Each module (faster_whisper / parakeet / sensevoice) lazy-imports its
heavy ML library inside _get_model() so this package can be imported
even when none of faster-whisper / nemo_toolkit / funasr are installed.
"""

from __future__ import annotations


def installed_providers() -> list[str]:
    """Return provider IDs whose ML library is importable in this venv.

    Used by /v1/models to surface only the ASR backends that will
    actually work, so clients can avoid 503s on cold-call.

    Pure import probe — no model weights are loaded.
    """
    out: list[str] = []
    try:
        import faster_whisper  # noqa: F401
        out.append("faster-whisper")
    except ImportError:
        pass
    try:
        from nemo.collections.asr.models import ASRModel  # noqa: F401
        out.append("parakeet")
    except ImportError:
        pass
    try:
        from funasr import AutoModel  # noqa: F401
        out.append("sensevoice")
    except ImportError:
        pass
    return out


# Representative model id surfaced in /v1/models per provider. Clients
# may also pass any whisper size (whisper-tiny, whisper-medium, ...)
# directly to /v1/audio/transcriptions.
_REPRESENTATIVE_MODELS = {
    "faster-whisper": {
        "id": "whisper-small",
        "owned_by": "openai",
    },
    "parakeet": {
        "id": "nvidia/parakeet-tdt-0.6b-v3",
        "owned_by": "nvidia",
    },
    "sensevoice": {
        "id": "iic/SenseVoiceSmall",
        "owned_by": "alibaba",
    },
}

# ISO 639-1 codes each backend can transcribe. Surfaced in /v1/models so
# clients can build language-aware pickers without baking a per-backend
# language table into their own code. Listed as a per-provider data
# table here (rather than imported from inside aistack/api/asr.py) so
# this module stays a pure import probe with no router dependencies.
_LANGUAGES = {
    # Whisper officially supports 99 languages — listed in full so
    # clients can filter precisely, not approximated as "multilingual".
    "faster-whisper": [
        "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo",
        "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es",
        "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw",
        "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja",
        "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo",
        "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
        "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt",
        "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq",
        "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl",
        "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "yue", "zh",
    ],
    # Parakeet TDT 0.6B v3 was trained on 25 European languages plus
    # English — full list per the NVIDIA model card.
    "parakeet": [
        "en", "bg", "hr", "cs", "da", "nl", "et", "fi", "fr", "de",
        "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
        "sl", "es", "sv", "ru", "uk",
    ],
    # SenseVoice Small officially supports the CJK family plus English,
    # Japanese, and Korean.
    "sensevoice": ["zh", "yue", "en", "ja", "ko"],
}


def model_entries() -> list[dict]:
    """OpenAI-shape model entries for every installed ASR provider, plus
    the `auto` routing alias when at least one provider is reachable.

    The auto entry is emitted first so picker UIs can lead with it
    without needing to sort. It carries `is_routing_alias: true` so
    aistack-aware clients can render it specially while OpenAI-shape
    clients simply see it as a model id (which the gateway accepts).

    Each real ASR entry carries a `languages` array listing the ISO
    639-1 codes the backend can transcribe — clients should use this
    to filter pickers by user-requested language, not bake their own
    per-backend language table.
    """
    out: list[dict] = []
    installed = installed_providers()

    if installed:
        # Auto routing — only meaningful when at least one ASR backend
        # is reachable; otherwise it would route to nothing.
        out.append({
            "id": "auto",
            "object": "model",
            "owned_by": "aistack",
            "capabilities": ["asr"],
            "is_routing_alias": True,
        })

    for pid in installed:
        rep = _REPRESENTATIVE_MODELS[pid]
        out.append({
            "id": rep["id"],
            "object": "model",
            "owned_by": rep["owned_by"],
            "capabilities": ["asr"],
            "languages": list(_LANGUAGES[pid]),
        })
    return out
