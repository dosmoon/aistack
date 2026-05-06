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
        "capabilities": ["asr"],
    },
    "parakeet": {
        "id": "nvidia/parakeet-tdt-0.6b-v3",
        "owned_by": "nvidia",
        "capabilities": ["asr"],
    },
    "sensevoice": {
        "id": "iic/SenseVoiceSmall",
        "owned_by": "alibaba",
        "capabilities": ["asr"],
    },
}


def model_entries() -> list[dict]:
    """OpenAI-shape model entries for every installed ASR provider."""
    out: list[dict] = []
    for pid in installed_providers():
        entry = _REPRESENTATIVE_MODELS[pid]
        out.append({"object": "model", **entry})
    return out
