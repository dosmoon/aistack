"""Single source of truth for aistack configuration.

All env-driven knobs live here, grouped by feature domain. Each domain
is a frozen dataclass populated once at import (`from_env`). Modules
import the singleton `config` and read attributes; tests mutate by
constructing a new Config and monkey-patching the module-level binding.

Convention:
    from aistack.config import config
    interval = config.parakeet.chunk_overlap_sec

Variables that admin/API toggles flip at runtime (currently only the
three observability switches) live in `aistack/observability/config.py`
on top of a thread-locked mutable dict. The static fields they share
(paths, size budgets) come from this module.

Adding a new env variable:
    1. Pick or create the right domain dataclass.
    2. Add the field with its parsed type and default.
    3. Wire the env read in that domain's `from_env` classmethod.
    4. Update docs/configuration.md (the user-facing reference points
       its file:line links here).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ── Generic env parsers ─────────────────────────────────────────────────────

_TRUTHY = {"1", "on", "true", "yes", "y"}
_FALSY = {"0", "off", "false", "no", "n", ""}


def _env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw if raw is not None else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in _TRUTHY:
        return True
    if v in _FALSY:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ── Model lifecycle (shared across all backends) ────────────────────────────

@dataclass(frozen=True)
class ModelCacheConfig:
    """Idle eviction policy for the global model cache."""
    keep_alive_sec: float
    scan_interval_sec: float

    @classmethod
    def from_env(cls) -> "ModelCacheConfig":
        return cls(
            keep_alive_sec=_env_float("AISTACK_MODEL_KEEP_ALIVE_SEC", 300.0),
            scan_interval_sec=_env_float("AISTACK_MODEL_SCAN_INTERVAL_SEC", 60.0),
        )


# ── Parakeet ASR ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ParakeetConfig:
    """Attention mode + long-audio chunking knobs for Parakeet TDT."""
    attention_mode: str           # 'local' | 'full'
    att_context_size: str         # '<left>,<right>' frames at 80 ms each
    chunk_disable: bool
    chunk_window_sec: float       # default 720 = 12 min
    chunk_overlap_sec: float      # default 120 = 2 min
    chunk_min_last_sec: float     # default 300 = 5 min — tail-merge floor

    @classmethod
    def from_env(cls) -> "ParakeetConfig":
        return cls(
            attention_mode=_env("AISTACK_PARAKEET_ATTENTION_MODE", "local").lower(),
            att_context_size=_env("AISTACK_PARAKEET_ATT_CONTEXT_SIZE", "256,256"),
            chunk_disable=_env_bool("AISTACK_PARAKEET_CHUNK_DISABLE", False),
            chunk_window_sec=_env_float("AISTACK_PARAKEET_CHUNK_WINDOW_SEC", 720.0),
            chunk_overlap_sec=_env_float("AISTACK_PARAKEET_CHUNK_OVERLAP_SEC", 120.0),
            chunk_min_last_sec=_env_float("AISTACK_PARAKEET_CHUNK_MIN_LAST_SEC", 300.0),
        )


# ── Backend upstream URLs ───────────────────────────────────────────────────

@dataclass(frozen=True)
class BackendsConfig:
    """Where local LLM / TTS daemons live."""
    ollama_url: str
    qwen3_tts_upstream: str

    @classmethod
    def from_env(cls) -> "BackendsConfig":
        return cls(
            ollama_url=_env("AISTACK_OLLAMA_URL", "http://127.0.0.1:11434"),
            qwen3_tts_upstream=_env("AISTACK_QWEN3_TTS_UPSTREAM", "http://127.0.0.1:17860"),
        )


# ── Observability (static fields only — runtime toggles live in observability/config.py) ──

def _default_payload_dir() -> Path:
    explicit = os.environ.get("AISTACK_OBS_PAYLOAD_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    hf = os.environ.get("HF_HOME")
    if hf:
        return Path(hf).expanduser().resolve().parent / "aistack_captures"
    return Path.cwd() / "captures"


def _default_log_dir() -> Path:
    return Path(_env("AISTACK_OBS_LOG_DIR", "./logs")).expanduser().resolve()


@dataclass(frozen=True)
class ObservabilityConfig:
    """Static observability config — paths and budgets.

    Boot-time defaults for the three runtime-mutable toggles
    (metrics / access_log / payload) live here too; the actual mutable
    state is in `aistack.observability.config` because admin POST flips
    it at runtime.
    """
    metrics_default: bool
    access_log_default: bool
    payload_default: bool
    log_dir: Path
    payload_dir: Path
    payload_max_bytes: int
    payload_max_days: int
    payload_resp_max_bytes: int
    metrics_window_sec: int

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        return cls(
            metrics_default=_env_bool("AISTACK_OBS_METRICS", True),
            access_log_default=_env_bool("AISTACK_OBS_ACCESS_LOG", True),
            payload_default=_env_bool("AISTACK_OBS_PAYLOAD", False),
            log_dir=_default_log_dir(),
            payload_dir=_default_payload_dir(),
            payload_max_bytes=int(_env_float("AISTACK_OBS_PAYLOAD_MAX_GB", 5.0) * (1024 ** 3)),
            payload_max_days=_env_int("AISTACK_OBS_PAYLOAD_MAX_DAYS", 7),
            payload_resp_max_bytes=int(_env_float("AISTACK_OBS_PAYLOAD_RESP_MAX_MB", 50.0) * (1024 ** 2)),
            metrics_window_sec=_env_int("AISTACK_OBS_METRICS_WINDOW_MIN", 60) * 60,
        )


# ── Top-level singleton ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    model_cache: ModelCacheConfig
    parakeet: ParakeetConfig
    backends: BackendsConfig
    observability: ObservabilityConfig

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            model_cache=ModelCacheConfig.from_env(),
            parakeet=ParakeetConfig.from_env(),
            backends=BackendsConfig.from_env(),
            observability=ObservabilityConfig.from_env(),
        )


config: Config = Config.from_env()
