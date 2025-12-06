from __future__ import annotations

from pathlib import Path
from typing import Final, TypedDict

from ._utils import _optional_env_str, _require_env_csv, _require_env_str

DEFAULT_PORT: Final[int] = 8000
DEFAULT_THREADS: Final[int] = 0
DEFAULT_PREDICT_TIMEOUT_SECONDS: Final[int] = 5
DEFAULT_VISUALIZE_MAX_KB: Final[int] = 64
DEFAULT_MAX_IMAGE_MB: Final[int] = 10
DEFAULT_MAX_IMAGE_SIDE_PX: Final[int] = 2048
DEFAULT_RETENTION_KEEP_RUNS: Final[int] = 5
DEFAULT_UNCERTAIN_THRESHOLD: Final[float] = 0.7
DEFAULT_ALLOWED_HOSTS: Final[frozenset[str]] = frozenset()


class HandwritingAiAppConfig(TypedDict, total=False):
    data_root: Path
    artifacts_root: Path
    logs_root: Path
    threads: int
    port: int
    data_bank_api_url: str
    data_bank_api_key: str


class HandwritingAiDigitsConfig(TypedDict, total=False):
    model_dir: Path
    active_model: str
    tta: bool
    uncertain_threshold: float
    max_image_mb: int
    max_image_side_px: int
    predict_timeout_seconds: int
    visualize_max_kb: int
    retention_keep_runs: int
    allowed_hosts: frozenset[str]


class HandwritingAiSecurityConfig(TypedDict, total=False):
    api_key: str
    api_key_enabled: bool


class HandwritingAiSettings(TypedDict):
    app: HandwritingAiAppConfig
    digits: HandwritingAiDigitsConfig
    security: HandwritingAiSecurityConfig


class HandwritingAiLimits(TypedDict):
    max_bytes: int
    max_side_px: int


def _bool_env(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    normalized = val.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError("Invalid boolean value")


def _base_settings(create_dirs: bool) -> HandwritingAiSettings:
    data_root = Path(_require_env_str("HANDWRITING_DATA_ROOT")).resolve()
    artifacts_root = Path(_require_env_str("HANDWRITING_ARTIFACTS_ROOT")).resolve()
    logs_root = Path(_require_env_str("HANDWRITING_LOGS_ROOT")).resolve()
    model_dir = Path(_optional_env_str("HANDWRITING_MODEL_DIR") or "models").resolve()
    if create_dirs:
        for p in (data_root, artifacts_root, logs_root, model_dir):
            p.mkdir(parents=True, exist_ok=True)
    allowed_hosts_env = _optional_env_str("HANDWRITING_ALLOWED_HOSTS")
    allowed_hosts = (
        _require_env_csv("HANDWRITING_ALLOWED_HOSTS")
        if allowed_hosts_env
        else DEFAULT_ALLOWED_HOSTS
    )
    return {
        "app": {
            "data_root": data_root,
            "artifacts_root": artifacts_root,
            "logs_root": logs_root,
            "threads": DEFAULT_THREADS,
            "port": DEFAULT_PORT,
        },
        "digits": {
            "model_dir": model_dir,
            "active_model": _require_env_str("HANDWRITING_MODEL_ID"),
            "tta": False,
            "uncertain_threshold": DEFAULT_UNCERTAIN_THRESHOLD,
            "max_image_mb": DEFAULT_MAX_IMAGE_MB,
            "max_image_side_px": DEFAULT_MAX_IMAGE_SIDE_PX,
            "predict_timeout_seconds": DEFAULT_PREDICT_TIMEOUT_SECONDS,
            "visualize_max_kb": DEFAULT_VISUALIZE_MAX_KB,
            "retention_keep_runs": DEFAULT_RETENTION_KEEP_RUNS,
            "allowed_hosts": allowed_hosts,
        },
        "security": {"api_key": _require_env_str("HANDWRITING_API_KEY"), "api_key_enabled": True},
    }


def _apply_app_env(app: HandwritingAiAppConfig) -> HandwritingAiAppConfig:
    out = app.copy()
    if (v := _optional_env_str("APP__DATA_ROOT")) is not None:
        out["data_root"] = Path(v).resolve()
    if (v := _optional_env_str("APP__ARTIFACTS_ROOT")) is not None:
        out["artifacts_root"] = Path(v).resolve()
    if (v := _optional_env_str("APP__LOGS_ROOT")) is not None:
        out["logs_root"] = Path(v).resolve()
    if (v := _optional_env_str("APP__THREADS")) is not None:
        out["threads"] = int(v)
    if (v := _optional_env_str("APP__PORT")) is not None:
        port = int(v)
        if port <= 0 or port > 65535:
            raise RuntimeError("port out of range")
        out["port"] = port
    if (v := _optional_env_str("APP__DATA_BANK_API_URL")) is not None:
        out["data_bank_api_url"] = v
    if (v := _optional_env_str("APP__DATA_BANK_API_KEY")) is not None:
        out["data_bank_api_key"] = v
    return out


def _apply_digits_env(digits: HandwritingAiDigitsConfig) -> HandwritingAiDigitsConfig:
    out = digits.copy()
    if (v := _optional_env_str("DIGITS__MODEL_DIR")) is not None:
        out["model_dir"] = Path(v).resolve()
    if (v := _optional_env_str("DIGITS__ACTIVE_MODEL")) is not None:
        out["active_model"] = v
    if (v := _optional_env_str("DIGITS__TTA")) is not None:
        out["tta"] = _bool_env(v, out["tta"])
    if (v := _optional_env_str("DIGITS__UNCERTAIN_THRESHOLD")) is not None:
        out["uncertain_threshold"] = float(v)
    if (v := _optional_env_str("DIGITS__MAX_IMAGE_MB")) is not None:
        out["max_image_mb"] = int(v)
    if (v := _optional_env_str("DIGITS__MAX_IMAGE_SIDE_PX")) is not None:
        out["max_image_side_px"] = int(v)
    if (v := _optional_env_str("DIGITS__PREDICT_TIMEOUT_SECONDS")) is not None:
        out["predict_timeout_seconds"] = int(v)
    if (v := _optional_env_str("DIGITS__VISUALIZE_MAX_KB")) is not None:
        out["visualize_max_kb"] = int(v)
    if (v := _optional_env_str("DIGITS__RETENTION_KEEP_RUNS")) is not None:
        out["retention_keep_runs"] = int(v)
    return out


def _apply_security_env(sec: HandwritingAiSecurityConfig) -> HandwritingAiSecurityConfig:
    out = sec.copy()
    if (v := _optional_env_str("SECURITY__API_KEY")) is not None:
        out["api_key"] = v
    if (v := _optional_env_str("SECURITY__API_KEY_ENABLED")) is not None:
        out["api_key_enabled"] = _bool_env(v, True)
    return out


def _finalize(settings: HandwritingAiSettings) -> HandwritingAiSettings:
    sec = settings["security"].copy()
    if sec.get("api_key_enabled") is False:
        sec["api_key"] = ""
        sec["api_key_enabled"] = False
    else:
        sec["api_key"] = str(sec.get("api_key", ""))
        sec["api_key_enabled"] = True
    return {"app": settings["app"], "digits": settings["digits"], "security": sec}


def load_handwriting_ai_settings(*, create_dirs: bool = True) -> HandwritingAiSettings:
    """Load handwriting-ai settings from environment variables.

    Environment variables:
    - HANDWRITING_DATA_ROOT (required)
    - HANDWRITING_ARTIFACTS_ROOT (required)
    - HANDWRITING_LOGS_ROOT (required)
    - HANDWRITING_MODEL_DIR (optional, defaults to "models")
    - HANDWRITING_MODEL_ID (required)
    - HANDWRITING_API_KEY (required)
    - HANDWRITING_ALLOWED_HOSTS (optional, CSV list)

    Override env vars (optional):
    - APP__DATA_ROOT, APP__ARTIFACTS_ROOT, APP__LOGS_ROOT
    - APP__THREADS, APP__PORT
    - DIGITS__MODEL_DIR, DIGITS__ACTIVE_MODEL, DIGITS__TTA
    - DIGITS__UNCERTAIN_THRESHOLD, DIGITS__MAX_IMAGE_MB, DIGITS__MAX_IMAGE_SIDE_PX
    - DIGITS__PREDICT_TIMEOUT_SECONDS, DIGITS__VISUALIZE_MAX_KB, DIGITS__RETENTION_KEEP_RUNS
    - SECURITY__API_KEY, SECURITY__API_KEY_ENABLED
    """
    base = _base_settings(create_dirs)
    app = _apply_app_env(base["app"])
    digits = _apply_digits_env(base["digits"])
    sec = _apply_security_env(base["security"])
    after_env: HandwritingAiSettings = {"app": app, "digits": digits, "security": sec}
    return _finalize(after_env)


def limits_from_handwriting_ai_settings(settings: HandwritingAiSettings) -> HandwritingAiLimits:
    """Extract image upload limits from settings."""
    digits = settings["digits"]
    max_mb = int(digits.get("max_image_mb", DEFAULT_MAX_IMAGE_MB))
    max_side = int(digits.get("max_image_side_px", DEFAULT_MAX_IMAGE_SIDE_PX))
    return {"max_bytes": max_mb * 1024 * 1024, "max_side_px": max_side}


__all__ = [
    "HandwritingAiAppConfig",
    "HandwritingAiDigitsConfig",
    "HandwritingAiLimits",
    "HandwritingAiSecurityConfig",
    "HandwritingAiSettings",
    "limits_from_handwriting_ai_settings",
    "load_handwriting_ai_settings",
]
