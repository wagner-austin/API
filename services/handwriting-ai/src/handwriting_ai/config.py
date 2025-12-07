from __future__ import annotations

from pathlib import Path
from typing import Final

from platform_core.config import (
    HandwritingAiAppConfig as AppConfig,
)
from platform_core.config import (
    HandwritingAiDigitsConfig as DigitsConfig,
)
from platform_core.config import (
    HandwritingAiLimits as Limits,
)
from platform_core.config import (
    HandwritingAiSecurityConfig as SecurityConfig,
)
from platform_core.config import (
    HandwritingAiSettings as Settings,
)
from platform_core.config import (
    limits_from_handwriting_ai_settings as limits_from_settings,
)
from platform_core.config import (
    load_handwriting_ai_settings,
)
from platform_core.config._utils import _EnvError
from platform_core.config.handwriting_ai import (
    DEFAULT_ALLOWED_HOSTS,
    DEFAULT_MAX_IMAGE_MB,
    DEFAULT_MAX_IMAGE_SIDE_PX,
    DEFAULT_PORT,
    DEFAULT_PREDICT_TIMEOUT_SECONDS,
    DEFAULT_RETENTION_KEEP_RUNS,
    DEFAULT_THREADS,
    DEFAULT_UNCERTAIN_THRESHOLD,
    DEFAULT_VISUALIZE_MAX_KB,
    _apply_app_env,
    _apply_digits_env,
    _apply_security_env,
    _finalize,
)
from platform_core.logging import get_logger

MNIST_N_CLASSES: Final[int] = 10


def _default_settings(*, create_dirs: bool) -> Settings:
    """Build a local default settings object when required env vars are absent."""
    base_dir = (Path.cwd() / ".handwriting-ai").resolve()
    data_root = base_dir / "data"
    artifacts_root = base_dir / "artifacts"
    logs_root = base_dir / "logs"
    model_dir = base_dir / "models"
    if create_dirs:
        for path in (data_root, artifacts_root, logs_root, model_dir):
            path.mkdir(parents=True, exist_ok=True)
    app: AppConfig = {
        "data_root": data_root,
        "artifacts_root": artifacts_root,
        "logs_root": logs_root,
        "threads": DEFAULT_THREADS,
        "port": DEFAULT_PORT,
    }
    digits: DigitsConfig = {
        "model_dir": model_dir,
        "active_model": "default",
        "tta": False,
        "uncertain_threshold": DEFAULT_UNCERTAIN_THRESHOLD,
        "max_image_mb": DEFAULT_MAX_IMAGE_MB,
        "max_image_side_px": DEFAULT_MAX_IMAGE_SIDE_PX,
        "predict_timeout_seconds": DEFAULT_PREDICT_TIMEOUT_SECONDS,
        "visualize_max_kb": DEFAULT_VISUALIZE_MAX_KB,
        "retention_keep_runs": DEFAULT_RETENTION_KEEP_RUNS,
        "allowed_hosts": DEFAULT_ALLOWED_HOSTS,
    }
    security: SecurityConfig = {"api_key": "", "api_key_enabled": False}
    return {"app": app, "digits": digits, "security": security}


def ensure_settings(settings: Settings, *, create_dirs: bool) -> Settings:
    """Fill missing settings fields using platform_core defaults.

    Tests supply partial Settings dicts; this helper keeps behavior consistent
    with the centralized defaults while avoiding hard failures on missing keys.
    """
    app_cfg = settings["app"].copy()
    digits_cfg = settings["digits"].copy()
    security_cfg = settings["security"].copy()

    app_cfg.setdefault("data_root", (Path.cwd() / ".handwriting-ai" / "data").resolve())
    app_cfg.setdefault("artifacts_root", (Path.cwd() / ".handwriting-ai" / "artifacts").resolve())
    app_cfg.setdefault("logs_root", (Path.cwd() / ".handwriting-ai" / "logs").resolve())
    app_cfg.setdefault("threads", DEFAULT_THREADS)
    app_cfg.setdefault("port", DEFAULT_PORT)

    digits_cfg.setdefault("model_dir", (Path.cwd() / ".handwriting-ai" / "models").resolve())
    digits_cfg.setdefault("seed_root", Path("/app/seed/digits/models"))
    digits_cfg.setdefault("active_model", "default")
    digits_cfg.setdefault("tta", False)
    digits_cfg.setdefault("uncertain_threshold", DEFAULT_UNCERTAIN_THRESHOLD)
    digits_cfg.setdefault("max_image_mb", DEFAULT_MAX_IMAGE_MB)
    digits_cfg.setdefault("max_image_side_px", DEFAULT_MAX_IMAGE_SIDE_PX)
    digits_cfg.setdefault("predict_timeout_seconds", DEFAULT_PREDICT_TIMEOUT_SECONDS)
    digits_cfg.setdefault("visualize_max_kb", DEFAULT_VISUALIZE_MAX_KB)
    digits_cfg.setdefault("retention_keep_runs", DEFAULT_RETENTION_KEEP_RUNS)
    digits_cfg.setdefault("allowed_hosts", DEFAULT_ALLOWED_HOSTS)

    api_key_val = str(security_cfg.get("api_key", ""))
    api_key_enabled = bool(security_cfg.get("api_key_enabled", api_key_val != ""))
    security_cfg["api_key"] = api_key_val
    security_cfg["api_key_enabled"] = api_key_enabled

    if create_dirs:
        for path in (
            app_cfg["data_root"],
            app_cfg["artifacts_root"],
            app_cfg["logs_root"],
            digits_cfg["model_dir"],
        ):
            path.mkdir(parents=True, exist_ok=True)

    return {"app": app_cfg, "digits": digits_cfg, "security": security_cfg}


def load_settings(*, create_dirs: bool = True) -> Settings:
    """Load settings via shared platform_core config helpers with safe fallback."""
    try:
        base = load_handwriting_ai_settings(create_dirs=create_dirs)
    except _EnvError:
        get_logger("handwriting_ai").warning("env_settings_missing_using_default")
        base_default = _default_settings(create_dirs=create_dirs)
        base = _finalize(
            {
                "app": _apply_app_env(base_default["app"]),
                "digits": _apply_digits_env(base_default["digits"]),
                "security": _apply_security_env(base_default["security"]),
            }
        )
    return ensure_settings(base, create_dirs=create_dirs)


__all__ = [
    "MNIST_N_CLASSES",
    "AppConfig",
    "DigitsConfig",
    "Limits",
    "SecurityConfig",
    "Settings",
    "ensure_settings",
    "limits_from_settings",
    "load_settings",
]
