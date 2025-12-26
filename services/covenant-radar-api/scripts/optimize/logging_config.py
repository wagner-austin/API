"""Logging configuration for optimization runs."""

from __future__ import annotations

import warnings

from platform_core.logging import stdlib_logging

# Module-level verbose flag for optuna logging control
_verbose_mode: bool = False


def set_verbose_mode(verbose: bool) -> None:
    """Set the verbose mode flag."""
    global _verbose_mode
    _verbose_mode = verbose


def suppress_verbose_logging() -> None:
    """Suppress verbose logging unless verbose mode is enabled.

    Suppresses Optuna and optimize_job loggers to WARNING level
    to avoid spammy output. Call this right before optimization starts.
    """
    if _verbose_mode:
        return

    # Suppress optuna and its subloggers
    optuna_logger = stdlib_logging.getLogger("optuna")
    optuna_logger.setLevel(stdlib_logging.WARNING)

    for name in ("optuna.trial", "optuna.study", "optuna._optimize"):
        stdlib_logging.getLogger(name).setLevel(stdlib_logging.WARNING)

    # Suppress optimize_job logging when using progress display
    stdlib_logging.getLogger("covenant_radar_api.worker.optimize_job").setLevel(
        stdlib_logging.WARNING
    )

    # Suppress training progress logs from backends
    for name in (
        "covenant_ml.trainer",
        "covenant_ml.backends.mlp.backend",
        "covenant_ml.backends.lstm.backend",
        "covenant_ml.optimizer.objectives.lightgbm_objective",
        "covenant_ml.backends.lightgbm.backend",
    ):
        stdlib_logging.getLogger(name).setLevel(stdlib_logging.WARNING)

    # Suppress covenant_ml optuna backend logging (Trial complete messages)
    stdlib_logging.getLogger("covenant_ml.optimizer.optuna_backend").setLevel(
        stdlib_logging.WARNING
    )

    # Also suppress real_data loading messages
    stdlib_logging.getLogger("covenant_radar_api.seeding.real_data").setLevel(
        stdlib_logging.WARNING
    )

    # Suppress XGBoost/LightGBM GPU fallback warnings
    warnings.filterwarnings("ignore", message=".*No visible GPU is found.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*Device is changed from GPU to CPU.*", category=UserWarning)
