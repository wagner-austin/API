"""Path utilities for Art-Trainer.

This module provides path helpers for various directories used by the service.
"""

from __future__ import annotations

from pathlib import Path

from art_trainer.core.config.settings import Settings


def lora_output_dir(settings: Settings, job_id: str) -> Path:
    """Get the output directory for a LoRA training job.

    Args:
        settings: Application settings.
        job_id: Job identifier.

    Returns:
        Path to the output directory for the job.
    """
    return Path(settings["app"]["output_root"]) / job_id


def lora_logs_path(settings: Settings, job_id: str) -> Path:
    """Get the log file path for a LoRA training job.

    Args:
        settings: Application settings.
        job_id: Job identifier.

    Returns:
        Path to the log file for the job.
    """
    return Path(settings["app"]["logs_root"]) / f"{job_id}.log"


def dataset_dir(settings: Settings, job_id: str) -> Path:
    """Get the dataset directory for a training job.

    Args:
        settings: Application settings.
        job_id: Job identifier.

    Returns:
        Path to the dataset directory.
    """
    return Path(settings["app"]["data_root"]) / "datasets" / job_id


__all__ = [
    "dataset_dir",
    "lora_logs_path",
    "lora_output_dir",
]
