"""Kohya_ss training backend adapter.

This module implements the TrainingBackend protocol using Kohya_ss scripts.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from art_trainer.core.config.settings import Settings
from art_trainer.core.contracts.backend import CancelledCheck, ProgressCallback
from art_trainer.core.contracts.lora import LoraTrainConfig, LoraTrainOutcome
from art_trainer.core.contracts.progress import ArtTrainingProgress

from .config import build_kohya_config, write_kohya_config
from .runner import run_subprocess

# Regex to extract loss from Kohya output
LOSS_PATTERN = re.compile(r"loss[=:]\s*([\d.]+)", re.IGNORECASE)
STEP_PATTERN = re.compile(r"step[s]?\s*[=:]\s*(\d+)", re.IGNORECASE)


class KohyaBackend:
    """Training backend using Kohya_ss scripts."""

    _settings: Settings

    def __init__(self, settings: Settings) -> None:
        """Initialize Kohya backend.

        Args:
            settings: Application settings.
        """
        self._settings = settings

    def name(self) -> str:
        """Get the backend name.

        Returns:
            Human-readable backend name.
        """
        return "kohya_ss"

    def is_available(self) -> bool:
        """Check if the backend is available for use.

        Checks if the Kohya_ss path exists and contains required scripts.

        Returns:
            True if the backend is available, False otherwise.
        """
        kohya_path = Path(self._settings["app"]["kohya_ss_path"])
        train_script = kohya_path / "train_network.py"
        return kohya_path.is_dir() and train_script.is_file()

    def train(
        self,
        config: LoraTrainConfig,
        *,
        progress_callback: ProgressCallback | None = None,
        cancelled: CancelledCheck | None = None,
    ) -> LoraTrainOutcome:
        """Execute LoRA training using Kohya_ss.

        Args:
            config: Training configuration.
            progress_callback: Optional callback for progress reporting.
            cancelled: Optional callback to check for cancellation.

        Returns:
            Training outcome with success status and results.
        """
        # Check cancellation before starting
        if cancelled is not None and cancelled():
            return _cancelled_outcome()

        # Report initial progress
        if progress_callback is not None:
            progress_callback(_make_progress(config["job_id"], "preparing", 0, config["steps"]))

        # Build and write config
        kohya_config = build_kohya_config(config)
        config_path = Path(config["output_dir"]) / "config.toml"
        write_kohya_config(kohya_config, config_path)

        # Check cancellation after config
        if cancelled is not None and cancelled():
            return _cancelled_outcome()

        # Report training start
        if progress_callback is not None:
            progress_callback(_make_progress(config["job_id"], "training", 0, config["steps"]))

        # Run training
        kohya_path = Path(self._settings["app"]["kohya_ss_path"])
        train_script = kohya_path / "train_network.py"

        args = [
            "python",
            str(train_script),
            "--config_file",
            str(config_path),
        ]

        result = run_subprocess(args, cwd=kohya_path, timeout=86400)

        # Check for cancellation or failure
        if cancelled is not None and cancelled():
            return _cancelled_outcome()

        if result.returncode != 0:
            return {
                "success": False,
                "lora_path": None,
                "final_loss": None,
                "error_message": f"Training failed with code {result.returncode}: {result.stderr}",
            }

        # Extract final loss from output
        final_loss = _extract_final_loss(result.stdout)

        # Find output file
        output_dir = Path(config["output_dir"])
        lora_files = list(output_dir.glob("*.safetensors"))
        lora_path = str(lora_files[0]) if lora_files else None

        # Report completion
        if progress_callback is not None:
            progress_callback(
                _make_progress(
                    config["job_id"],
                    "completed",
                    config["steps"],
                    config["steps"],
                    loss=final_loss,
                )
            )

        return {
            "success": True,
            "lora_path": lora_path,
            "final_loss": final_loss,
            "error_message": None,
        }


def _make_progress(
    job_id: str,
    phase: str,
    step: int,
    total_steps: int,
    *,
    loss: float | None = None,
) -> ArtTrainingProgress:
    """Create a progress update.

    Args:
        job_id: Job identifier.
        phase: Current phase.
        step: Current step.
        total_steps: Total steps.
        loss: Current loss value.

    Returns:
        ArtTrainingProgress instance.
    """
    from art_trainer.core.contracts.progress import ArtTrainingPhase

    # Narrow the phase string
    phase_typed: ArtTrainingPhase
    if phase == "queued":
        phase_typed = "queued"
    elif phase == "preparing":
        phase_typed = "preparing"
    elif phase == "training":
        phase_typed = "training"
    elif phase == "saving":
        phase_typed = "saving"
    elif phase == "uploading":
        phase_typed = "uploading"
    elif phase == "completed":
        phase_typed = "completed"
    elif phase == "failed":
        phase_typed = "failed"
    elif phase == "cancelled":
        phase_typed = "cancelled"
    else:
        phase_typed = "training"

    return {
        "job_id": job_id,
        "phase": phase_typed,
        "step": step,
        "total_steps": total_steps,
        "loss": loss,
        "learning_rate": 0.0,
        "updated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _cancelled_outcome() -> LoraTrainOutcome:
    """Create a cancelled outcome.

    Returns:
        LoraTrainOutcome indicating cancellation.
    """
    return {
        "success": False,
        "lora_path": None,
        "final_loss": None,
        "error_message": "Training cancelled by user",
    }


def _extract_final_loss(output: str) -> float | None:
    """Extract the final loss value from training output.

    Args:
        output: Training stdout output.

    Returns:
        Final loss value or None if not found.
    """
    matches: list[str] = LOSS_PATTERN.findall(output)
    if len(matches) > 0:
        last_loss: str = matches[-1]
        return float(last_loss)
    return None


__all__ = [
    "KohyaBackend",
]
