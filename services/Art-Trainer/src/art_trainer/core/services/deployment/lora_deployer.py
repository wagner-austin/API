"""LoRA deployment service for Art-Trainer.

This module provides functionality to deploy trained LoRA files
to the ComfyUI models directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from art_trainer.core.config.settings import Settings

from . import _test_hooks


class DeploymentResult(TypedDict, total=True):
    """Result of LoRA deployment.

    Attributes:
        success: Whether deployment succeeded.
        source_path: Path to the source LoRA file.
        deployed_path: Path where LoRA was deployed.
        error_message: Error message if deployment failed.
    """

    success: bool
    source_path: str
    deployed_path: str | None
    error_message: str | None


def deploy_lora(
    settings: Settings,
    lora_path: Path,
    lora_name: str,
) -> DeploymentResult:
    """Deploy a trained LoRA to the ComfyUI models directory.

    Args:
        settings: Application settings with comfyui_lora_path.
        lora_path: Path to the trained LoRA file.
        lora_name: Name for the deployed LoRA file.

    Returns:
        DeploymentResult with deployment status.
    """
    # Validate source exists
    if not lora_path.exists():
        return {
            "success": False,
            "source_path": str(lora_path),
            "deployed_path": None,
            "error_message": f"Source LoRA not found: {lora_path}",
        }

    # Build destination path
    comfyui_lora_dir = Path(settings["app"]["comfyui_lora_path"])
    if not lora_name.endswith(".safetensors"):
        lora_name = f"{lora_name}.safetensors"
    dest_path = comfyui_lora_dir / lora_name

    copied_path = _test_hooks.Hooks.file_copier(lora_path, dest_path)

    return {
        "success": True,
        "source_path": str(lora_path),
        "deployed_path": str(copied_path),
        "error_message": None,
    }


__all__ = [
    "DeploymentResult",
    "deploy_lora",
]
