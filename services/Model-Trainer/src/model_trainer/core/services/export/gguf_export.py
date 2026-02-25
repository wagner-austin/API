"""GGUF export service for LoRA adapters.

This module provides functionality to export PEFT LoRA adapters to GGUF format
for use with llama.cpp backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from typing_extensions import TypedDict

from . import _test_hooks


class GgufExportResult(TypedDict):
    """Result of GGUF export operation.

    Attributes:
        output_path: Absolute path to the generated GGUF file.
        output_size_bytes: Size of the GGUF file in bytes.
    """

    output_path: str
    output_size_bytes: int


def export_lora_to_gguf(
    adapter_dir: str,
    base_model_id: str,
    output_dir: str,
    output_type: Literal["f32", "f16", "bf16", "q8_0"],
) -> GgufExportResult:
    """Export PEFT LoRA adapter to GGUF format.

    Uses llama.cpp's convert_lora_to_gguf.py script via test hook.
    The adapter directory must contain adapter_model.safetensors and
    adapter_config.json files from PEFT.

    Args:
        adapter_dir: Path to PEFT adapter directory containing saved weights.
        base_model_id: HuggingFace model ID of the base model used for training.
        output_dir: Directory where the GGUF file will be written.
        output_type: Output precision format for the GGUF file.

    Returns:
        GgufExportResult with output path and file size.

    Raises:
        RuntimeError: If the GGUF converter hook is not configured or
            if conversion fails.
    """
    output_path = str(Path(output_dir) / "adapter.gguf")

    output_size = _test_hooks.gguf_converter(
        adapter_dir=adapter_dir,
        base_model_id=base_model_id,
        output_path=output_path,
        output_type=output_type,
    )

    return {
        "output_path": output_path,
        "output_size_bytes": output_size,
    }


__all__ = [
    "GgufExportResult",
    "export_lora_to_gguf",
]
