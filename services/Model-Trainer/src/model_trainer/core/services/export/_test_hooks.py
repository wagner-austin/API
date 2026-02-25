"""Test hooks for GGUF export service.

Follows the covenant pattern: production code sets hooks to real implementations,
tests set hooks to fakes for isolation.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal, Protocol


class GgufConverterProto(Protocol):
    """Protocol for GGUF converter.

    Implementations convert a PEFT LoRA adapter to GGUF format using
    llama.cpp's convert_lora_to_gguf.py script.
    """

    def __call__(
        self,
        adapter_dir: str,
        base_model_id: str,
        output_path: str,
        output_type: Literal["f32", "f16", "bf16", "q8_0"],
    ) -> int:
        """Convert adapter to GGUF format.

        Args:
            adapter_dir: Path to PEFT adapter directory containing
                adapter_model.safetensors and adapter_config.json.
            base_model_id: HuggingFace model ID of the base model
                used for the adapter.
            output_path: Destination path for the GGUF file.
            output_type: Output precision format for the GGUF file.

        Returns:
            Size of the generated GGUF file in bytes.
        """
        ...


class ConvertScriptPathsProto(Protocol):
    """Protocol for getting convert script search paths."""

    def __call__(self) -> tuple[Path, ...]:
        """Return tuple of paths to search for convert script.

        Returns:
            Tuple of Path objects to check in order.
        """
        ...


def _default_convert_script_paths() -> tuple[Path, ...]:
    """Production convert script paths - known installation locations.

    Returns:
        Tuple of paths to check for convert_lora_to_gguf.py script.
    """
    return (
        Path.home() / "PROJECTS" / "llama.cpp-src" / "convert_lora_to_gguf.py",
        Path.home() / "llama.cpp" / "convert_lora_to_gguf.py",
        Path("/opt/llama.cpp/convert_lora_to_gguf.py"),
    )


# Hook for getting convert script paths - tests can override to control search.
convert_script_paths: ConvertScriptPathsProto = _default_convert_script_paths


def _find_convert_script() -> str:
    """Find llama.cpp convert_lora_to_gguf.py script in known locations.

    Uses convert_script_paths hook to get search paths.

    Returns:
        Absolute path to the convert script.

    Raises:
        RuntimeError: If script cannot be found in any known location.
    """
    paths = convert_script_paths()
    for path in paths:
        if path.exists():
            return str(path)

    paths_str = ", ".join(str(p) for p in paths)
    raise RuntimeError(
        f"llama.cpp convert_lora_to_gguf.py script not found. "
        f"Checked: {paths_str}. "
        f"Clone llama.cpp to ~/PROJECTS/llama.cpp-src/"
    )


def _real_gguf_converter(
    adapter_dir: str,
    base_model_id: str,
    output_path: str,
    output_type: Literal["f32", "f16", "bf16", "q8_0"],
) -> int:
    """Production GGUF converter implementation.

    Uses llama.cpp's convert_lora_to_gguf.py script via subprocess.

    Args:
        adapter_dir: Path to PEFT adapter directory.
        base_model_id: HuggingFace model ID of the base model.
        output_path: Destination path for the GGUF file.
        output_type: Output precision format.

    Returns:
        Size of the generated GGUF file in bytes.

    Raises:
        RuntimeError: If conversion fails or script not found.
    """
    script_path = _find_convert_script()

    cmd = [
        "python",
        script_path,
        adapter_dir,
        "--base-model-id",
        base_model_id,
        "--outfile",
        output_path,
        "--outtype",
        output_type,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed: {result.stderr}")

    output_file = Path(output_path)
    return output_file.stat().st_size


# Hook for GGUF converter - initialized to production implementation.
# Tests replace this with fakes before calling export code.
gguf_converter: GgufConverterProto = _real_gguf_converter


def reset_hooks() -> None:
    """Reset all hooks to production defaults (for test cleanup)."""
    global gguf_converter, convert_script_paths
    gguf_converter = _real_gguf_converter
    convert_script_paths = _default_convert_script_paths


__all__ = [
    "ConvertScriptPathsProto",
    "GgufConverterProto",
    "convert_script_paths",
    "gguf_converter",
    "reset_hooks",
]
