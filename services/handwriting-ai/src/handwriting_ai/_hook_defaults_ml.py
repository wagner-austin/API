"""Default (production) implementations for the ML hooks."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import torch
from PIL.Image import Image as PILImage
from platform_core.config import HandwritingAiSettings

from handwriting_ai._hook_protocols_ml import (
    InferencePoolProtocol,
    PreprocessOptionsDict,
    PreprocessOutputDict,
    ResourceLimitsDict,
)


def _default_detect_resource_limits() -> ResourceLimitsDict:
    """Production implementation - detects real resource limits."""
    from .training.resources import detect_resource_limits as _detect

    return _detect()


def _default_pil_image_open(fp: BinaryIO) -> PILImage:
    """Production implementation - opens image using PIL."""
    from PIL import Image

    return Image.open(fp)


def _default_run_preprocess(img: PILImage, opts: PreprocessOptionsDict) -> PreprocessOutputDict:
    """Production implementation - runs actual preprocessing."""
    from .preprocess import run_preprocess as _run_preprocess

    return _run_preprocess(img, opts)


def _default_preprocess_signature() -> str:
    """Production implementation - returns actual signature."""
    from .preprocess import preprocess_signature as _preprocess_signature

    return _preprocess_signature()


def _default_principal_angle_confidence(
    img: PILImage, width: int, height: int
) -> tuple[float, float] | None:
    """Production implementation - computes angle confidence."""
    from .preprocess import _principal_angle_confidence as _pac

    return _pac(img, width, height)


def _default_load_state_dict_file(path: Path) -> dict[str, torch.Tensor]:
    """Production implementation - loads state dict from file."""
    result: dict[str, torch.Tensor] = torch.load(path, map_location="cpu", weights_only=True)
    return result


def _default_validate_state_dict(sd: dict[str, torch.Tensor], arch: str, n_classes: int) -> None:
    """Production implementation - validates state dict structure."""
    from .inference.engine import _validate_state_dict as _validate

    _validate(sd, arch, n_classes)


def _default_make_inference_pool(settings: HandwritingAiSettings) -> InferencePoolProtocol:
    """Build the real bounded thread pool.

    The engine module is imported here rather than at module scope because it
    imports this one.

    Args:
        settings: Application settings.

    Returns:
        ThreadPoolExecutor sized from settings.
    """
    from .inference.engine import _make_pool

    return _make_pool(settings)


def _default_download_remote(
    settings: HandwritingAiSettings, model_dir: Path, manifest_path: Path
) -> None:
    """Fetch the remote artifact named by a v2 manifest.

    Args:
        settings: Application settings, carrying the data-bank config.
        model_dir: Directory the artifact belongs in.
        manifest_path: Manifest to read the file id from.
    """
    from .inference.engine import download_remote_artifact

    download_remote_artifact(settings, model_dir, manifest_path)


def _default_principal_angle(img: PILImage, width: int, height: int) -> float | None:
    """Production implementation."""
    from .preprocess import _principal_angle as _pa

    return _pa(img, width, height)


def _default_is_wrapped_state_dict(
    value: dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]],
) -> bool:
    """Production implementation - checks if state dict is wrapped."""
    return set(value.keys()) == {"state_dict"}


def _default_is_flat_state_dict(
    value: dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]],
) -> bool:
    """Production implementation - checks if state dict is flat."""
    return not _default_is_wrapped_state_dict(value)


def _default_pil_histogram(img: PILImage) -> list[int]:
    """Production implementation - calls PIL histogram."""
    return img.histogram()
