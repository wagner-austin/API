"""ML hook protocols (torch, preprocessing, inference, model artifacts)."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import BinaryIO, Protocol, Self, TypedDict

import torch
from PIL.Image import Image as PILImage
from platform_core.config import HandwritingAiSettings

from handwriting_ai.inference.types import PredictOutput


class LoadStateResultProtocol(Protocol):
    """Protocol for load_state_dict return value."""

    @property
    def missing_keys(self) -> tuple[str, ...] | Sequence[str]: ...

    @property
    def unexpected_keys(self) -> tuple[str, ...] | Sequence[str]: ...


class AugmentKnobsDict(TypedDict):
    """TypedDict mirroring _AugmentKnobs from dataset.py to avoid circular imports."""

    enable: bool
    rotate_deg: float
    translate_frac: float
    noise_prob: float
    noise_salt_vs_pepper: float
    dots_prob: float
    dots_count: int
    dots_size_px: int
    blur_sigma: float
    morph_mode: str
    morph_kernel_px: int


class _PixelAccessProtocol(Protocol):
    """Protocol for PIL pixel access."""

    def __getitem__(self, xy: tuple[int, int]) -> int: ...


class TorchCudaIsAvailableProtocol(Protocol):
    """Protocol for torch.cuda.is_available."""

    def __call__(self) -> bool: ...


class TorchCudaCurrentDeviceProtocol(Protocol):
    """Protocol for torch.cuda.current_device."""

    def __call__(self) -> int: ...


class TorchCudaMemoryProtocol(Protocol):
    """Protocol for torch.cuda memory functions."""

    def __call__(self, device: int) -> int: ...


class TorchCudaEmptyCacheProtocol(Protocol):
    """Protocol for torch.cuda.empty_cache."""

    def __call__(self) -> None: ...


class InteropConfiguredGetterProtocol(Protocol):
    """Protocol for getting _INTEROP_CONFIGURED state."""

    def __call__(self) -> bool: ...


class InteropConfiguredSetterProtocol(Protocol):
    """Protocol for setting _INTEROP_CONFIGURED state."""

    def __call__(self, value: bool) -> None: ...


class ResourceLimitsDict(TypedDict):
    """Resource limits returned by detect_resource_limits."""

    cpu_cores: int
    memory_bytes: int | None
    optimal_threads: int
    optimal_workers: int
    max_batch_size: int | None


class DetectResourceLimitsProtocol(Protocol):
    """Protocol for detect_resource_limits."""

    def __call__(self) -> ResourceLimitsDict:
        """Detect resource limits."""
        ...


class PILImageOpenProtocol(Protocol):
    """Protocol for PIL.Image.open."""

    def __call__(self, fp: BinaryIO) -> PILImage: ...


class PreprocessOptionsDict(TypedDict):
    """Options for preprocessing (matches preprocess.PreprocessOptions)."""

    invert: bool | None
    center: bool
    visualize: bool
    visualize_max_kb: int


class PreprocessOutputDict(TypedDict):
    """Output from preprocessing (matches inference.types.PreprocessOutput)."""

    tensor: torch.Tensor
    visual_png: bytes | None


class RunPreprocessProtocol(Protocol):
    """Protocol for run_preprocess function."""

    def __call__(self, img: PILImage, opts: PreprocessOptionsDict) -> PreprocessOutputDict: ...


class PreprocessSignatureProtocol(Protocol):
    """Protocol for preprocess_signature function."""

    def __call__(self) -> str: ...


class PrincipalAngleConfidenceProtocol(Protocol):
    """Protocol for _principal_angle_confidence function."""

    def __call__(self, img: PILImage, width: int, height: int) -> tuple[float, float] | None: ...


class LoadStateDictFileProtocol(Protocol):
    """Protocol for loading state dict from file."""

    def __call__(self, path: Path) -> dict[str, torch.Tensor]: ...


class ValidateStateDictProtocol(Protocol):
    """Protocol for validating state dict."""

    def __call__(self, sd: dict[str, torch.Tensor], arch: str, n_classes: int) -> None: ...


class PredictImplProtocol(Protocol):
    """Protocol for the engine's per-request inference implementation."""

    def __call__(self, preprocessed: torch.Tensor) -> PredictOutput:
        """Run inference on one preprocessed image.

        Args:
            preprocessed: Preprocessed image tensor.

        Returns:
            Prediction for that image.
        """
        ...


class InferencePoolProtocol(Protocol):
    """Protocol for the executor the engine submits inference to."""

    def submit(self, fn: PredictImplProtocol, preprocessed: torch.Tensor) -> Future[PredictOutput]:
        """Schedule one inference and return its future.

        Args:
            fn: Implementation to run.
            preprocessed: Preprocessed image tensor.

        Returns:
            Future carrying the prediction.
        """
        ...


class MakeInferencePoolProtocol(Protocol):
    """Protocol for the inference pool factory."""

    def __call__(self, settings: HandwritingAiSettings) -> InferencePoolProtocol:
        """Build the pool the engine submits inference to.

        Args:
            settings: Application settings.

        Returns:
            Pool sized from settings.
        """
        ...


class DownloadRemoteProtocol(Protocol):
    """Protocol for fetching a v2 manifest's remote model artifact."""

    def __call__(
        self, settings: HandwritingAiSettings, model_dir: Path, manifest_path: Path
    ) -> None:
        """Download the artifact the manifest names, if it names one.

        Args:
            settings: Application settings, carrying the data-bank config.
            model_dir: Directory the artifact belongs in.
            manifest_path: Manifest to read the file id from.
        """
        ...


class InferenceTorchModelProtocol(Protocol):
    """Protocol for torch model used in inference."""

    def eval(self) -> Self: ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> LoadStateResultProtocol: ...

    def train(self, mode: bool = True) -> Self: ...

    def state_dict(self) -> dict[str, torch.Tensor]: ...

    def parameters(self) -> Sequence[torch.nn.Parameter]: ...


class PreprocessDatasetProtocol(Protocol):
    """Protocol for PreprocessDataset to avoid circular imports.

    Mirrors the interface of handwriting_ai.training.dataset.PreprocessDataset
    without importing it.
    """

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...

    @property
    def knobs(self) -> AugmentKnobsDict:
        """Expose augmentation knobs for runtime access."""
        ...


class PrincipalAngleProtocol(Protocol):
    """Protocol for _principal_angle function."""

    def __call__(self, img: PILImage, width: int, height: int) -> float | None: ...


class IsWrappedStateDictProtocol(Protocol):
    """Protocol for _is_wrapped_state_dict type guard."""

    def __call__(
        self, value: dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]]
    ) -> bool: ...


class IsFlatStateDictProtocol(Protocol):
    """Protocol for _is_flat_state_dict type guard."""

    def __call__(
        self, value: dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]]
    ) -> bool: ...


class TorchModelProtocol(Protocol):
    """Protocol for torch.nn.Module used in training."""

    def train(self, mode: bool = True) -> TorchModelProtocol: ...

    def eval(self) -> TorchModelProtocol: ...

    def parameters(self) -> Generator[torch.Tensor, None, None]: ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class PILHistogramProtocol(Protocol):
    """Protocol for PIL Image histogram method."""

    def __call__(self, img: PILImage) -> list[int]: ...


class FakeImageForPrincipalAngleProtocol(Protocol):
    """Protocol for fake images used in _principal_angle tests.

    This is the minimal interface that _principal_angle and
    _principal_angle_confidence need from an image.
    """

    def load(self) -> _PixelAccessProtocol | None:
        """Return pixel access or None to test defensive branch."""
        ...
