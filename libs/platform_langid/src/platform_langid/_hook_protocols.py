"""_test_hooks: ClassificationOutput and related definitions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .types import DetectorConfig, SpokenLanguageResult


class ClassificationOutput(Protocol):
    """Protocol for model classification output."""

    @property
    def logits(self) -> TensorProtocol:
        """Get logits tensor from classification output."""
        ...


class TensorProtocol(Protocol):
    """Protocol for tensor operations needed for classification."""

    def argmax(self, dim: int = -1) -> TensorProtocol:
        """Return indices of maximum values along dimension."""
        ...

    def softmax(self, dim: int = -1) -> TensorProtocol:
        """Apply softmax along dimension."""
        ...

    def item(self) -> float:
        """Extract single scalar value from tensor."""
        ...

    def __getitem__(self, index: int) -> TensorProtocol:
        """Index into tensor."""
        ...


class ProcessorProtocol(Protocol):
    """Protocol for audio feature processor."""

    def __call__(
        self,
        audio: Sequence[float],
        sampling_rate: int,
        return_tensors: str = "pt",
    ) -> ProcessorOutputProtocol:
        """Process audio waveform into model inputs.

        Args:
            audio: Audio samples as sequence of floats.
            sampling_rate: Sample rate of the audio.
            return_tensors: Format for return tensors ("pt" for PyTorch).

        Returns:
            Processed features ready for model input.
        """
        ...


class ProcessorOutputProtocol(Protocol):
    """Protocol for processor output with input_values."""

    @property
    def input_values(self) -> TensorProtocol:
        """Get input values tensor."""
        ...


class ModelProtocol(Protocol):
    """Protocol for language identification model."""

    @property
    def config(self) -> ModelConfigProtocol:
        """Get model configuration."""
        ...

    def __call__(self, input_values: TensorProtocol) -> ClassificationOutput:
        """Run inference on processed audio.

        Args:
            input_values: Processed audio features from processor.

        Returns:
            Classification output with logits.
        """
        ...

    def to(self, device: str) -> ModelProtocol:
        """Move model to specified device.

        Args:
            device: Target device ("cpu", "cuda", "mps").

        Returns:
            Model on target device.
        """
        ...


class ModelConfigProtocol(Protocol):
    """Protocol for model configuration with id2label mapping."""

    @property
    def id2label(self) -> dict[int, str]:
        """Get mapping from class index to label string."""
        ...


class _ModelClassProtocol(Protocol):
    """Protocol for model class with from_pretrained method."""

    def from_pretrained(self, model_id: str) -> ModelProtocol:
        """Load model from pretrained weights."""
        ...


class _ProcessorClassProtocol(Protocol):
    """Protocol for processor class with from_pretrained method."""

    def from_pretrained(self, model_id: str) -> ProcessorProtocol:
        """Load processor from pretrained configuration."""
        ...


class _TorchTensorProtocol(Protocol):
    """Protocol for torch tensor operations."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Get tensor shape."""
        ...

    def __getitem__(self, index: int) -> _TorchTensorProtocol:
        """Index into tensor."""
        ...

    def unsqueeze(self, dim: int) -> _TorchTensorProtocol:
        """Add dimension at specified position."""
        ...

    def squeeze(self, dim: int) -> _TorchTensorProtocol:
        """Remove dimension at specified position."""
        ...

    def tolist(self) -> list[float]:
        """Convert tensor to list of floats."""
        ...


class _ResamplerProtocol(Protocol):
    """Protocol for torchaudio resampler."""

    def __call__(self, tensor: _TorchTensorProtocol) -> _TorchTensorProtocol:
        """Resample audio tensor."""
        ...


class _ResampleClassProtocol(Protocol):
    """Protocol for torchaudio Resample class."""

    def __call__(self, source_rate: int, target_rate: int) -> _ResamplerProtocol:
        """Create resampler from source to target rate."""
        ...


# =============================================================================
# Langcodes Protocols
# =============================================================================


class _LanguageProtocol(Protocol):
    """Protocol for langcodes Language instance."""

    @property
    def language(self) -> str:
        """Get primary language subtag (ISO 639-1 or 639-3)."""
        ...

    def broader_tags(self) -> list[str]:
        """Get broader language tags for fallback matching.

        Returns:
            List of broader language codes that encompass this language.
        """
        ...


class _LanguageClassProtocol(Protocol):
    """Protocol for langcodes Language class."""

    def get(self, code: str) -> _LanguageProtocol:
        """Parse language code and return Language object.

        Args:
            code: Language code in any standard format (ISO 639-1, 639-3, etc.).

        Returns:
            Parsed Language object.
        """
        ...


# =============================================================================
# Detector Protocol
# =============================================================================


class SpokenLanguageDetectorProtocol(Protocol):
    """Protocol for spoken language detector."""

    def detect(self, audio_bytes: bytes, sample_rate: int) -> SpokenLanguageResult:
        """Detect spoken language from audio.

        Args:
            audio_bytes: Audio bytes in any format supported by torchaudio
                (MP3, WAV, webm, ogg, flac, etc.).
            sample_rate: Ignored - sample rate is detected from audio format.

        Returns:
            SpokenLanguageResult with detected language and confidence.
        """
        ...


# =============================================================================
# Factory Protocols
# =============================================================================


class ModelFactoryProtocol(Protocol):
    """Protocol for model factory function."""

    def __call__(self, model_id: str) -> ModelProtocol:
        """Load model from HuggingFace hub.

        Args:
            model_id: HuggingFace model identifier.

        Returns:
            Loaded model instance.
        """
        ...


class ProcessorFactoryProtocol(Protocol):
    """Protocol for processor factory function."""

    def __call__(self, model_id: str) -> ProcessorProtocol:
        """Load processor from HuggingFace hub.

        Args:
            model_id: HuggingFace model identifier.

        Returns:
            Loaded processor instance.
        """
        ...


class DetectorFactoryProtocol(Protocol):
    """Protocol for detector factory function."""

    def __call__(self, config: DetectorConfig) -> SpokenLanguageDetectorProtocol:
        """Create detector with given configuration.

        Args:
            config: Detector configuration.

        Returns:
            Configured detector instance.
        """
        ...


class AudioLoaderProtocol(Protocol):
    """Protocol for loading audio from bytes."""

    def __call__(self, audio_bytes: bytes, sample_rate: int) -> Sequence[float]:
        """Load and resample audio bytes to float samples.

        Args:
            audio_bytes: Raw audio bytes.
            sample_rate: Source sample rate in Hz.

        Returns:
            Audio samples as sequence of floats, resampled to 16kHz.
        """
        ...


# =============================================================================
# Default Implementations
# =============================================================================
