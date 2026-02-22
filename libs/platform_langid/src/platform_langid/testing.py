"""Public test utilities for platform_langid consumers.

Provides fake implementations and test helpers for services using platform_langid.

Usage:
    from platform_langid.testing import (
        FakeSpokenLanguageDetector,
        FakeTensor,
        reset_hooks,
    )

    # Set up fakes for testing
    from platform_langid import _test_hooks
    _test_hooks.detector_factory = lambda config: FakeSpokenLanguageDetector()

    # Reset to production after test
    reset_hooks()
"""

from __future__ import annotations

from collections.abc import Sequence

from . import _test_hooks
from .types import DetectorConfig, SpokenLanguageResult

# =============================================================================
# Fake Tensor
# =============================================================================


class FakeTensor:
    """Fake tensor for testing model operations.

    Simulates PyTorch tensor behavior without requiring torch.
    Supports both 1D tensor behavior and 2D batch dimension simulation.

    When is_batched=True, indexing with [0] returns a tensor with all values
    (simulating getting the first row of a 2D tensor).
    """

    __slots__ = ("_is_batched", "_values")

    def __init__(self, values: list[float], *, is_batched: bool = False) -> None:
        """Initialize fake tensor with values.

        Args:
            values: List of float values.
            is_batched: If True, first index accesses "row" (returns all values).
        """
        self._values = values
        self._is_batched = is_batched

    @property
    def shape(self) -> tuple[int, ...]:
        """Get tensor shape.

        Returns:
            Tuple representing tensor dimensions.
        """
        if self._is_batched:
            return (1, len(self._values))
        return (len(self._values),)

    def argmax(self, dim: int = -1) -> FakeTensor:
        """Return index of maximum value.

        Args:
            dim: Dimension (ignored for 1D).

        Returns:
            FakeTensor containing argmax index.
        """
        del dim
        max_idx = 0
        max_val = self._values[0]
        for i, v in enumerate(self._values):
            if v > max_val:
                max_val = v
                max_idx = i
        return FakeTensor([float(max_idx)])

    def softmax(self, dim: int = -1) -> FakeTensor:
        """Apply softmax to values.

        Args:
            dim: Dimension (ignored for 1D).

        Returns:
            FakeTensor with softmax probabilities, marked as batched
            to simulate 2D tensor behavior where [0] gets first row.
        """
        del dim
        import math

        max_val = max(self._values)
        exp_values = [math.exp(v - max_val) for v in self._values]
        total = sum(exp_values)
        softmax_values = [v / total for v in exp_values]
        return FakeTensor(softmax_values, is_batched=True)

    def item(self) -> float:
        """Extract single scalar value.

        Returns:
            First value as float.
        """
        return self._values[0]

    def __getitem__(self, index: int) -> FakeTensor:
        """Index into tensor.

        Args:
            index: Index to access.

        Returns:
            FakeTensor with indexed value(s).
            If batched and index is 0, returns all values (first row).
            Otherwise returns single indexed value.
        """
        if self._is_batched and index == 0:
            return FakeTensor(self._values, is_batched=False)
        return FakeTensor([self._values[index]])


# =============================================================================
# Fake Classification Output
# =============================================================================


class FakeClassificationOutput:
    """Fake classification output for testing."""

    __slots__ = ("logits",)

    def __init__(self, logits: FakeTensor) -> None:
        """Initialize with logits tensor.

        Args:
            logits: Logits tensor from model.
        """
        self.logits = logits


# =============================================================================
# Fake Processor Output
# =============================================================================


class FakeProcessorOutput:
    """Fake processor output for testing."""

    __slots__ = ("input_values",)

    def __init__(self, input_values: FakeTensor) -> None:
        """Initialize with input values tensor.

        Args:
            input_values: Input values tensor for model.
        """
        self.input_values = input_values


# =============================================================================
# Fake Model Config
# =============================================================================


class FakeModelConfig:
    """Fake model configuration for testing."""

    __slots__ = ("id2label",)

    def __init__(self, id2label: dict[int, str]) -> None:
        """Initialize with id to label mapping.

        Args:
            id2label: Mapping from class index to language label.
        """
        self.id2label = id2label


# =============================================================================
# Fake Model
# =============================================================================


class FakeModel:
    """Fake language identification model for testing.

    Returns configurable language detection results.
    """

    __slots__ = ("_logits", "_predicted_id", "config")

    def __init__(
        self,
        predicted_id: int = 0,
        confidence: float = 0.95,
        id2label: dict[int, str] | None = None,
    ) -> None:
        """Initialize fake model.

        Args:
            predicted_id: Class index to return as prediction.
            confidence: Confidence score for prediction (0.0-1.0).
            id2label: Label mapping. Defaults to {0: "en", 1: "vi", 2: "es"}.
        """
        self._predicted_id = predicted_id
        if id2label is None:
            id2label = {0: "en", 1: "vi", 2: "es"}
        self.config = FakeModelConfig(id2label)

        # Build logits that produce desired confidence after softmax
        # For softmax([L, 0, 0, ...]) where L is the logit for predicted class:
        # P = e^L / (e^L + (n-1))
        # Solving for L: L = log(P * (n-1) / (1 - P))
        num_classes = len(id2label)
        logits: list[float] = [0.0] * num_classes
        import math

        if num_classes == 1:
            # Single class: softmax always produces 1.0
            logits[0] = 0.0
        elif confidence >= 1.0:
            logits[predicted_id] = 100.0
        elif confidence <= 0.0:
            logits[predicted_id] = -100.0
        else:
            other_classes = num_classes - 1
            logits[predicted_id] = math.log(confidence * other_classes / (1 - confidence))
        self._logits = logits

    def __call__(self, input_values: _test_hooks.TensorProtocol) -> FakeClassificationOutput:
        """Run fake inference.

        Args:
            input_values: Input tensor (ignored).

        Returns:
            FakeClassificationOutput with configured logits.
        """
        del input_values
        return FakeClassificationOutput(FakeTensor(self._logits))

    def to(self, device: str) -> FakeModel:
        """Fake device transfer.

        Args:
            device: Target device (ignored).

        Returns:
            Self (no actual transfer).
        """
        del device
        return self


# =============================================================================
# Fake Processor
# =============================================================================


class FakeProcessor:
    """Fake audio processor for testing."""

    __slots__ = ()

    def __call__(
        self,
        audio: Sequence[float],
        sampling_rate: int,
        return_tensors: str = "pt",
    ) -> FakeProcessorOutput:
        """Process audio (returns fake output).

        Args:
            audio: Audio samples (ignored).
            sampling_rate: Sample rate (ignored).
            return_tensors: Tensor format (ignored).

        Returns:
            FakeProcessorOutput with dummy input values.
        """
        del audio, sampling_rate, return_tensors
        return FakeProcessorOutput(FakeTensor([0.0]))


# =============================================================================
# Fake Detector
# =============================================================================


class FakeSpokenLanguageDetector:
    """Fake spoken language detector for testing.

    Returns configurable detection results without running model inference.
    """

    __slots__ = ("_confidence", "_language", "_model_id", "call_count")

    def __init__(
        self,
        language: str = "en",
        confidence: float = 0.95,
        model_id: str = "fake-model",
    ) -> None:
        """Initialize fake detector.

        Args:
            language: Language code to return.
            confidence: Confidence score to return.
            model_id: Model ID to include in result.
        """
        self._language = language
        self._confidence = confidence
        self._model_id = model_id
        self.call_count = 0

    def detect(self, audio_bytes: bytes, sample_rate: int) -> SpokenLanguageResult:
        """Return configured detection result.

        Args:
            audio_bytes: Audio bytes (ignored, but must be non-empty).
            sample_rate: Sample rate (ignored).

        Returns:
            Configured SpokenLanguageResult.

        Raises:
            ValueError: If audio_bytes is empty.
        """
        del sample_rate
        if len(audio_bytes) == 0:
            raise ValueError("audio_bytes cannot be empty")
        self.call_count += 1
        return SpokenLanguageResult(
            language=self._language,
            confidence=self._confidence,
            model_id=self._model_id,
        )


# =============================================================================
# Fake Audio Loader
# =============================================================================


class FakeAudioLoader:
    """Fake audio loader for testing.

    Returns configurable audio samples without actual audio processing.
    """

    __slots__ = ("_samples", "calls")

    def __init__(self, samples: Sequence[float] | None = None) -> None:
        """Initialize fake audio loader.

        Args:
            samples: Samples to return. Defaults to [0.0, 0.1, -0.1].
        """
        if samples is None:
            samples = [0.0, 0.1, -0.1]
        self._samples = samples
        self.calls: list[tuple[bytes, int]] = []

    def __call__(self, audio_bytes: bytes, sample_rate: int) -> Sequence[float]:
        """Return configured samples.

        Args:
            audio_bytes: Audio bytes (recorded but not processed).
            sample_rate: Sample rate (recorded but not used).

        Returns:
            Configured audio samples.
        """
        self.calls.append((audio_bytes, sample_rate))
        return self._samples


# =============================================================================
# Hook Management
# =============================================================================


def set_production_hooks() -> None:
    """Set all hooks to production implementations."""
    _test_hooks.model_factory = _test_hooks._default_model_factory
    _test_hooks.processor_factory = _test_hooks._default_processor_factory
    _test_hooks.audio_loader = _test_hooks._default_audio_loader
    _test_hooks.detector_factory = _test_hooks._default_detector_factory


def reset_hooks() -> None:
    """Reset all hooks to production implementations."""
    set_production_hooks()


def make_fake_detector_factory(
    language: str = "en",
    confidence: float = 0.95,
    model_id: str = "fake-model",
) -> _test_hooks.DetectorFactoryProtocol:
    """Create a fake detector factory.

    Args:
        language: Language code to return from detector.
        confidence: Confidence score to return.
        model_id: Model ID to include in results.

    Returns:
        Factory function that creates FakeSpokenLanguageDetector.
    """

    def factory(config: DetectorConfig) -> _test_hooks.SpokenLanguageDetectorProtocol:
        del config
        return FakeSpokenLanguageDetector(
            language=language,
            confidence=confidence,
            model_id=model_id,
        )

    return factory


def make_fake_model_factory(
    predicted_id: int = 0,
    confidence: float = 0.95,
    id2label: dict[int, str] | None = None,
) -> _test_hooks.ModelFactoryProtocol:
    """Create a fake model factory.

    Args:
        predicted_id: Class index to predict.
        confidence: Confidence score for prediction.
        id2label: Label mapping.

    Returns:
        Factory function that creates FakeModel.
    """

    def factory(model_id: str) -> _test_hooks.ModelProtocol:
        del model_id
        return FakeModel(
            predicted_id=predicted_id,
            confidence=confidence,
            id2label=id2label,
        )

    return factory


def make_fake_processor_factory() -> _test_hooks.ProcessorFactoryProtocol:
    """Create a fake processor factory.

    Returns:
        Factory function that creates FakeProcessor.
    """

    def factory(model_id: str) -> _test_hooks.ProcessorProtocol:
        del model_id
        return FakeProcessor()

    return factory


__all__ = [
    "FakeAudioLoader",
    "FakeClassificationOutput",
    "FakeModel",
    "FakeModelConfig",
    "FakeProcessor",
    "FakeProcessorOutput",
    "FakeSpokenLanguageDetector",
    "FakeTensor",
    "make_fake_detector_factory",
    "make_fake_model_factory",
    "make_fake_processor_factory",
    "reset_hooks",
    "set_production_hooks",
]
