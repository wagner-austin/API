"""Spoken language detection from audio waveforms.

This module provides the core detection logic using MMS-LID models.
Uses dependency injection via _test_hooks for testability.
"""

from __future__ import annotations

from typing import Final

from . import _test_hooks
from .types import DetectorConfig, SpokenLanguageResult

# Target sample rate for MMS-LID models
TARGET_SAMPLE_RATE: Final[int] = 16000


class SpokenLanguageDetector:
    """Spoken language detector using MMS-LID model.

    Detects the language spoken in audio using Meta's MMS-LID model.
    Supports 4017 languages with ISO 639-3 code output.
    """

    __slots__ = ("_config", "_model", "_processor")

    def __init__(self, config: DetectorConfig) -> None:
        """Initialize detector with configuration.

        Args:
            config: Detector configuration including model_id and device.
        """
        self._config = config
        self._model = _test_hooks.model_factory(config["model_id"])
        self._model = self._model.to(config["device"])
        self._processor = _test_hooks.processor_factory(config["model_id"])

    def detect(self, audio_bytes: bytes, sample_rate: int) -> SpokenLanguageResult:
        """Detect spoken language from audio.

        Args:
            audio_bytes: Raw audio bytes (16-bit PCM).
            sample_rate: Sample rate of the audio in Hz.

        Returns:
            SpokenLanguageResult with detected language, confidence, and model_id.

        Raises:
            ValueError: If audio_bytes is empty.
        """
        if len(audio_bytes) == 0:
            raise ValueError("audio_bytes cannot be empty")

        # Load and resample audio to 16kHz
        audio_samples = _test_hooks.audio_loader(audio_bytes, sample_rate)

        # Process audio through feature extractor
        inputs = self._processor(
            audio_samples,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )

        # Run inference
        outputs = self._model(inputs.input_values)
        logits = outputs.logits

        # Get predicted class and confidence
        predicted_id = logits.argmax(dim=-1).item()
        probabilities = logits.softmax(dim=-1)
        confidence = probabilities[0][int(predicted_id)].item()

        # Get language label from model config (ISO 639-3)
        label_639_3 = self._model.config.id2label[int(predicted_id)]

        # Convert to Whisper-compatible code (ISO 639-1)
        # Returns None if language not supported by Whisper
        whisper_code = _test_hooks.convert_language_code(label_639_3)

        # Apply confidence threshold and handle unsupported languages
        # "auto" means let Whisper auto-detect (for unsupported languages)
        threshold = self._config["confidence_threshold"]
        if confidence < threshold or whisper_code is None:
            language = "auto"
        else:
            language = whisper_code

        return SpokenLanguageResult(
            language=language,
            confidence=confidence,
            model_id=self._config["model_id"],
        )


def detect_spoken_language(
    audio_bytes: bytes,
    sample_rate: int = TARGET_SAMPLE_RATE,
    config: DetectorConfig | None = None,
) -> SpokenLanguageResult:
    """Detect spoken language from audio bytes.

    Convenience function that creates a detector and runs detection.
    For repeated detections, create a SpokenLanguageDetector instance
    to avoid reloading the model each time.

    Args:
        audio_bytes: Raw audio bytes (16-bit PCM).
        sample_rate: Sample rate of the audio in Hz. Defaults to 16000.
        config: Optional detector configuration. Uses defaults if not provided.

    Returns:
        SpokenLanguageResult with detected language, confidence, and model_id.

    Raises:
        ValueError: If audio_bytes is empty.
    """
    if config is None:
        from .types import default_detector_config

        config = default_detector_config()

    detector = _test_hooks.detector_factory(config)
    return detector.detect(audio_bytes, sample_rate)


def create_detector(
    config: DetectorConfig | None = None,
) -> _test_hooks.SpokenLanguageDetectorProtocol:
    """Create a spoken language detector instance.

    Args:
        config: Optional detector configuration. Uses defaults if not provided.

    Returns:
        Configured detector instance ready for detection.
    """
    if config is None:
        from .types import default_detector_config

        config = default_detector_config()

    return _test_hooks.detector_factory(config)


__all__ = [
    "TARGET_SAMPLE_RATE",
    "SpokenLanguageDetector",
    "create_detector",
    "detect_spoken_language",
]
