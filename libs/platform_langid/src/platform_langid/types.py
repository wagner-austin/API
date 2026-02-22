"""Type definitions for platform_langid.

Provides TypedDict schemas with encode/decode/require_* validation for
spoken language detection results.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_float,
    require_str,
)
from typing_extensions import TypedDict


class SpokenLanguageResult(TypedDict):
    """Result from spoken language detection on audio.

    Attributes:
        language: ISO 639-1 language code (e.g., "vi", "en", "es").
        confidence: Confidence score between 0.0 and 1.0.
        model_id: Model identifier used for detection (e.g., "facebook/mms-lid-4017").
    """

    language: str
    confidence: float
    model_id: str


def encode_spoken_language_result(result: SpokenLanguageResult) -> JSONObject:
    """Encode SpokenLanguageResult to JSON-compatible dict.

    Args:
        result: The result to encode.

    Returns:
        JSON-compatible dictionary.
    """
    return {
        "language": result["language"],
        "confidence": result["confidence"],
        "model_id": result["model_id"],
    }


def decode_spoken_language_result(obj: JSONObject) -> SpokenLanguageResult:
    """Decode JSON object to SpokenLanguageResult with validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated SpokenLanguageResult.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
        ValueError: If confidence is outside valid range.
    """
    language = require_str(obj, "language")
    confidence = require_float(obj, "confidence")
    model_id = require_str(obj, "model_id")

    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"confidence must be between 0.0 and 1.0, got {confidence}")

    return SpokenLanguageResult(
        language=language,
        confidence=confidence,
        model_id=model_id,
    )


def require_spoken_language_result(obj: JSONValue) -> SpokenLanguageResult:
    """Validate and convert JSONValue to SpokenLanguageResult.

    Args:
        obj: JSON value to validate.

    Returns:
        Validated SpokenLanguageResult.

    Raises:
        JSONTypeError: If validation fails.
        ValueError: If semantic validation fails.
    """
    if not isinstance(obj, dict):
        raise JSONTypeError(f"Expected object, got {type(obj).__name__}")
    return decode_spoken_language_result(obj)


class AudioInput(TypedDict):
    """Audio input for language detection.

    Attributes:
        waveform: Audio samples as bytes (16-bit PCM or float32).
        sample_rate: Sample rate in Hz.
        format: Audio format identifier.
    """

    waveform: bytes
    sample_rate: int
    format: str


def encode_audio_input(audio: AudioInput) -> JSONObject:
    """Encode AudioInput to JSON-compatible dict.

    Note: waveform bytes are base64 encoded for JSON transport.

    Args:
        audio: The audio input to encode.

    Returns:
        JSON-compatible dictionary with base64-encoded waveform.
    """
    import base64

    waveform_b64 = base64.b64encode(audio["waveform"]).decode("ascii")
    return {
        "waveform": waveform_b64,
        "sample_rate": audio["sample_rate"],
        "format": audio["format"],
    }


def decode_audio_input(obj: JSONObject) -> AudioInput:
    """Decode JSON object to AudioInput with validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated AudioInput.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
        ValueError: If sample_rate is invalid.
    """
    import base64

    from platform_core.json_utils import require_int

    waveform_b64 = require_str(obj, "waveform")
    sample_rate = require_int(obj, "sample_rate")
    format_str = require_str(obj, "format")

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    waveform = base64.b64decode(waveform_b64)

    return AudioInput(
        waveform=waveform,
        sample_rate=sample_rate,
        format=format_str,
    )


def require_audio_input(obj: JSONValue) -> AudioInput:
    """Validate and convert JSONValue to AudioInput.

    Args:
        obj: JSON value to validate.

    Returns:
        Validated AudioInput.

    Raises:
        JSONTypeError: If validation fails.
        ValueError: If semantic validation fails.
    """
    if not isinstance(obj, dict):
        raise JSONTypeError(f"Expected object, got {type(obj).__name__}")
    return decode_audio_input(obj)


class DetectorConfig(TypedDict):
    """Configuration for spoken language detector.

    Attributes:
        model_id: HuggingFace model identifier.
        device: Device to run inference on ("cpu", "cuda", "mps").
        confidence_threshold: Minimum confidence to report a language.
    """

    model_id: str
    device: str
    confidence_threshold: float


def encode_detector_config(config: DetectorConfig) -> JSONObject:
    """Encode DetectorConfig to JSON-compatible dict.

    Args:
        config: The config to encode.

    Returns:
        JSON-compatible dictionary.
    """
    return {
        "model_id": config["model_id"],
        "device": config["device"],
        "confidence_threshold": config["confidence_threshold"],
    }


def decode_detector_config(obj: JSONObject) -> DetectorConfig:
    """Decode JSON object to DetectorConfig with validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated DetectorConfig.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
        ValueError: If confidence_threshold is outside valid range.
    """
    model_id = require_str(obj, "model_id")
    device = require_str(obj, "device")
    confidence_threshold = require_float(obj, "confidence_threshold")

    if confidence_threshold < 0.0 or confidence_threshold > 1.0:
        raise ValueError(
            f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
        )

    return DetectorConfig(
        model_id=model_id,
        device=device,
        confidence_threshold=confidence_threshold,
    )


def require_detector_config(obj: JSONValue) -> DetectorConfig:
    """Validate and convert JSONValue to DetectorConfig.

    Args:
        obj: JSON value to validate.

    Returns:
        Validated DetectorConfig.

    Raises:
        JSONTypeError: If validation fails.
        ValueError: If semantic validation fails.
    """
    if not isinstance(obj, dict):
        raise JSONTypeError(f"Expected object, got {type(obj).__name__}")
    return decode_detector_config(obj)


# Default model configuration
DEFAULT_MODEL_ID = "facebook/mms-lid-4017"
DEFAULT_DEVICE = "cpu"
DEFAULT_CONFIDENCE_THRESHOLD = 0.0


def default_detector_config() -> DetectorConfig:
    """Create default detector configuration.

    Returns:
        DetectorConfig with default values.
    """
    return DetectorConfig(
        model_id=DEFAULT_MODEL_ID,
        device=DEFAULT_DEVICE,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    )


__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_DEVICE",
    "DEFAULT_MODEL_ID",
    "AudioInput",
    "DetectorConfig",
    "SpokenLanguageResult",
    "decode_audio_input",
    "decode_detector_config",
    "decode_spoken_language_result",
    "default_detector_config",
    "encode_audio_input",
    "encode_detector_config",
    "encode_spoken_language_result",
    "require_audio_input",
    "require_detector_config",
    "require_spoken_language_result",
]
