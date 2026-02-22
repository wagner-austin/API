"""Platform LangID - Spoken language identification from audio.

This library provides spoken language detection from audio waveforms using
Meta's MMS-LID model, which supports 4017 languages.

Usage:
    from platform_langid import detect_spoken_language, SpokenLanguageResult

    # Detect language from audio bytes
    result = detect_spoken_language(audio_bytes, sample_rate=16000)
    result["language"]  # "vi"
    result["confidence"]  # 0.94

    # For repeated detections, create a detector instance
    from platform_langid import create_detector, default_detector_config

    config = default_detector_config()
    detector = create_detector(config)
    result = detector.detect(audio_bytes, sample_rate=16000)
"""

from platform_langid.detector import (
    TARGET_SAMPLE_RATE,
    SpokenLanguageDetector,
    create_detector,
    detect_spoken_language,
)
from platform_langid.types import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_ID,
    AudioInput,
    DetectorConfig,
    SpokenLanguageResult,
    decode_audio_input,
    decode_detector_config,
    decode_spoken_language_result,
    default_detector_config,
    encode_audio_input,
    encode_detector_config,
    encode_spoken_language_result,
    require_audio_input,
    require_detector_config,
    require_spoken_language_result,
)

__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_DEVICE",
    "DEFAULT_MODEL_ID",
    "TARGET_SAMPLE_RATE",
    "AudioInput",
    "DetectorConfig",
    "SpokenLanguageDetector",
    "SpokenLanguageResult",
    "create_detector",
    "decode_audio_input",
    "decode_detector_config",
    "decode_spoken_language_result",
    "default_detector_config",
    "detect_spoken_language",
    "encode_audio_input",
    "encode_detector_config",
    "encode_spoken_language_result",
    "require_audio_input",
    "require_detector_config",
    "require_spoken_language_result",
]
