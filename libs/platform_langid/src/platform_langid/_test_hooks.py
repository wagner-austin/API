"""Internal test hooks for platform_langid.

The protocols describing the model surface live in ``_hook_protocols`` and the
production implementations in ``_hook_defaults``; this module is where each hook
is bound to its real implementation. Production reads and tests rebind through
this module, so each binding must live here rather than in the module that
defines the default.
"""

from __future__ import annotations

from platform_langid._hook_defaults import (
    WHISPER_SUPPORTED_CODES,
    LanguageCodeConverterProtocol,
    _default_audio_loader,
    _default_convert_language_code,
    _default_detector_factory,
    _default_model_factory,
    _default_processor_factory,
)
from platform_langid._hook_protocols import (
    AudioLoaderProtocol,
    ClassificationOutput,
    DetectorFactoryProtocol,
    ModelConfigProtocol,
    ModelFactoryProtocol,
    ModelProtocol,
    ProcessorFactoryProtocol,
    ProcessorOutputProtocol,
    ProcessorProtocol,
    SpokenLanguageDetectorProtocol,
    TensorProtocol,
)

# Hook for model construction.
model_factory: ModelFactoryProtocol = _default_model_factory

# Hook for processor construction.
processor_factory: ProcessorFactoryProtocol = _default_processor_factory

# Hook for decoding audio bytes into samples.
audio_loader: AudioLoaderProtocol = _default_audio_loader

# Hook for detector construction.
detector_factory: DetectorFactoryProtocol = _default_detector_factory

# Hook for language code conversion.
convert_language_code: LanguageCodeConverterProtocol = _default_convert_language_code


__all__ = [
    "WHISPER_SUPPORTED_CODES",
    "AudioLoaderProtocol",
    "ClassificationOutput",
    "DetectorFactoryProtocol",
    "LanguageCodeConverterProtocol",
    "ModelConfigProtocol",
    "ModelFactoryProtocol",
    "ModelProtocol",
    "ProcessorFactoryProtocol",
    "ProcessorOutputProtocol",
    "ProcessorProtocol",
    "SpokenLanguageDetectorProtocol",
    "TensorProtocol",
    "_default_audio_loader",
    "_default_convert_language_code",
    "_default_detector_factory",
    "_default_model_factory",
    "_default_processor_factory",
    "audio_loader",
    "convert_language_code",
    "detector_factory",
    "model_factory",
    "processor_factory",
]
