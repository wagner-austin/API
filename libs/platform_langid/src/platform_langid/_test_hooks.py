"""Internal test hooks for platform_langid - allows injecting test dependencies.

This module provides dependency injection hooks following the pattern:
- Production code sets hooks to real implementations at startup
- Tests set hooks to fakes before running

Usage in production:
    # At startup, hooks are already set to defaults (production implementations)

Usage in tests:
    from platform_langid import _test_hooks
    _test_hooks.model_factory = fake_model_factory
    # ... run test ...
    # Reset after test if needed
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .types import DetectorConfig, SpokenLanguageResult

# =============================================================================
# Model Protocol
# =============================================================================


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


def _default_model_factory(model_id: str) -> ModelProtocol:
    """Production implementation - loads model from HuggingFace."""
    mod = __import__("transformers", fromlist=["Wav2Vec2ForSequenceClassification"])
    model_cls: _ModelClassProtocol = mod.Wav2Vec2ForSequenceClassification
    model: ModelProtocol = model_cls.from_pretrained(model_id)
    return model


def _default_processor_factory(model_id: str) -> ProcessorProtocol:
    """Production implementation - loads processor from HuggingFace."""
    mod = __import__("transformers", fromlist=["AutoFeatureExtractor"])
    processor_cls: _ProcessorClassProtocol = mod.AutoFeatureExtractor
    processor: ProcessorProtocol = processor_cls.from_pretrained(model_id)
    return processor


def _detect_audio_format(audio_bytes: bytes) -> str:
    """Detect audio format from magic bytes.

    Args:
        audio_bytes: Audio file bytes.

    Returns:
        Format string for torchaudio (wav, mp3, ogg, flac, webm).

    Raises:
        ValueError: If format cannot be detected.
    """
    if len(audio_bytes) < 12:
        raise ValueError("Audio data too short to detect format")

    # Check magic bytes
    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return "wav"
    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        return "mp3"
    if audio_bytes[:4] == b"OggS":
        return "ogg"
    if audio_bytes[:4] == b"fLaC":
        return "flac"
    if audio_bytes[:4] == b"\x1a\x45\xdf\xa3":  # EBML header (webm/mkv)
        return "webm"

    raise ValueError("Unknown audio format")


class _PathProtocol(Protocol):
    """Protocol for Path-like objects."""

    def __fspath__(self) -> str:
        """Return filesystem path as string."""
        ...


class _TorchaudioLoadFileProtocol(Protocol):
    """Protocol for torchaudio.load with file path."""

    def __call__(self, filepath: _PathProtocol) -> tuple[_TorchTensorProtocol, int]:
        """Load audio from file path.

        Args:
            filepath: Path to audio file.

        Returns:
            Tuple of (waveform tensor, sample rate).
        """
        ...


def _default_audio_loader(audio_bytes: bytes, sample_rate: int) -> Sequence[float]:
    """Production implementation - loads and resamples audio.

    Decodes audio from various formats (MP3, WAV, webm, etc.) using torchaudio.
    The sample_rate parameter is ignored as torchaudio detects it automatically.
    Uses a temp file for cross-platform compatibility.

    Args:
        audio_bytes: Audio bytes in any format supported by torchaudio.
        sample_rate: Ignored - torchaudio detects sample rate from file.

    Returns:
        Audio samples as sequence of floats at 16kHz.

    Raises:
        ValueError: If audio format cannot be detected or is unsupported.
    """
    del sample_rate  # Ignored - torchaudio detects sample rate

    import tempfile
    from pathlib import Path

    # Detect audio format from magic bytes
    audio_format = _detect_audio_format(audio_bytes)

    # Write to temp file (required for Windows compatibility)
    with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    # Use torchaudio to decode audio from temp file
    torchaudio_mod = __import__("torchaudio")
    load_fn: _TorchaudioLoadFileProtocol = torchaudio_mod.load
    waveform_and_rate = load_fn(tmp_path)
    waveform_raw: _TorchTensorProtocol = waveform_and_rate[0]
    source_rate: int = waveform_and_rate[1]

    # Clean up temp file
    tmp_path.unlink()

    # torchaudio returns (channels, samples) - take first channel if stereo
    waveform: _TorchTensorProtocol = waveform_raw
    if waveform_raw.shape[0] > 1:
        waveform = waveform_raw[0].unsqueeze(0)

    # Resample to 16kHz if needed
    target_rate = 16000
    if source_rate != target_rate:
        transforms_mod = __import__("torchaudio.transforms", fromlist=["Resample"])
        resample_cls: _ResampleClassProtocol = transforms_mod.Resample
        resampler = resample_cls(source_rate, target_rate)
        waveform = resampler(waveform)

    # Convert to list of floats
    result: list[float] = waveform.squeeze(0).tolist()
    return result


def _default_detector_factory(config: DetectorConfig) -> SpokenLanguageDetectorProtocol:
    """Production implementation - creates real detector."""
    from .detector import SpokenLanguageDetector

    detector: SpokenLanguageDetectorProtocol = SpokenLanguageDetector(config)
    return detector


# =============================================================================
# Module-level Hooks
# =============================================================================


# Hook for model loading
model_factory: ModelFactoryProtocol = _default_model_factory

# Hook for processor loading
processor_factory: ProcessorFactoryProtocol = _default_processor_factory

# Hook for audio loading/resampling
audio_loader: AudioLoaderProtocol = _default_audio_loader

# Hook for detector creation
detector_factory: DetectorFactoryProtocol = _default_detector_factory


# =============================================================================
# Language Code Conversion
# =============================================================================


class LanguageCodeConverterProtocol(Protocol):
    """Protocol for language code conversion."""

    def __call__(self, code: str) -> str | None:
        """Convert ISO 639-3 code to Whisper-compatible code.

        Args:
            code: ISO 639-3 language code (e.g., "eng", "vie").

        Returns:
            Whisper-compatible code if supported, None if not supported.
        """
        ...


# Whisper-supported language codes (from openai/whisper tokenizer.py)
# These are the only codes Whisper API accepts
WHISPER_SUPPORTED_CODES: frozenset[str] = frozenset({
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca",
    "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
    "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
    "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
    "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
    "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
    "ba", "jw", "su", "yue",
})


def _default_convert_language_code(code: str) -> str | None:
    """Convert ISO 639-3 to Whisper-compatible code.

    Args:
        code: ISO 639-3 language code (e.g., "eng", "vie", "cmn").

    Returns:
        Whisper-compatible code if supported, None if not supported.
        When None is returned, callers should let Whisper auto-detect.
    """
    # Already a Whisper-supported code
    if code in WHISPER_SUPPORTED_CODES:
        return code

    langcodes_mod = __import__("langcodes")
    language_cls = langcodes_mod.Language
    lang = language_cls.get(code)

    # .language gives the primary subtag (may be 2 or 3 letters)
    primary: str = lang.language

    # If 2 letters and supported by Whisper, return it
    if len(primary) == 2 and primary in WHISPER_SUPPORTED_CODES:
        return primary

    # For 3-letter codes like cmn, check broader_tags for Whisper-supported code
    broader: list[str] = lang.broader_tags()
    for tag in broader:
        if tag in WHISPER_SUPPORTED_CODES:
            return tag

    # No Whisper-supported code found - return None to trigger auto-detect
    return None


# Hook for language code conversion
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
