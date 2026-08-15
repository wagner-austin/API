"""_test_hooks: _default_model_factory and related definitions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from platform_langid._hook_protocols import (
    ModelProtocol,
    ProcessorProtocol,
    SpokenLanguageDetectorProtocol,
    _LanguageClassProtocol,
    _LanguageProtocol,
    _ModelClassProtocol,
    _ProcessorClassProtocol,
    _ResampleClassProtocol,
    _TorchTensorProtocol,
)

from .types import DetectorConfig


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

# Hook for processor loading

# Hook for audio loading/resampling

# Hook for detector creation


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
WHISPER_SUPPORTED_CODES: frozenset[str] = frozenset(
    {
        "en",
        "zh",
        "de",
        "es",
        "ru",
        "ko",
        "fr",
        "ja",
        "pt",
        "tr",
        "pl",
        "ca",
        "nl",
        "ar",
        "sv",
        "it",
        "id",
        "hi",
        "fi",
        "vi",
        "he",
        "uk",
        "el",
        "ms",
        "cs",
        "ro",
        "da",
        "hu",
        "ta",
        "no",
        "th",
        "ur",
        "hr",
        "bg",
        "lt",
        "la",
        "mi",
        "ml",
        "cy",
        "sk",
        "te",
        "fa",
        "lv",
        "bn",
        "sr",
        "az",
        "sl",
        "kn",
        "et",
        "mk",
        "br",
        "eu",
        "is",
        "hy",
        "ne",
        "mn",
        "bs",
        "kk",
        "sq",
        "sw",
        "gl",
        "mr",
        "pa",
        "si",
        "km",
        "sn",
        "yo",
        "so",
        "af",
        "oc",
        "ka",
        "be",
        "tg",
        "sd",
        "gu",
        "am",
        "yi",
        "lo",
        "uz",
        "fo",
        "ht",
        "ps",
        "tk",
        "nn",
        "mt",
        "sa",
        "lb",
        "my",
        "bo",
        "tl",
        "mg",
        "as",
        "tt",
        "haw",
        "ln",
        "ha",
        "ba",
        "jw",
        "su",
        "yue",
    }
)


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
    language_cls: _LanguageClassProtocol = langcodes_mod.Language
    lang: _LanguageProtocol = language_cls.get(code)

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
