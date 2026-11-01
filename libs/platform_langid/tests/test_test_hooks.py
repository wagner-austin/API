"""Tests for the platform_langid hook modules."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pytest

from platform_langid import _hook_defaults, _test_hooks
from platform_langid._hook_protocols import ModelConfigProtocol
from platform_langid.testing import reset_hooks
from platform_langid.types import default_detector_config


class TestDetectAudioFormat:
    """Tests for _detect_audio_format function."""

    def test_detect_wav_format(self) -> None:
        """Detect WAV format from RIFF/WAVE header."""
        wav_header = b"RIFF\x00\x00\x00\x00WAVEfmt "
        result = _hook_defaults._detect_audio_format(wav_header)
        assert result == "wav"

    def test_detect_mp3_id3_format(self) -> None:
        """Detect MP3 format from ID3 header."""
        mp3_header = b"ID3\x04\x00\x00\x00\x00\x00\x00\x00\x00"
        result = _hook_defaults._detect_audio_format(mp3_header)
        assert result == "mp3"

    def test_detect_mp3_sync_format(self) -> None:
        """Detect MP3 format from sync bytes."""
        mp3_header = b"\xff\xfb\x90\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        result = _hook_defaults._detect_audio_format(mp3_header)
        assert result == "mp3"

    def test_detect_ogg_format(self) -> None:
        """Detect OGG format from OggS header."""
        ogg_header = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00"
        result = _hook_defaults._detect_audio_format(ogg_header)
        assert result == "ogg"

    def test_detect_flac_format(self) -> None:
        """Detect FLAC format from fLaC header."""
        flac_header = b"fLaC\x00\x00\x00\x22\x00\x00\x00\x00"
        result = _hook_defaults._detect_audio_format(flac_header)
        assert result == "flac"

    def test_detect_webm_format(self) -> None:
        """Detect WebM format from EBML header."""
        webm_header = b"\x1a\x45\xdf\xa3\x01\x00\x00\x00\x00\x00\x00\x00"
        result = _hook_defaults._detect_audio_format(webm_header)
        assert result == "webm"

    def test_audio_too_short_raises(self) -> None:
        """Raise ValueError if audio data is too short."""
        short_data = b"\x00\x00\x00"
        with pytest.raises(ValueError, match="Audio data too short"):
            _hook_defaults._detect_audio_format(short_data)

    def test_unknown_format_raises(self) -> None:
        """Raise ValueError for unknown audio format."""
        unknown_data = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b"
        with pytest.raises(ValueError, match="Unknown audio format"):
            _hook_defaults._detect_audio_format(unknown_data)


class TestDefaultAudioLoader:
    """Tests for _default_audio_loader function.

    Note: These tests use real audio files via torchaudio.
    The audio loader decodes audio formats (MP3, WAV, etc.) using torchaudio.load.
    """

    def test_load_wav_audio(self) -> None:
        """Load WAV audio from bytes using torchaudio."""
        # Create a minimal valid WAV file header + 16kHz mono audio
        # WAV format: RIFF header + fmt chunk + data chunk
        import struct

        sample_rate = 16000
        num_samples = 1600  # 0.1 seconds of audio
        bits_per_sample = 16
        num_channels = 1
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = num_samples * block_align

        # Generate samples: simple sine wave approximation
        samples: list[int] = []
        for i in range(num_samples):
            # Simple triangle wave
            phase = (i % 100) / 100.0
            value = int((phase * 2 - 1) * 16000)  # Scale to ~half of int16 range
            samples.append(value)

        # Build WAV file
        wav_data = bytearray()
        # RIFF header
        wav_data.extend(b"RIFF")
        wav_data.extend(struct.pack("<I", 36 + data_size))  # File size - 8
        wav_data.extend(b"WAVE")
        # fmt chunk
        wav_data.extend(b"fmt ")
        wav_data.extend(struct.pack("<I", 16))  # fmt chunk size
        wav_data.extend(struct.pack("<H", 1))  # PCM format
        wav_data.extend(struct.pack("<H", num_channels))
        wav_data.extend(struct.pack("<I", sample_rate))
        wav_data.extend(struct.pack("<I", byte_rate))
        wav_data.extend(struct.pack("<H", block_align))
        wav_data.extend(struct.pack("<H", bits_per_sample))
        # data chunk
        wav_data.extend(b"data")
        wav_data.extend(struct.pack("<I", data_size))
        for sample in samples:
            wav_data.extend(struct.pack("<h", sample))

        audio_bytes = bytes(wav_data)
        result = _test_hooks._default_audio_loader(audio_bytes, 0)  # sample_rate ignored

        # Should return float samples
        assert len(result) == num_samples
        # First sample should be negative (start of triangle wave)
        assert result[0] < 0.0
        # Values should be in valid range
        for idx in range(len(result)):
            sample_value = result[idx]
            assert -1.0 <= sample_value <= 1.0

    def test_load_stereo_takes_first_channel(self) -> None:
        """Stereo audio is converted to mono by taking first channel."""
        import struct

        sample_rate = 16000
        num_samples = 800
        bits_per_sample = 16
        num_channels = 2  # Stereo
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = num_samples * block_align

        # Build WAV file with stereo audio
        wav_data = bytearray()
        wav_data.extend(b"RIFF")
        wav_data.extend(struct.pack("<I", 36 + data_size))
        wav_data.extend(b"WAVE")
        wav_data.extend(b"fmt ")
        wav_data.extend(struct.pack("<I", 16))
        wav_data.extend(struct.pack("<H", 1))  # PCM
        wav_data.extend(struct.pack("<H", num_channels))
        wav_data.extend(struct.pack("<I", sample_rate))
        wav_data.extend(struct.pack("<I", byte_rate))
        wav_data.extend(struct.pack("<H", block_align))
        wav_data.extend(struct.pack("<H", bits_per_sample))
        wav_data.extend(b"data")
        wav_data.extend(struct.pack("<I", data_size))

        # Left channel: positive, Right channel: negative
        for _ in range(num_samples):
            left_value = 8000  # Positive
            right_value = -8000  # Negative
            wav_data.extend(struct.pack("<h", left_value))
            wav_data.extend(struct.pack("<h", right_value))

        audio_bytes = bytes(wav_data)
        result = _test_hooks._default_audio_loader(audio_bytes, 0)

        # Should have mono output (first channel only)
        assert len(result) == num_samples
        # All values should be positive (from left channel)
        for idx in range(len(result)):
            sample_value = result[idx]
            assert sample_value > 0.0

    def test_resample_from_48000_to_16000(self) -> None:
        """Resample audio from 48kHz to 16kHz."""
        import struct

        source_rate = 48000
        num_samples = 4800  # 0.1 seconds at 48kHz
        bits_per_sample = 16
        num_channels = 1
        byte_rate = source_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = num_samples * block_align

        wav_data = bytearray()
        wav_data.extend(b"RIFF")
        wav_data.extend(struct.pack("<I", 36 + data_size))
        wav_data.extend(b"WAVE")
        wav_data.extend(b"fmt ")
        wav_data.extend(struct.pack("<I", 16))
        wav_data.extend(struct.pack("<H", 1))
        wav_data.extend(struct.pack("<H", num_channels))
        wav_data.extend(struct.pack("<I", source_rate))
        wav_data.extend(struct.pack("<I", byte_rate))
        wav_data.extend(struct.pack("<H", block_align))
        wav_data.extend(struct.pack("<H", bits_per_sample))
        wav_data.extend(b"data")
        wav_data.extend(struct.pack("<I", data_size))

        for _ in range(num_samples):
            wav_data.extend(struct.pack("<h", 1000))

        audio_bytes = bytes(wav_data)
        result = _test_hooks._default_audio_loader(audio_bytes, 0)

        # After resampling from 48000 to 16000, expect 1/3 of samples
        expected_samples = num_samples * 16000 // source_rate
        assert len(result) == expected_samples


class _SavesPretrained(Protocol):
    """A transformers object that writes itself to a directory."""

    def save_pretrained(self, save_directory: str) -> None:
        """Write the object's files into the directory."""
        ...


class _ConfigClass(Protocol):
    """transformers.Wav2Vec2Config, with the fields a tiny classifier sets."""

    def __call__(
        self,
        *,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        conv_dim: tuple[int, ...],
        conv_kernel: tuple[int, ...],
        conv_stride: tuple[int, ...],
        num_conv_pos_embeddings: int,
        num_conv_pos_embedding_groups: int,
        classifier_proj_size: int,
        id2label: dict[int, str],
        label2id: dict[str, int],
    ) -> ModelConfigProtocol:
        """Build the configuration."""
        ...


class _ModelClass(Protocol):
    """transformers.Wav2Vec2ForSequenceClassification, built from a configuration."""

    def __call__(self, config: ModelConfigProtocol) -> _SavesPretrained:
        """Build a randomly initialised model."""
        ...


class _ExtractorClass(Protocol):
    """transformers.Wav2Vec2FeatureExtractor, with its defaults."""

    def __call__(self) -> _SavesPretrained:
        """Build the feature extractor."""
        ...


#: The labels of the tiny classifier, MMS-LID-126's first and last.
TINY_LABELS: dict[int, str] = {0: "ara", 1: "ina"}


def _save_tiny_classifier(directory: Path) -> str:
    """Save a two-label Wav2Vec2 classifier and its feature extractor.

    The default factories are tested by loading this from disk, the same
    ``from_pretrained`` path a hub id takes once resolved, so the suite
    downloads nothing: MMS-LID-126 is a 3.86 GB checkpoint, and fetching it
    in a check on a node without it took 240 s of fleet job f2af5334's 634 s
    (MCPs board task abb0a10e).

    Args:
        directory: An empty directory.

    Returns:
        The directory, as the model id the factories take.
    """
    mod = __import__(
        "transformers",
        fromlist=[
            "Wav2Vec2Config",
            "Wav2Vec2ForSequenceClassification",
            "Wav2Vec2FeatureExtractor",
        ],
    )
    config_cls: _ConfigClass = mod.Wav2Vec2Config
    model_cls: _ModelClass = mod.Wav2Vec2ForSequenceClassification
    extractor_cls: _ExtractorClass = mod.Wav2Vec2FeatureExtractor
    config = config_cls(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        conv_dim=(8, 8),
        conv_kernel=(10, 3),
        conv_stride=(5, 2),
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        classifier_proj_size=8,
        id2label=TINY_LABELS,
        label2id={label: index for index, label in TINY_LABELS.items()},
    )
    model_cls(config).save_pretrained(str(directory))
    extractor_cls().save_pretrained(str(directory))
    return str(directory)


class TestDefaultModelFactory:
    """Tests for _default_model_factory function."""

    def test_loads_model_from_transformers(self, tmp_path: Path) -> None:
        """Load a saved Wav2Vec2 classifier through transformers, labels intact."""
        model = _test_hooks._default_model_factory(_save_tiny_classifier(tmp_path))
        assert model.config.id2label == TINY_LABELS


class TestDefaultProcessorFactory:
    """Tests for _default_processor_factory function."""

    def test_loads_processor_from_transformers(self, tmp_path: Path) -> None:
        """Load a saved feature extractor through transformers' AutoFeatureExtractor."""
        processor = _test_hooks._default_processor_factory(_save_tiny_classifier(tmp_path))
        assert callable(processor)


class TestDefaultDetectorFactory:
    """Tests for _default_detector_factory function."""

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_creates_spoken_language_detector(self) -> None:
        """Factory creates SpokenLanguageDetector instance."""
        from platform_langid.testing import (
            FakeAudioLoader,
            make_fake_model_factory,
            make_fake_processor_factory,
        )

        # Set up fakes so we don't load real models
        _test_hooks.model_factory = make_fake_model_factory()
        _test_hooks.processor_factory = make_fake_processor_factory()
        _test_hooks.audio_loader = FakeAudioLoader()

        config = default_detector_config()
        detector = _test_hooks._default_detector_factory(config)

        assert detector.__class__.__name__ == "SpokenLanguageDetector"


class TestDefaultConvertLanguageCode:
    """Tests for _default_convert_language_code function."""

    def test_returns_whisper_code_directly(self) -> None:
        """Return code unchanged if already Whisper-supported.

        Whisper supports standard ISO 639-1 codes like 'en', 'vi', 'es'.
        These should be returned as-is without conversion.
        """
        result = _test_hooks._default_convert_language_code("en")
        assert result == "en"

        result = _test_hooks._default_convert_language_code("vi")
        assert result == "vi"

    def test_converts_iso_639_3_to_whisper_code(self) -> None:
        """Convert ISO 639-3 codes to Whisper-compatible ISO 639-1.

        MMS-LID returns 3-letter codes like 'eng', 'vie'. These need
        conversion to 2-letter codes that Whisper understands.
        """
        result = _test_hooks._default_convert_language_code("eng")
        assert result == "en"

        result = _test_hooks._default_convert_language_code("vie")
        assert result == "vi"

    def test_converts_chinese_mandarin_code(self) -> None:
        """Convert Mandarin Chinese code 'cmn' to Whisper 'zh'.

        Mandarin Chinese uses ISO 639-3 code 'cmn' but Whisper expects 'zh'.
        The broader_tags() method provides this mapping.
        """
        result = _test_hooks._default_convert_language_code("cmn")
        assert result == "zh"

    def test_returns_none_for_unsupported_language(self) -> None:
        """Return None for languages not supported by Whisper.

        When a language code cannot be mapped to a Whisper-supported code,
        return None to signal that Whisper should auto-detect.
        """
        # Afrikaans Oorlams variant - rare language not in Whisper
        result = _test_hooks._default_convert_language_code("oor")
        assert result is None

    def test_convert_language_code_hook_default(self) -> None:
        """convert_language_code hook defaults to _default_convert_language_code."""
        from platform_langid.testing import reset_hooks

        reset_hooks()
        assert _test_hooks.convert_language_code is _test_hooks._default_convert_language_code


class TestDefaultHooks:
    """Tests for default hook values."""

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_model_factory_default(self) -> None:
        """model_factory defaults to _default_model_factory."""
        reset_hooks()
        assert _test_hooks.model_factory is _test_hooks._default_model_factory

    def test_processor_factory_default(self) -> None:
        """processor_factory defaults to _default_processor_factory."""
        reset_hooks()
        assert _test_hooks.processor_factory is _test_hooks._default_processor_factory

    def test_audio_loader_default(self) -> None:
        """audio_loader defaults to _default_audio_loader."""
        reset_hooks()
        assert _test_hooks.audio_loader is _test_hooks._default_audio_loader

    def test_detector_factory_default(self) -> None:
        """detector_factory defaults to _default_detector_factory."""
        reset_hooks()
        assert _test_hooks.detector_factory is _test_hooks._default_detector_factory

    def test_convert_language_code_default(self) -> None:
        """convert_language_code defaults to _default_convert_language_code."""
        reset_hooks()
        assert _test_hooks.convert_language_code is _test_hooks._default_convert_language_code


class TestExports:
    """Tests for module exports."""

    def test_all_protocols_exported(self) -> None:
        """All protocol types are in __all__."""
        assert "ModelProtocol" in _test_hooks.__all__
        assert "ProcessorProtocol" in _test_hooks.__all__
        assert "TensorProtocol" in _test_hooks.__all__
        assert "SpokenLanguageDetectorProtocol" in _test_hooks.__all__
        assert "ModelFactoryProtocol" in _test_hooks.__all__
        assert "ProcessorFactoryProtocol" in _test_hooks.__all__
        assert "DetectorFactoryProtocol" in _test_hooks.__all__
        assert "AudioLoaderProtocol" in _test_hooks.__all__
        assert "LanguageCodeConverterProtocol" in _test_hooks.__all__

    def test_all_defaults_exported(self) -> None:
        """All default implementations are in __all__."""
        assert "_default_model_factory" in _test_hooks.__all__
        assert "_default_processor_factory" in _test_hooks.__all__
        assert "_default_audio_loader" in _test_hooks.__all__
        assert "_default_detector_factory" in _test_hooks.__all__
        assert "_default_convert_language_code" in _test_hooks.__all__

    def test_all_hooks_exported(self) -> None:
        """All hook variables are in __all__."""
        assert "model_factory" in _test_hooks.__all__
        assert "processor_factory" in _test_hooks.__all__
        assert "audio_loader" in _test_hooks.__all__
        assert "detector_factory" in _test_hooks.__all__
        assert "convert_language_code" in _test_hooks.__all__
