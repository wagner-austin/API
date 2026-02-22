"""Tests for platform_langid.detector module."""

from __future__ import annotations

from typing import ClassVar

import pytest

from platform_langid import _test_hooks
from platform_langid.detector import (
    TARGET_SAMPLE_RATE,
    SpokenLanguageDetector,
    create_detector,
    detect_spoken_language,
)
from platform_langid.testing import (
    FakeAudioLoader,
    FakeModel,
    make_fake_detector_factory,
    make_fake_model_factory,
    make_fake_processor_factory,
    reset_hooks,
)
from platform_langid.types import DetectorConfig, default_detector_config


class TestSpokenLanguageDetector:
    """Tests for SpokenLanguageDetector class."""

    def setup_method(self) -> None:
        """Reset hooks before each test."""
        reset_hooks()
        # Install fakes
        _test_hooks.model_factory = make_fake_model_factory(
            predicted_id=1,
            confidence=0.94,
            id2label={0: "en", 1: "vi", 2: "es"},
        )
        _test_hooks.processor_factory = make_fake_processor_factory()
        _test_hooks.audio_loader = FakeAudioLoader()

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_detect_returns_result(self) -> None:
        """Detect returns SpokenLanguageResult."""
        config = default_detector_config()
        detector = SpokenLanguageDetector(config)
        result = detector.detect(b"\x00\x01\x02\x03", 16000)
        assert result["language"] == "vi"
        assert result["confidence"] > 0.0
        assert result["model_id"] == config["model_id"]

    def test_detect_empty_audio_raises(self) -> None:
        """Detect raises ValueError for empty audio."""
        config = default_detector_config()
        detector = SpokenLanguageDetector(config)
        with pytest.raises(ValueError, match="cannot be empty"):
            detector.detect(b"", 16000)

    def test_detect_uses_audio_loader(self) -> None:
        """Detect passes audio through audio_loader hook."""
        fake_loader = FakeAudioLoader()
        _test_hooks.audio_loader = fake_loader

        config = default_detector_config()
        detector = SpokenLanguageDetector(config)
        detector.detect(b"\x00\x01", 44100)

        assert len(fake_loader.calls) == 1
        assert fake_loader.calls[0] == (b"\x00\x01", 44100)

    def test_detect_with_confidence_threshold(self) -> None:
        """Detect returns 'auto' when confidence below threshold.

        When confidence is below threshold, return 'auto' to let Whisper
        perform its own language detection instead of using our uncertain result.
        """
        # Set model to return low confidence
        # With 5 classes and 0.3 confidence for class 0, others have 0.175 each
        # so class 0 wins argmax but with low confidence
        _test_hooks.model_factory = make_fake_model_factory(
            predicted_id=0,
            confidence=0.3,
            id2label={0: "en", 1: "vi", 2: "es", 3: "fr", 4: "de"},
        )

        config = DetectorConfig(
            model_id="test",
            device="cpu",
            confidence_threshold=0.5,
        )
        detector = SpokenLanguageDetector(config)
        result = detector.detect(b"\x00\x01", 16000)

        assert result["language"] == "auto"
        assert result["confidence"] == pytest.approx(0.3, abs=0.05)

    def test_detect_above_threshold(self) -> None:
        """Detect returns language when confidence above threshold."""
        _test_hooks.model_factory = make_fake_model_factory(
            predicted_id=0,
            confidence=0.9,
            id2label={0: "en", 1: "vi"},
        )

        config = DetectorConfig(
            model_id="test",
            device="cpu",
            confidence_threshold=0.5,
        )
        detector = SpokenLanguageDetector(config)
        result = detector.detect(b"\x00\x01", 16000)

        assert result["language"] == "en"

    def test_detector_moves_model_to_device(self) -> None:
        """Detector moves model to specified device."""

        class DeviceTrackingModel(FakeModel):
            """Model that tracks device transfers."""

            device_calls: ClassVar[list[str]] = []

            def to(self, device: str) -> DeviceTrackingModel:
                DeviceTrackingModel.device_calls.append(device)
                return self

        DeviceTrackingModel.device_calls = []

        def tracking_factory(model_id: str) -> _test_hooks.ModelProtocol:
            del model_id
            return DeviceTrackingModel()

        _test_hooks.model_factory = tracking_factory

        config = DetectorConfig(
            model_id="test",
            device="cuda:0",
            confidence_threshold=0.0,
        )
        SpokenLanguageDetector(config)

        assert DeviceTrackingModel.device_calls == ["cuda:0"]


class TestDetectSpokenLanguage:
    """Tests for detect_spoken_language convenience function."""

    def setup_method(self) -> None:
        """Reset hooks before each test."""
        reset_hooks()
        _test_hooks.detector_factory = make_fake_detector_factory(
            language="es",
            confidence=0.88,
            model_id="test-model",
        )

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_detect_with_default_config(self) -> None:
        """Detect uses default config when none provided."""
        result = detect_spoken_language(b"\x00\x01", 16000)
        assert result["language"] == "es"
        assert result["confidence"] == 0.88

    def test_detect_with_custom_config(self) -> None:
        """Detect uses provided config."""
        config = DetectorConfig(
            model_id="custom-model",
            device="mps",
            confidence_threshold=0.5,
        )
        result = detect_spoken_language(b"\x00\x01", 16000, config=config)
        assert result["language"] == "es"

    def test_detect_with_default_sample_rate(self) -> None:
        """Detect uses TARGET_SAMPLE_RATE as default."""
        assert TARGET_SAMPLE_RATE == 16000
        result = detect_spoken_language(b"\x00\x01")
        assert result["language"] == "es"


class TestCreateDetector:
    """Tests for create_detector function."""

    def setup_method(self) -> None:
        """Reset hooks before each test."""
        reset_hooks()
        _test_hooks.detector_factory = make_fake_detector_factory()

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_create_with_default_config(self) -> None:
        """Create detector with default config."""
        detector = create_detector()
        result = detector.detect(b"\x00\x01", 16000)
        assert "language" in result

    def test_create_with_custom_config(self) -> None:
        """Create detector with custom config."""
        config = DetectorConfig(
            model_id="custom",
            device="cuda",
            confidence_threshold=0.8,
        )
        detector = create_detector(config)
        result = detector.detect(b"\x00\x01", 16000)
        assert "language" in result


class TestTargetSampleRate:
    """Tests for TARGET_SAMPLE_RATE constant."""

    def test_value(self) -> None:
        """TARGET_SAMPLE_RATE is 16000 Hz."""
        assert TARGET_SAMPLE_RATE == 16000
