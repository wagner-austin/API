"""Tests for platform_langid.types module."""

from __future__ import annotations

import base64

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

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


class TestSpokenLanguageResult:
    """Tests for SpokenLanguageResult TypedDict."""

    def test_encode_basic(self) -> None:
        """Encode basic result."""
        result = SpokenLanguageResult(
            language="vi",
            confidence=0.95,
            model_id="facebook/mms-lid-4017",
        )
        encoded = encode_spoken_language_result(result)
        assert encoded == {
            "language": "vi",
            "confidence": 0.95,
            "model_id": "facebook/mms-lid-4017",
        }

    def test_decode_valid(self) -> None:
        """Decode valid result."""
        obj: JSONObject = {
            "language": "en",
            "confidence": 0.99,
            "model_id": "test-model",
        }
        result = decode_spoken_language_result(obj)
        assert result["language"] == "en"
        assert result["confidence"] == 0.99
        assert result["model_id"] == "test-model"

    def test_decode_missing_language(self) -> None:
        """Raise for missing language field."""
        obj: JSONObject = {"confidence": 0.9, "model_id": "test"}
        with pytest.raises(JSONTypeError):
            decode_spoken_language_result(obj)

    def test_decode_missing_confidence(self) -> None:
        """Raise for missing confidence field."""
        obj: JSONObject = {"language": "en", "model_id": "test"}
        with pytest.raises(JSONTypeError):
            decode_spoken_language_result(obj)

    def test_decode_missing_model_id(self) -> None:
        """Raise for missing model_id field."""
        obj: JSONObject = {"language": "en", "confidence": 0.9}
        with pytest.raises(JSONTypeError):
            decode_spoken_language_result(obj)

    def test_decode_confidence_too_low(self) -> None:
        """Raise for confidence below 0."""
        obj: JSONObject = {
            "language": "en",
            "confidence": -0.1,
            "model_id": "test",
        }
        with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
            decode_spoken_language_result(obj)

    def test_decode_confidence_too_high(self) -> None:
        """Raise for confidence above 1."""
        obj: JSONObject = {
            "language": "en",
            "confidence": 1.5,
            "model_id": "test",
        }
        with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
            decode_spoken_language_result(obj)

    def test_decode_confidence_boundary_zero(self) -> None:
        """Accept confidence at 0.0 boundary."""
        obj: JSONObject = {
            "language": "en",
            "confidence": 0.0,
            "model_id": "test",
        }
        result = decode_spoken_language_result(obj)
        assert result["confidence"] == 0.0

    def test_decode_confidence_boundary_one(self) -> None:
        """Accept confidence at 1.0 boundary."""
        obj: JSONObject = {
            "language": "en",
            "confidence": 1.0,
            "model_id": "test",
        }
        result = decode_spoken_language_result(obj)
        assert result["confidence"] == 1.0

    def test_require_valid(self) -> None:
        """Require valid result from JSONValue."""
        obj: JSONObject = {
            "language": "es",
            "confidence": 0.88,
            "model_id": "model",
        }
        result = require_spoken_language_result(obj)
        assert result["language"] == "es"

    def test_require_non_dict(self) -> None:
        """Raise for non-dict value."""
        with pytest.raises(JSONTypeError, match="Expected object"):
            require_spoken_language_result("not a dict")

    def test_require_list(self) -> None:
        """Raise for list value."""
        with pytest.raises(JSONTypeError, match="Expected object"):
            require_spoken_language_result([1, 2, 3])

    def test_roundtrip(self) -> None:
        """Encode then decode produces identical result."""
        original = SpokenLanguageResult(
            language="vi",
            confidence=0.94,
            model_id="facebook/mms-lid-4017",
        )
        encoded = encode_spoken_language_result(original)
        decoded = decode_spoken_language_result(encoded)
        assert decoded == original


class TestAudioInput:
    """Tests for AudioInput TypedDict."""

    def test_encode_basic(self) -> None:
        """Encode audio input with base64 waveform."""
        audio = AudioInput(
            waveform=b"\x00\x01\x02\x03",
            sample_rate=16000,
            format="pcm_s16le",
        )
        encoded = encode_audio_input(audio)
        assert encoded["waveform"] == base64.b64encode(b"\x00\x01\x02\x03").decode("ascii")
        assert encoded["sample_rate"] == 16000
        assert encoded["format"] == "pcm_s16le"

    def test_decode_valid(self) -> None:
        """Decode valid audio input."""
        waveform_b64 = base64.b64encode(b"\xff\xfe").decode("ascii")
        obj: JSONObject = {
            "waveform": waveform_b64,
            "sample_rate": 44100,
            "format": "pcm_s16le",
        }
        audio = decode_audio_input(obj)
        assert audio["waveform"] == b"\xff\xfe"
        assert audio["sample_rate"] == 44100
        assert audio["format"] == "pcm_s16le"

    def test_decode_missing_waveform(self) -> None:
        """Raise for missing waveform field."""
        obj: JSONObject = {"sample_rate": 16000, "format": "pcm"}
        with pytest.raises(JSONTypeError):
            decode_audio_input(obj)

    def test_decode_missing_sample_rate(self) -> None:
        """Raise for missing sample_rate field."""
        obj: JSONObject = {
            "waveform": base64.b64encode(b"\x00").decode("ascii"),
            "format": "pcm",
        }
        with pytest.raises(JSONTypeError):
            decode_audio_input(obj)

    def test_decode_missing_format(self) -> None:
        """Raise for missing format field."""
        obj: JSONObject = {
            "waveform": base64.b64encode(b"\x00").decode("ascii"),
            "sample_rate": 16000,
        }
        with pytest.raises(JSONTypeError):
            decode_audio_input(obj)

    def test_decode_invalid_sample_rate(self) -> None:
        """Raise for non-positive sample_rate."""
        obj: JSONObject = {
            "waveform": base64.b64encode(b"\x00").decode("ascii"),
            "sample_rate": 0,
            "format": "pcm",
        }
        with pytest.raises(ValueError, match="must be positive"):
            decode_audio_input(obj)

    def test_decode_negative_sample_rate(self) -> None:
        """Raise for negative sample_rate."""
        obj: JSONObject = {
            "waveform": base64.b64encode(b"\x00").decode("ascii"),
            "sample_rate": -1000,
            "format": "pcm",
        }
        with pytest.raises(ValueError, match="must be positive"):
            decode_audio_input(obj)

    def test_require_valid(self) -> None:
        """Require valid audio input from JSONValue."""
        obj: JSONObject = {
            "waveform": base64.b64encode(b"\x00").decode("ascii"),
            "sample_rate": 16000,
            "format": "pcm",
        }
        audio = require_audio_input(obj)
        assert audio["sample_rate"] == 16000

    def test_require_non_dict(self) -> None:
        """Raise for non-dict value."""
        with pytest.raises(JSONTypeError, match="Expected object"):
            require_audio_input("not a dict")

    def test_roundtrip(self) -> None:
        """Encode then decode produces identical result."""
        original = AudioInput(
            waveform=b"\x00\x01\x02\x03\x04\x05",
            sample_rate=22050,
            format="float32",
        )
        encoded = encode_audio_input(original)
        decoded = decode_audio_input(encoded)
        assert decoded == original


class TestDetectorConfig:
    """Tests for DetectorConfig TypedDict."""

    def test_encode_basic(self) -> None:
        """Encode detector config."""
        config = DetectorConfig(
            model_id="facebook/mms-lid-4017",
            device="cuda",
            confidence_threshold=0.5,
        )
        encoded = encode_detector_config(config)
        assert encoded == {
            "model_id": "facebook/mms-lid-4017",
            "device": "cuda",
            "confidence_threshold": 0.5,
        }

    def test_decode_valid(self) -> None:
        """Decode valid config."""
        obj: JSONObject = {
            "model_id": "test-model",
            "device": "cpu",
            "confidence_threshold": 0.0,
        }
        config = decode_detector_config(obj)
        assert config["model_id"] == "test-model"
        assert config["device"] == "cpu"
        assert config["confidence_threshold"] == 0.0

    def test_decode_missing_model_id(self) -> None:
        """Raise for missing model_id field."""
        obj: JSONObject = {"device": "cpu", "confidence_threshold": 0.5}
        with pytest.raises(JSONTypeError):
            decode_detector_config(obj)

    def test_decode_missing_device(self) -> None:
        """Raise for missing device field."""
        obj: JSONObject = {"model_id": "test", "confidence_threshold": 0.5}
        with pytest.raises(JSONTypeError):
            decode_detector_config(obj)

    def test_decode_missing_confidence_threshold(self) -> None:
        """Raise for missing confidence_threshold field."""
        obj: JSONObject = {"model_id": "test", "device": "cpu"}
        with pytest.raises(JSONTypeError):
            decode_detector_config(obj)

    def test_decode_threshold_too_low(self) -> None:
        """Raise for confidence_threshold below 0."""
        obj: JSONObject = {
            "model_id": "test",
            "device": "cpu",
            "confidence_threshold": -0.1,
        }
        with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
            decode_detector_config(obj)

    def test_decode_threshold_too_high(self) -> None:
        """Raise for confidence_threshold above 1."""
        obj: JSONObject = {
            "model_id": "test",
            "device": "cpu",
            "confidence_threshold": 1.5,
        }
        with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
            decode_detector_config(obj)

    def test_require_valid(self) -> None:
        """Require valid config from JSONValue."""
        obj: JSONObject = {
            "model_id": "model",
            "device": "mps",
            "confidence_threshold": 0.8,
        }
        config = require_detector_config(obj)
        assert config["device"] == "mps"

    def test_require_non_dict(self) -> None:
        """Raise for non-dict value."""
        with pytest.raises(JSONTypeError, match="Expected object"):
            require_detector_config(123)

    def test_roundtrip(self) -> None:
        """Encode then decode produces identical result."""
        original = DetectorConfig(
            model_id="facebook/mms-lid-4017",
            device="cuda:0",
            confidence_threshold=0.75,
        )
        encoded = encode_detector_config(original)
        decoded = decode_detector_config(encoded)
        assert decoded == original


class TestDefaultConfig:
    """Tests for default_detector_config function."""

    def test_returns_correct_defaults(self) -> None:
        """Returns config with default values."""
        config = default_detector_config()
        assert config["model_id"] == DEFAULT_MODEL_ID
        assert config["device"] == DEFAULT_DEVICE
        assert config["confidence_threshold"] == DEFAULT_CONFIDENCE_THRESHOLD

    def test_default_model_id_value(self) -> None:
        """DEFAULT_MODEL_ID has expected value."""
        assert DEFAULT_MODEL_ID == "facebook/mms-lid-4017"

    def test_default_device_value(self) -> None:
        """DEFAULT_DEVICE has expected value."""
        assert DEFAULT_DEVICE == "cpu"

    def test_default_threshold_value(self) -> None:
        """DEFAULT_CONFIDENCE_THRESHOLD has expected value."""
        assert DEFAULT_CONFIDENCE_THRESHOLD == 0.0
