"""Tests for platform_langid.testing module."""

from __future__ import annotations

import pytest

from platform_langid import _test_hooks
from platform_langid.testing import (
    FakeAudioLoader,
    FakeClassificationOutput,
    FakeModel,
    FakeModelConfig,
    FakeProcessor,
    FakeProcessorOutput,
    FakeSpokenLanguageDetector,
    FakeTensor,
    make_fake_detector_factory,
    make_fake_model_factory,
    make_fake_processor_factory,
    reset_hooks,
    set_production_hooks,
)
from platform_langid.types import DetectorConfig


class TestFakeTensor:
    """Tests for FakeTensor class."""

    def test_argmax_first_max(self) -> None:
        """Argmax returns index of first maximum."""
        tensor = FakeTensor([0.1, 0.9, 0.5])
        result = tensor.argmax()
        assert result.item() == 1.0

    def test_argmax_single_value(self) -> None:
        """Argmax with single value returns 0."""
        tensor = FakeTensor([0.5])
        result = tensor.argmax()
        assert result.item() == 0.0

    def test_softmax_sums_to_one(self) -> None:
        """Softmax output sums to 1."""
        tensor = FakeTensor([1.0, 2.0, 3.0])
        result = tensor.softmax()
        total = sum(result._values)
        assert total == pytest.approx(1.0)

    def test_softmax_preserves_order(self) -> None:
        """Softmax preserves relative ordering."""
        tensor = FakeTensor([1.0, 3.0, 2.0])
        result = tensor.softmax()
        assert result._values[1] > result._values[2] > result._values[0]

    def test_item_returns_first(self) -> None:
        """Item returns first value."""
        tensor = FakeTensor([0.42, 0.58])
        assert tensor.item() == 0.42

    def test_getitem(self) -> None:
        """Getitem returns tensor with indexed value."""
        tensor = FakeTensor([0.1, 0.2, 0.3])
        result = tensor[1]
        assert result.item() == 0.2

    def test_shape_1d(self) -> None:
        """Shape returns 1D tuple for non-batched tensor."""
        tensor = FakeTensor([0.1, 0.2, 0.3])
        assert tensor.shape == (3,)

    def test_shape_batched(self) -> None:
        """Shape returns 2D tuple for batched tensor."""
        tensor = FakeTensor([0.1, 0.2, 0.3], is_batched=True)
        assert tensor.shape == (1, 3)


class TestFakeClassificationOutput:
    """Tests for FakeClassificationOutput class."""

    def test_has_logits(self) -> None:
        """Has logits attribute."""
        logits = FakeTensor([0.5])
        output = FakeClassificationOutput(logits)
        assert output.logits is logits


class TestFakeProcessorOutput:
    """Tests for FakeProcessorOutput class."""

    def test_has_input_values(self) -> None:
        """Has input_values attribute."""
        values = FakeTensor([0.0])
        output = FakeProcessorOutput(values)
        assert output.input_values is values


class TestFakeModelConfig:
    """Tests for FakeModelConfig class."""

    def test_has_id2label(self) -> None:
        """Has id2label attribute."""
        mapping = {0: "en", 1: "vi"}
        config = FakeModelConfig(mapping)
        assert config.id2label == mapping


class TestFakeModel:
    """Tests for FakeModel class."""

    def test_default_labels(self) -> None:
        """Uses default id2label mapping."""
        model = FakeModel()
        assert model.config.id2label == {0: "en", 1: "vi", 2: "es"}

    def test_custom_labels(self) -> None:
        """Uses custom id2label mapping."""
        model = FakeModel(id2label={0: "fr", 1: "de"})
        assert model.config.id2label == {0: "fr", 1: "de"}

    def test_call_returns_output_with_correct_logits(self) -> None:
        """Call returns classification output with logits that can be processed."""
        model = FakeModel(predicted_id=1, confidence=0.9, id2label={0: "en", 1: "vi", 2: "es"})
        output = model(FakeTensor([0.0]))
        # Verify logits work correctly: argmax should return predicted_id
        predicted_idx = int(output.logits.argmax().item())
        assert predicted_idx == 1

    def test_predicted_class_via_argmax(self) -> None:
        """Predicted class matches predicted_id via argmax."""
        model = FakeModel(predicted_id=2, id2label={0: "a", 1: "b", 2: "c"})
        output = model(FakeTensor([0.0]))
        predicted = output.logits.argmax().item()
        assert int(predicted) == 2

    def test_confidence_via_softmax(self) -> None:
        """Confidence approximately matches via softmax."""
        model = FakeModel(predicted_id=0, confidence=0.9, id2label={0: "en", 1: "vi"})
        output = model(FakeTensor([0.0]))
        probs = output.logits.softmax()
        conf = probs[0].item()
        # Allow some tolerance due to softmax approximation
        assert conf == pytest.approx(0.9, abs=0.15)

    def test_to_returns_self(self) -> None:
        """To method returns self."""
        model = FakeModel()
        result = model.to("cuda")
        assert result is model

    def test_single_class_model(self) -> None:
        """Single class model always produces 100% confidence."""
        model = FakeModel(predicted_id=0, confidence=0.5, id2label={0: "en"})
        output = model(FakeTensor([0.0]))
        probs = output.logits.softmax()
        conf = probs[0][0].item()
        # Single class softmax is always 1.0
        assert conf == pytest.approx(1.0, abs=0.01)

    def test_confidence_at_one(self) -> None:
        """Model with confidence=1.0 produces near-100% probability."""
        model = FakeModel(predicted_id=0, confidence=1.0, id2label={0: "en", 1: "vi"})
        output = model(FakeTensor([0.0]))
        probs = output.logits.softmax()
        conf = probs[0][0].item()
        assert conf > 0.99

    def test_confidence_at_zero(self) -> None:
        """Model with confidence=0.0 produces near-0% probability for target."""
        model = FakeModel(predicted_id=0, confidence=0.0, id2label={0: "en", 1: "vi"})
        output = model(FakeTensor([0.0]))
        probs = output.logits.softmax()
        conf = probs[0][0].item()
        assert conf < 0.01


class TestFakeProcessor:
    """Tests for FakeProcessor class."""

    def test_call_returns_output_with_usable_input_values(self) -> None:
        """Call returns processor output with usable input_values."""
        processor = FakeProcessor()
        output = processor([0.0, 0.1], 16000)
        # Verify input_values has expected behavior: item() returns float value
        value = output.input_values.item()
        assert value == 0.0

    def test_call_ignores_args(self) -> None:
        """Call ignores arguments and returns dummy output."""
        processor = FakeProcessor()
        output = processor([1.0, 2.0, 3.0], 44100, "pt")
        assert output.input_values.item() == 0.0


class TestFakeSpokenLanguageDetector:
    """Tests for FakeSpokenLanguageDetector class."""

    def test_detect_returns_configured_language(self) -> None:
        """Detect returns configured language."""
        detector = FakeSpokenLanguageDetector(language="vi")
        result = detector.detect(b"\x00\x01", 16000)
        assert result["language"] == "vi"

    def test_detect_returns_configured_confidence(self) -> None:
        """Detect returns configured confidence."""
        detector = FakeSpokenLanguageDetector(confidence=0.88)
        result = detector.detect(b"\x00\x01", 16000)
        assert result["confidence"] == 0.88

    def test_detect_returns_configured_model_id(self) -> None:
        """Detect returns configured model_id."""
        detector = FakeSpokenLanguageDetector(model_id="my-model")
        result = detector.detect(b"\x00\x01", 16000)
        assert result["model_id"] == "my-model"

    def test_detect_empty_raises(self) -> None:
        """Detect raises for empty audio."""
        detector = FakeSpokenLanguageDetector()
        with pytest.raises(ValueError, match="cannot be empty"):
            detector.detect(b"", 16000)

    def test_detect_increments_call_count(self) -> None:
        """Detect increments call_count."""
        detector = FakeSpokenLanguageDetector()
        assert detector.call_count == 0
        detector.detect(b"\x00", 16000)
        assert detector.call_count == 1
        detector.detect(b"\x00", 16000)
        assert detector.call_count == 2


class TestFakeAudioLoader:
    """Tests for FakeAudioLoader class."""

    def test_default_samples(self) -> None:
        """Returns default samples when none configured."""
        loader = FakeAudioLoader()
        samples = loader(b"\x00\x01", 16000)
        assert list(samples) == [0.0, 0.1, -0.1]

    def test_custom_samples(self) -> None:
        """Returns configured samples."""
        loader = FakeAudioLoader(samples=[1.0, 2.0])
        samples = loader(b"\x00", 16000)
        assert list(samples) == [1.0, 2.0]

    def test_records_calls(self) -> None:
        """Records all calls."""
        loader = FakeAudioLoader()
        loader(b"\x00\x01", 16000)
        loader(b"\x02\x03", 44100)
        assert len(loader.calls) == 2
        assert loader.calls[0] == (b"\x00\x01", 16000)
        assert loader.calls[1] == (b"\x02\x03", 44100)


class TestHookManagement:
    """Tests for hook management functions."""

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_set_production_hooks(self) -> None:
        """set_production_hooks restores defaults."""
        # Modify hooks
        _test_hooks.model_factory = make_fake_model_factory()
        _test_hooks.processor_factory = make_fake_processor_factory()

        set_production_hooks()

        assert _test_hooks.model_factory is _test_hooks._default_model_factory
        assert _test_hooks.processor_factory is _test_hooks._default_processor_factory
        assert _test_hooks.audio_loader is _test_hooks._default_audio_loader
        assert _test_hooks.detector_factory is _test_hooks._default_detector_factory

    def test_reset_hooks(self) -> None:
        """reset_hooks restores defaults."""
        _test_hooks.detector_factory = make_fake_detector_factory()

        reset_hooks()

        assert _test_hooks.detector_factory is _test_hooks._default_detector_factory


class TestMakeFakeDetectorFactory:
    """Tests for make_fake_detector_factory function."""

    def test_creates_factory(self) -> None:
        """Creates factory that returns FakeSpokenLanguageDetector."""
        factory = make_fake_detector_factory(language="fr", confidence=0.7)
        config = DetectorConfig(model_id="x", device="cpu", confidence_threshold=0.0)
        detector = factory(config)
        result = detector.detect(b"\x00", 16000)
        assert result["language"] == "fr"
        assert result["confidence"] == 0.7


class TestMakeFakeModelFactory:
    """Tests for make_fake_model_factory function."""

    def test_creates_factory(self) -> None:
        """Creates factory that returns FakeModel."""
        factory = make_fake_model_factory(predicted_id=1, confidence=0.8)
        model = factory("any-model-id")
        output = model(FakeTensor([0.0]))
        predicted = int(output.logits.argmax().item())
        assert predicted == 1


class TestMakeFakeProcessorFactory:
    """Tests for make_fake_processor_factory function."""

    def test_creates_factory_with_usable_processor(self) -> None:
        """Creates factory that returns processor with usable output."""
        factory = make_fake_processor_factory()
        processor = factory("any-model-id")
        output = processor([0.0], 16000)
        # Verify output has expected behavior: item() returns float value
        value = output.input_values.item()
        assert value == 0.0
