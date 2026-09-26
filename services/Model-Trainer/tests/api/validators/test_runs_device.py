"""Tests for device and precision validation in runs.py."""

from __future__ import annotations

from platform_core.json_utils import JSONValue
from platform_ml import RequestedDevice, RequestedPrecision

from model_trainer.api.validators.runs import _decode_train_request


def _base_gpt2_payload() -> dict[str, JSONValue]:
    """Return base payload for gpt2 tests."""
    return {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "corpus_format": "lines",
        "tokenizer_id": "tok",
        "user_id": 0,
    }


class TestDeviceValidation:
    """Tests for device validation."""

    def test_device_auto_default(self) -> None:
        """Device defaults to AUTO."""
        out = _decode_train_request(_base_gpt2_payload())
        assert out["device"] is RequestedDevice.AUTO

    def test_every_device_word_narrows_to_its_member(self) -> None:
        """Each device word, 'auto' included, decodes to its own member."""
        for device in RequestedDevice:
            payload = _base_gpt2_payload()
            payload["device"] = device.value
            assert _decode_train_request(payload)["device"] is device


class TestPrecisionValidation:
    """Tests for precision validation."""

    def test_precision_auto_default(self) -> None:
        """Precision defaults to AUTO."""
        out = _decode_train_request(_base_gpt2_payload())
        assert out["precision"] is RequestedPrecision.AUTO

    def test_every_precision_word_narrows_to_its_member(self) -> None:
        """Each precision word, 'auto' included, decodes to its own member."""
        for precision in RequestedPrecision:
            payload = _base_gpt2_payload()
            payload["precision"] = precision.value
            assert _decode_train_request(payload)["precision"] is precision
