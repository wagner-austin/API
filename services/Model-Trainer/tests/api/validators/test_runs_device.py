"""Tests for device and precision validation in runs.py."""

from __future__ import annotations

from platform_core.json_utils import JSONValue

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
        """Test device defaults to 'auto'."""
        payload = _base_gpt2_payload()
        out = _decode_train_request(payload)
        assert out["device"] == "auto"

    def test_device_cuda(self) -> None:
        """Test device='cuda'."""
        payload = _base_gpt2_payload()
        payload["device"] = "cuda"
        out = _decode_train_request(payload)
        assert out["device"] == "cuda"

    def test_device_cpu(self) -> None:
        """Test device='cpu'."""
        payload = _base_gpt2_payload()
        payload["device"] = "cpu"
        out = _decode_train_request(payload)
        assert out["device"] == "cpu"

    def test_device_auto_explicit(self) -> None:
        """Test device='auto' explicit."""
        payload = _base_gpt2_payload()
        payload["device"] = "auto"
        out = _decode_train_request(payload)
        assert out["device"] == "auto"


class TestPrecisionValidation:
    """Tests for precision validation."""

    def test_precision_auto_default(self) -> None:
        """Test precision defaults to 'auto'."""
        payload = _base_gpt2_payload()
        out = _decode_train_request(payload)
        assert out["precision"] == "auto"

    def test_precision_fp32(self) -> None:
        """Test precision='fp32'."""
        payload = _base_gpt2_payload()
        payload["precision"] = "fp32"
        out = _decode_train_request(payload)
        assert out["precision"] == "fp32"

    def test_precision_fp16(self) -> None:
        """Test precision='fp16'."""
        payload = _base_gpt2_payload()
        payload["precision"] = "fp16"
        out = _decode_train_request(payload)
        assert out["precision"] == "fp16"

    def test_precision_bf16(self) -> None:
        """Test precision='bf16'."""
        payload = _base_gpt2_payload()
        payload["precision"] = "bf16"
        out = _decode_train_request(payload)
        assert out["precision"] == "bf16"

    def test_precision_auto_explicit(self) -> None:
        """Test precision='auto' explicit."""
        payload = _base_gpt2_payload()
        payload["precision"] = "auto"
        out = _decode_train_request(payload)
        assert out["precision"] == "auto"
