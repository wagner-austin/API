"""Queue payload codecs: decode side and errors."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.json_utils import JSONObject, JSONTypeError

from model_trainer.core.contracts.queue import TrainJobPayload, TrainRequestPayload
from model_trainer.core.contracts.queue_encoding import (
    decode_train_job_payload,
    decode_train_request_payload,
    encode_train_job_payload,
    encode_train_request_payload,
)


class TestTrainRequestPayloadEncoding:
    """Tests for TrainRequestPayload encode/decode functions."""

    def _make_minimal_payload(self) -> TrainRequestPayload:
        """Create a minimal valid TrainRequestPayload."""
        return {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 512,
            "num_epochs": 1,
            "batch_size": 4,
            "learning_rate": 5e-5,
            "corpus_file_id": "test-corpus-id",
            "corpus_format": "lines",
            "tokenizer_id": "tok-123",
            "holdout_fraction": 0.1,
            "seed": 42,
            "pretrained_run_id": None,
            "freeze_embed": False,
            "gradient_clipping": 1.0,
            "optimizer": "adamw",
            "device": "cpu",
            "precision": "fp32",
            "data_num_workers": None,
            "data_pin_memory": None,
            "early_stopping_patience": 3,
            "test_split_ratio": 0.1,
            "finetune_lr_cap": 1e-4,
            "loss_mask_prefix_separator": None,
            "hub_model_id": None,
            "finetuning_strategy": "full",
            "lora": None,
            "cartridge": None,
            "quantization": None,
            "gguf_export": None,
        }

    def test_encode_decode_roundtrip_minimal(self) -> None:
        """Test roundtrip with minimal payload (no nested configs)."""
        payload = self._make_minimal_payload()
        encoded = encode_train_request_payload(payload)
        decoded = decode_train_request_payload(encoded)

        assert decoded["model_family"] == "gpt2"
        assert decoded["model_size"] == "small"
        assert decoded["finetuning_strategy"] == "full"
        assert decoded["lora"] is None
        assert decoded["quantization"] is None

    def test_roundtrip_preserves_the_loss_mask_separator(self) -> None:
        """Whitespace in the separator is significant, so it must survive the
        queue hop byte for byte."""
        payload = self._make_minimal_payload()
        payload["loss_mask_prefix_separator"] = " | "
        decoded = decode_train_request_payload(encode_train_request_payload(payload))
        assert decoded["loss_mask_prefix_separator"] == " | "

    def test_decode_rejects_an_empty_loss_mask_separator(self) -> None:
        """An empty separator masks nothing while claiming a masked arm, so the
        queue boundary rejects it exactly as the API edge does."""
        payload = self._make_minimal_payload()
        encoded = encode_train_request_payload(payload)
        encoded["loss_mask_prefix_separator"] = ""
        with pytest.raises(JSONTypeError, match="must not be empty"):
            decode_train_request_payload(encoded)

    def test_encode_decode_roundtrip_with_lora(self) -> None:
        """Test roundtrip with LoRA config."""
        payload = self._make_minimal_payload()
        payload["finetuning_strategy"] = "lora"
        payload["lora"] = {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ("q_proj", "v_proj"),
            "bias": "none",
        }

        encoded = encode_train_request_payload(payload)
        decoded = decode_train_request_payload(encoded)

        assert decoded["finetuning_strategy"] == "lora"
        lora = decoded["lora"]
        assert lora == {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ("q_proj", "v_proj"),
            "bias": "none",
        }

    def test_encode_decode_roundtrip_with_quantization(self) -> None:
        """Test roundtrip with quantization config."""
        payload = self._make_minimal_payload()
        payload["finetuning_strategy"] = "qlora"
        payload["lora"] = {
            "enabled": True,
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": ("q_proj",),
            "bias": "none",
        }
        payload["quantization"] = {
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": False,
        }

        encoded = encode_train_request_payload(payload)
        decoded = decode_train_request_payload(encoded)

        quantization = decoded["quantization"]
        assert quantization == {
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": False,
        }

    def test_decode_all_model_families(self) -> None:
        """Test decoding all valid model family values."""
        for family in ("gpt2", "llama", "qwen", "char_lstm", "hf_lm"):
            encoded = encode_train_request_payload(self._make_minimal_payload())
            encoded["model_family"] = family
            decoded = decode_train_request_payload(encoded)
            assert decoded["model_family"] == family

    def test_decode_invalid_model_family(self) -> None:
        """Test that invalid model family raises JSONTypeError."""
        encoded = encode_train_request_payload(self._make_minimal_payload())
        encoded["model_family"] = "invalid"
        with pytest.raises(JSONTypeError, match=r"model_family.*must be"):
            decode_train_request_payload(encoded)

    def test_decode_all_optimizers(self) -> None:
        """Test decoding all valid optimizer values."""
        for opt in ("adamw", "adam", "sgd"):
            encoded = encode_train_request_payload(self._make_minimal_payload())
            encoded["optimizer"] = opt
            decoded = decode_train_request_payload(encoded)
            assert decoded["optimizer"] == opt

    def test_decode_invalid_optimizer(self) -> None:
        """Test that invalid optimizer raises JSONTypeError."""
        encoded = encode_train_request_payload(self._make_minimal_payload())
        encoded["optimizer"] = "invalid"
        with pytest.raises(JSONTypeError, match=r"optimizer.*must be"):
            decode_train_request_payload(encoded)

    def test_decode_all_devices(self) -> None:
        """Test decoding all valid device values."""
        for device in ("cpu", "cuda", "auto"):
            encoded = encode_train_request_payload(self._make_minimal_payload())
            encoded["device"] = device
            decoded = decode_train_request_payload(encoded)
            assert decoded["device"] == device

    def test_decode_invalid_device(self) -> None:
        """Test that invalid device raises JSONTypeError."""
        encoded = encode_train_request_payload(self._make_minimal_payload())
        encoded["device"] = "tpu"
        with pytest.raises(JSONTypeError, match=r"device.*must be"):
            decode_train_request_payload(encoded)

    def test_decode_all_precisions(self) -> None:
        """Test decoding all valid precision values."""
        for precision in ("fp32", "fp16", "bf16", "auto"):
            encoded = encode_train_request_payload(self._make_minimal_payload())
            encoded["precision"] = precision
            decoded = decode_train_request_payload(encoded)
            assert decoded["precision"] == precision

    def test_decode_invalid_precision(self) -> None:
        """Test that invalid precision raises JSONTypeError."""
        encoded = encode_train_request_payload(self._make_minimal_payload())
        encoded["precision"] = "int8"
        with pytest.raises(JSONTypeError, match=r"precision.*must be"):
            decode_train_request_payload(encoded)

    def test_decode_all_finetuning_strategies(self) -> None:
        """Test decoding all valid finetuning strategy values."""
        for strategy in ("full", "lora", "qlora"):
            encoded = encode_train_request_payload(self._make_minimal_payload())
            encoded["finetuning_strategy"] = strategy
            decoded = decode_train_request_payload(encoded)
            assert decoded["finetuning_strategy"] == strategy

    def test_decode_invalid_finetuning_strategy(self) -> None:
        """A queued payload naming no declared strategy carries its own code.

        Not a ``JSONTypeError``: the payload's shape is correct and its value
        is not, and the queue path now reports that with the same
        ``STRATEGY_NAME_UNKNOWN`` the request path uses, so one code covers
        the mistake wherever it enters.
        """
        encoded = encode_train_request_payload(self._make_minimal_payload())
        encoded["finetuning_strategy"] = "freeze"
        with pytest.raises(AppError) as excinfo:
            decode_train_request_payload(encoded)
        assert excinfo.value.code is ModelTrainerErrorCode.STRATEGY_NAME_UNKNOWN

    def test_decode_data_pin_memory_bool(self) -> None:
        """Test decoding data_pin_memory as bool."""
        payload = self._make_minimal_payload()
        payload["data_pin_memory"] = True
        encoded = encode_train_request_payload(payload)
        decoded = decode_train_request_payload(encoded)
        assert decoded["data_pin_memory"] is True

    def test_decode_data_pin_memory_invalid(self) -> None:
        """Test that non-bool data_pin_memory raises JSONTypeError."""
        encoded = encode_train_request_payload(self._make_minimal_payload())
        encoded["data_pin_memory"] = "yes"
        with pytest.raises(JSONTypeError, match=r"data_pin_memory.*must be a boolean or null"):
            decode_train_request_payload(encoded)

    def test_decode_lora_not_dict(self) -> None:
        """Test that non-dict lora field raises JSONTypeError."""
        encoded = encode_train_request_payload(self._make_minimal_payload())
        encoded["lora"] = "invalid"
        with pytest.raises(JSONTypeError, match=r"lora.*must be an object or null"):
            decode_train_request_payload(encoded)

    def test_decode_quantization_not_dict(self) -> None:
        """Test that non-dict quantization field raises JSONTypeError."""
        encoded = encode_train_request_payload(self._make_minimal_payload())
        encoded["quantization"] = []
        with pytest.raises(JSONTypeError, match=r"quantization.*must be an object or null"):
            decode_train_request_payload(encoded)

    def test_encode_preserves_all_fields(self) -> None:
        """Test that encoding preserves all fields."""
        payload = self._make_minimal_payload()
        payload["pretrained_run_id"] = "prev-run-123"
        payload["data_num_workers"] = 4
        payload["data_pin_memory"] = True
        payload["hub_model_id"] = "gpt2"

        encoded = encode_train_request_payload(payload)

        assert encoded["pretrained_run_id"] == "prev-run-123"
        assert encoded["data_num_workers"] == 4
        assert encoded["data_pin_memory"] is True
        assert encoded["hub_model_id"] == "gpt2"


class TestTrainJobPayloadEncoding:
    """Tests for TrainJobPayload encode/decode functions."""

    def _make_minimal_request(self) -> TrainRequestPayload:
        """Create a minimal valid TrainRequestPayload."""
        return {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 512,
            "num_epochs": 1,
            "batch_size": 4,
            "learning_rate": 5e-5,
            "corpus_file_id": "test-corpus-id",
            "corpus_format": "lines",
            "tokenizer_id": "tok-123",
            "holdout_fraction": 0.1,
            "seed": 42,
            "pretrained_run_id": None,
            "freeze_embed": False,
            "gradient_clipping": 1.0,
            "optimizer": "adamw",
            "device": "cpu",
            "precision": "fp32",
            "data_num_workers": None,
            "data_pin_memory": None,
            "early_stopping_patience": 3,
            "test_split_ratio": 0.1,
            "finetune_lr_cap": 1e-4,
            "loss_mask_prefix_separator": None,
            "hub_model_id": None,
            "finetuning_strategy": "full",
            "lora": None,
            "cartridge": None,
            "quantization": None,
            "gguf_export": None,
        }

    def _make_job_payload(self) -> TrainJobPayload:
        """Create a valid TrainJobPayload."""
        return {
            "run_id": "run-12345",
            "user_id": 42,
            "resume": False,
            "request": self._make_minimal_request(),
        }

    def test_resume_true_round_trips(self) -> None:
        """A resume execution's flag survives encode and decode."""
        payload = self._make_job_payload()
        payload["resume"] = True
        encoded = encode_train_job_payload(payload)
        assert encoded["resume"] is True
        assert decode_train_job_payload(encoded)["resume"] is True

    def test_decode_missing_resume_raises(self) -> None:
        """The resume flag is required; an old-shape payload is refused."""
        encoded = encode_train_job_payload(self._make_job_payload())
        del encoded["resume"]
        with pytest.raises(JSONTypeError, match="resume"):
            _ = decode_train_job_payload(encoded)

    def test_encode_decode_roundtrip(self) -> None:
        """Test that encode and decode are inverse operations."""
        payload = self._make_job_payload()
        encoded = encode_train_job_payload(payload)
        decoded = decode_train_job_payload(encoded)

        assert decoded["run_id"] == "run-12345"
        assert decoded["user_id"] == 42
        assert decoded["request"]["model_family"] == "gpt2"
        assert decoded["request"]["corpus_file_id"] == "test-corpus-id"

    def test_encode_produces_json_object(self) -> None:
        """Test that encoding produces a plain dict suitable for JSON."""
        payload = self._make_job_payload()
        encoded = encode_train_job_payload(payload)

        assert encoded["run_id"] == "run-12345"
        assert encoded["user_id"] == 42
        # Decode back to verify the nested request was properly encoded
        decoded = decode_train_job_payload(encoded)
        assert decoded["request"]["model_family"] == "gpt2"
        assert decoded["request"]["corpus_file_id"] == "test-corpus-id"

    def test_decode_missing_run_id(self) -> None:
        """Test that missing run_id raises JSONTypeError."""
        encoded: JSONObject = {
            "user_id": 42,
            "request": encode_train_request_payload(self._make_minimal_request()),
        }
        with pytest.raises(JSONTypeError, match=r"Missing required field 'run_id'"):
            decode_train_job_payload(encoded)

    def test_decode_missing_user_id(self) -> None:
        """Test that missing user_id raises JSONTypeError."""
        encoded: JSONObject = {
            "run_id": "run-123",
            "request": encode_train_request_payload(self._make_minimal_request()),
        }
        with pytest.raises(JSONTypeError, match=r"Missing required field 'user_id'"):
            decode_train_job_payload(encoded)

    def test_decode_missing_request(self) -> None:
        """Test that missing request raises JSONTypeError."""
        encoded: JSONObject = {
            "run_id": "run-123",
            "user_id": 42,
        }
        with pytest.raises(JSONTypeError, match=r"Missing required field 'request'"):
            decode_train_job_payload(encoded)

    def test_decode_request_not_dict(self) -> None:
        """Test that non-dict request raises JSONTypeError."""
        encoded: JSONObject = {
            "run_id": "run-123",
            "user_id": 42,
            "request": "invalid",
        }
        with pytest.raises(JSONTypeError, match=r"request.*must be an object"):
            decode_train_job_payload(encoded)

    def test_decode_with_nested_lora(self) -> None:
        """Test decoding a full payload with nested LoRA config."""
        request = self._make_minimal_request()
        request["finetuning_strategy"] = "lora"
        request["lora"] = {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ("q_proj", "v_proj"),
            "bias": "none",
        }
        payload: TrainJobPayload = {
            "run_id": "lora-run",
            "user_id": 99,
            "resume": False,
            "request": request,
        }
        encoded = encode_train_job_payload(payload)
        decoded = decode_train_job_payload(encoded)

        assert decoded["run_id"] == "lora-run"
        assert decoded["user_id"] == 99
        lora = decoded["request"]["lora"]
        assert lora == {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ("q_proj", "v_proj"),
            "bias": "none",
        }
