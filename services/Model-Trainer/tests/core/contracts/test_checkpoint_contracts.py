"""Tests for the training-checkpoint metadata contract and codecs."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from model_trainer.core.contracts.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    EpochSummaryRecord,
    TrainingCheckpointMeta,
    decode_epoch_summary,
    decode_model_train_config,
    decode_training_checkpoint_meta,
    encode_epoch_summary,
    encode_model_train_config,
    encode_training_checkpoint_meta,
    model_train_config_mismatches,
)
from model_trainer.core.contracts.model import LoraConfig, ModelTrainConfig


def _make_cfg() -> ModelTrainConfig:
    """Build a fully-populated resolved training config."""
    return {
        "model_family": "hf_lm",
        "model_size": "medium",
        "max_seq_len": 512,
        "num_epochs": 20,
        "batch_size": 8,
        "learning_rate": 5e-05,
        "tokenizer_id": None,
        "corpus_path": "/data/corpus_cache/abc123.txt",
        "corpus_format": "lines",
        "holdout_fraction": 0.0,
        "seed": 42,
        "pretrained_run_id": None,
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "optimizer": "adamw",
        "device": "cuda",
        "precision": "fp32",
        "data_num_workers": 2,
        "data_pin_memory": True,
        "early_stopping_patience": 0,
        "test_split_ratio": 0.0,
        "finetune_lr_cap": 0.0001,
        "loss_mask_prefix_separator": " @@HUB@@ ",
        "finetuning_strategy": "full",
        "hub_model_id": "gpt2-medium",
        "lora": None,
        "quantization": None,
        "gguf_export": None,
    }


def _make_summary() -> EpochSummaryRecord:
    """Build one epoch summary record."""
    return {
        "epoch": 3,
        "train_loss": 1.25,
        "train_ppl": 3.49,
        "val_loss": 1.5,
        "val_ppl": 4.48,
    }


def _make_meta() -> TrainingCheckpointMeta:
    """Build a fully-populated checkpoint metadata record."""
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": "run-abc",
        "epochs_completed": 4,
        "global_step": 5388,
        "last_loss": 1.01,
        "best_val_loss": 1.5,
        "epochs_no_improve": 1,
        "best_saved": True,
        "total_samples_processed": 43104,
        "total_tokens_processed": 22069248,
        "elapsed_seconds": 812.5,
        "started_at_iso": "2026-08-18T05:25:52",
        "epoch_summaries": [_make_summary()],
        "config": _make_cfg(),
    }


class TestEpochSummaryCodec:
    """Round-trip and validation for EpochSummaryRecord."""

    def test_round_trip(self) -> None:
        record = _make_summary()
        assert decode_epoch_summary(encode_epoch_summary(record)) == record

    def test_missing_field_raises(self) -> None:
        encoded = encode_epoch_summary(_make_summary())
        del encoded["train_ppl"]
        with pytest.raises(JSONTypeError):
            decode_epoch_summary(encoded)


class TestModelTrainConfigCodec:
    """Round-trip and validation for the config codec."""

    def test_round_trip_full(self) -> None:
        cfg = _make_cfg()
        assert decode_model_train_config(encode_model_train_config(cfg)) == cfg

    def test_round_trip_with_nested_configs(self) -> None:
        lora: LoraConfig = {
            "enabled": True,
            "r": 8,
            "lora_alpha": 8,
            "lora_dropout": 0.05,
            "target_modules": ("q_proj", "v_proj"),
            "bias": "none",
        }
        cfg = _make_cfg()
        cfg["lora"] = lora
        cfg["quantization"] = {
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": False,
        }
        cfg["gguf_export"] = {"enabled": True, "output_type": "q8_0"}
        assert decode_model_train_config(encode_model_train_config(cfg)) == cfg

    def test_cpu_and_reduced_precision_round_trip(self) -> None:
        cfg = _make_cfg()
        cfg["device"] = "cpu"
        cfg["precision"] = "fp16"
        assert decode_model_train_config(encode_model_train_config(cfg)) == cfg
        cfg["precision"] = "bf16"
        assert decode_model_train_config(encode_model_train_config(cfg)) == cfg

    def test_auto_device_rejected(self) -> None:
        encoded = encode_model_train_config(_make_cfg())
        encoded["device"] = "auto"
        with pytest.raises(JSONTypeError, match="device"):
            decode_model_train_config(encoded)

    def test_auto_precision_rejected(self) -> None:
        encoded = encode_model_train_config(_make_cfg())
        encoded["precision"] = "auto"
        with pytest.raises(JSONTypeError, match="precision"):
            decode_model_train_config(encoded)


class TestConfigMismatches:
    """Field-level mismatch reporting for the resume fingerprint."""

    def test_identical_configs_have_no_mismatches(self) -> None:
        assert model_train_config_mismatches(_make_cfg(), _make_cfg()) == []

    def test_differing_fields_are_named_and_sorted(self) -> None:
        expected = _make_cfg()
        actual = _make_cfg()
        actual["seed"] = 43
        actual["batch_size"] = 16
        assert model_train_config_mismatches(expected, actual) == ["batch_size", "seed"]

    def test_nested_config_difference_is_detected(self) -> None:
        expected = _make_cfg()
        actual = _make_cfg()
        actual["lora"] = {
            "enabled": True,
            "r": 8,
            "lora_alpha": 8,
            "lora_dropout": 0.05,
            "target_modules": ("q_proj",),
            "bias": "none",
        }
        assert model_train_config_mismatches(expected, actual) == ["lora"]

    def test_tuple_versus_list_never_false_positives(self) -> None:
        lora: LoraConfig = {
            "enabled": True,
            "r": 8,
            "lora_alpha": 8,
            "lora_dropout": 0.05,
            "target_modules": ("q_proj", "v_proj"),
            "bias": "none",
        }
        expected = _make_cfg()
        expected["lora"] = lora
        actual = _make_cfg()
        actual["lora"] = {
            "enabled": True,
            "r": 8,
            "lora_alpha": 8,
            "lora_dropout": 0.05,
            "target_modules": ("q_proj", "v_proj"),
            "bias": "none",
        }
        assert model_train_config_mismatches(expected, actual) == []


class TestTrainingCheckpointMetaCodec:
    """Round-trip and validation for the checkpoint metadata codec."""

    def test_round_trip_full(self) -> None:
        meta = _make_meta()
        assert decode_training_checkpoint_meta(encode_training_checkpoint_meta(meta)) == meta

    def test_round_trip_with_no_validation_state(self) -> None:
        meta = _make_meta()
        meta["best_val_loss"] = None
        meta["best_saved"] = False
        meta["epoch_summaries"] = []
        assert decode_training_checkpoint_meta(encode_training_checkpoint_meta(meta)) == meta

    def test_missing_field_raises(self) -> None:
        encoded = encode_training_checkpoint_meta(_make_meta())
        del encoded["global_step"]
        with pytest.raises(JSONTypeError):
            decode_training_checkpoint_meta(encoded)

    def test_non_object_epoch_summary_entry_raises(self) -> None:
        encoded = encode_training_checkpoint_meta(_make_meta())
        encoded["epoch_summaries"] = ["not-an-object"]
        with pytest.raises(JSONTypeError, match=r"epoch_summaries\[0\]"):
            decode_training_checkpoint_meta(encoded)

    def test_missing_config_raises(self) -> None:
        encoded: JSONObject = encode_training_checkpoint_meta(_make_meta())
        del encoded["config"]
        with pytest.raises(JSONTypeError, match="config"):
            decode_training_checkpoint_meta(encoded)
