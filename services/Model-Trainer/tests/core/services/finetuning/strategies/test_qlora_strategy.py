"""Tests for the QLoRAStrategy."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from model_trainer.core.contracts.model import LoraConfig, ModelTrainConfig, QuantizationConfig
from model_trainer.core.services.finetuning.strategies._test_hooks import Hooks, reset_hooks
from model_trainer.core.services.finetuning.strategies.qlora import (
    QLoRAStrategy,
    _require_lora_config,
    _require_quantization_config,
    create_qlora_strategy,
)
from model_trainer.core.types import LMModelProto
from tests.core.services.finetuning.testing import FakeModel


def make_lora_config() -> LoraConfig:
    """Create a valid LoraConfig for testing."""
    return {
        "enabled": True,
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ("q_proj", "v_proj"),
        "bias": "none",
    }


def make_quantization_config() -> QuantizationConfig:
    """Create a valid QuantizationConfig for testing."""
    return {
        "load_in_4bit": True,
        "load_in_8bit": False,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": False,
    }


def make_test_config(
    lora: LoraConfig | None = None,
    quantization: QuantizationConfig | None = None,
) -> ModelTrainConfig:
    """Create a minimal ModelTrainConfig for testing."""
    return {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 128,
        "num_epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "tokenizer_id": "test-tok",
        "corpus_path": "/tmp/corpus",
        "corpus_format": "lines",
        "holdout_fraction": 0.1,
        "seed": 42,
        "pretrained_run_id": None,
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "optimizer": "adamw",
        "device": "cuda",
        "precision": "fp16",
        "data_num_workers": 0,
        "data_pin_memory": False,
        "early_stopping_patience": 3,
        "test_split_ratio": 0.1,
        "finetune_lr_cap": 0.0001,
        "loss_mask_prefix_separator": None,
        "finetuning_strategy": "qlora",
        "hub_model_id": "meta/llama-7b",
        "lora": lora,
        "cartridge": None,
        "quantization": quantization,
        "gguf_export": None,
    }


@pytest.fixture(autouse=True)
def _reset_hooks() -> Generator[None, None, None]:
    """Reset hooks before and after each test."""
    reset_hooks()
    yield
    reset_hooks()


class TestRequireLoraConfigQLoRA:
    """Tests for _require_lora_config helper in QLoRA module."""

    def test_returns_lora_config_when_valid(self) -> None:
        """Test that valid config is returned."""
        lora_cfg = make_lora_config()
        cfg = make_test_config(lora=lora_cfg, quantization=make_quantization_config())
        result = _require_lora_config(cfg)
        assert result["r"] == 16

    def test_raises_when_lora_is_none(self) -> None:
        """Test that ValueError is raised when lora is None."""
        cfg = make_test_config(lora=None, quantization=make_quantization_config())
        with pytest.raises(ValueError, match="QLoRA strategy requires lora config"):
            _require_lora_config(cfg)

    def test_raises_when_lora_disabled(self) -> None:
        """Test that ValueError is raised when lora.enabled is False."""
        lora_cfg = make_lora_config()
        lora_cfg["enabled"] = False
        cfg = make_test_config(lora=lora_cfg, quantization=make_quantization_config())
        with pytest.raises(ValueError, match=r"lora\.enabled=True"):
            _require_lora_config(cfg)


class TestRequireQuantizationConfig:
    """Tests for _require_quantization_config helper."""

    def test_returns_quant_config_when_valid_4bit(self) -> None:
        """Test that valid 4-bit config is returned."""
        quant_cfg = make_quantization_config()
        cfg = make_test_config(lora=make_lora_config(), quantization=quant_cfg)
        result = _require_quantization_config(cfg)
        assert result["load_in_4bit"] is True

    def test_returns_quant_config_when_valid_8bit(self) -> None:
        """Test that valid 8-bit config is returned."""
        quant_cfg: QuantizationConfig = {
            "load_in_4bit": False,
            "load_in_8bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": False,
        }
        cfg = make_test_config(lora=make_lora_config(), quantization=quant_cfg)
        result = _require_quantization_config(cfg)
        assert result["load_in_8bit"] is True

    def test_raises_when_quantization_is_none(self) -> None:
        """Test that ValueError is raised when quantization is None."""
        cfg = make_test_config(lora=make_lora_config(), quantization=None)
        with pytest.raises(ValueError, match="QLoRA strategy requires quantization config"):
            _require_quantization_config(cfg)

    def test_raises_when_neither_4bit_nor_8bit(self) -> None:
        """Test that ValueError is raised when neither 4-bit nor 8-bit enabled."""
        quant_cfg: QuantizationConfig = {
            "load_in_4bit": False,
            "load_in_8bit": False,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": False,
        }
        cfg = make_test_config(lora=make_lora_config(), quantization=quant_cfg)
        with pytest.raises(ValueError, match="load_in_4bit=True or load_in_8bit=True"):
            _require_quantization_config(cfg)


class TestQLoRAStrategyBasics:
    """Tests for basic QLoRAStrategy functionality."""

    def test_name_returns_qlora(self) -> None:
        """Test that name() returns 'qlora'."""
        strategy = QLoRAStrategy()
        assert strategy.name() == "qlora"

    def test_capabilities_correct(self) -> None:
        """Test that capabilities are correctly configured."""
        strategy = QLoRAStrategy()
        caps = strategy.capabilities()
        assert caps["supports_quantization"] is True
        assert caps["supports_gradient_checkpointing"] is True
        assert caps["requires_peft"] is True
        assert caps["trainable_param_fraction"] == 0.01


class TestQLoRAStrategyAdapt:
    """Tests for QLoRAStrategy.adapt()."""

    def test_adapt_raises_when_lora_config_missing(self) -> None:
        """Test that adapt() raises ValueError when lora config is missing."""
        strategy = QLoRAStrategy()
        model = FakeModel("base")
        cfg = make_test_config(lora=None, quantization=make_quantization_config())

        with pytest.raises(ValueError, match="QLoRA strategy requires lora config"):
            strategy.adapt(model, "test/model", cfg)

    def test_adapt_raises_when_quantization_config_missing(self) -> None:
        """Test that adapt() raises ValueError when quantization config is missing."""
        strategy = QLoRAStrategy()
        model = FakeModel("base")
        cfg = make_test_config(lora=make_lora_config(), quantization=None)

        with pytest.raises(ValueError, match="QLoRA strategy requires quantization config"):
            strategy.adapt(model, "test/model", cfg)

    def test_adapt_returns_adapted_model(self) -> None:
        """Test that adapt() returns correctly configured AdaptedModel."""
        returned_model = FakeModel("qlora-peft-model")
        captured_params: list[int] = []

        def fake_create_peft(
            model: LMModelProto,
            *,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            target_modules: tuple[str, ...],
            bias: str,
        ) -> LMModelProto:
            captured_params.append(r)
            return returned_model

        Hooks.create_peft_model = fake_create_peft

        strategy = QLoRAStrategy()
        model = FakeModel("quantized-base")
        lora_cfg = make_lora_config()
        cfg = make_test_config(lora=lora_cfg, quantization=make_quantization_config())

        adapted = strategy.adapt(model, "test/model-id", cfg)

        assert len(captured_params) == 1
        assert captured_params[0] == 16  # r value from make_lora_config
        assert adapted.model is returned_model
        assert adapted.base_model_id == "test/model-id"
        assert adapted.strategy_name == "qlora"
        assert adapted.is_peft_model is True
        assert adapted.lora_config is lora_cfg

    def test_adapt_calls_gradient_checkpointing_when_hook_set(self) -> None:
        """Test that adapt() enables gradient checkpointing when hook is set."""
        checkpointed: list[LMModelProto] = []

        def fake_enable_checkpointing(model: LMModelProto) -> None:
            checkpointed.append(model)

        def fake_create_peft(
            model: LMModelProto,
            *,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            target_modules: tuple[str, ...],
            bias: str,
        ) -> LMModelProto:
            return FakeModel("peft")

        Hooks.enable_gradient_checkpointing = fake_enable_checkpointing
        Hooks.create_peft_model = fake_create_peft

        strategy = QLoRAStrategy()
        model = FakeModel("base")
        cfg = make_test_config(lora=make_lora_config(), quantization=make_quantization_config())

        strategy.adapt(model, "test/model", cfg)

        assert len(checkpointed) == 1
        assert checkpointed[0] is model


class TestQLoRAStrategySave:
    """Tests for QLoRAStrategy.save_adapted()."""

    def test_save_adapted_calls_hook_and_creates_directory(self) -> None:
        """Test that save_adapted() calls hook and creates directory."""
        saved: list[str] = []

        def fake_create_peft(
            model: LMModelProto,
            *,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            target_modules: tuple[str, ...],
            bias: str,
        ) -> LMModelProto:
            return FakeModel("peft")

        def fake_save_peft(model: LMModelProto, out_dir: str) -> None:
            saved.append(out_dir)

        Hooks.create_peft_model = fake_create_peft
        Hooks.save_peft_model = fake_save_peft

        strategy = QLoRAStrategy()
        cfg = make_test_config(lora=make_lora_config(), quantization=make_quantization_config())
        adapted = strategy.adapt(FakeModel(), "test/model", cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = str(Path(tmpdir) / "nested" / "adapters")
            strategy.save_adapted(adapted, out_dir)

            assert len(saved) == 1
            assert saved[0] == out_dir
            assert Path(out_dir).exists()


class TestQLoRAStrategyLoad:
    """Tests for QLoRAStrategy.load_adapted()."""

    def test_load_adapted_raises_when_path_not_found(self) -> None:
        """Test that load_adapted() raises FileNotFoundError for missing path."""
        strategy = QLoRAStrategy()
        base_model = FakeModel("base")

        with pytest.raises(FileNotFoundError, match="Adapter path not found"):
            strategy.load_adapted(base_model, "test/model", "/nonexistent/path")

    def test_load_adapted_returns_adapted_model(self) -> None:
        """Test that load_adapted() returns correctly configured AdaptedModel."""
        returned_model = FakeModel("loaded-adapter")
        captured_paths: list[str] = []

        def fake_load_peft(model: LMModelProto, adapter_path: str) -> LMModelProto:
            captured_paths.append(adapter_path)
            return returned_model

        Hooks.load_peft_model = fake_load_peft

        strategy = QLoRAStrategy()
        base_model = FakeModel("quantized-base")

        with tempfile.TemporaryDirectory() as tmpdir:
            adapted = strategy.load_adapted(base_model, "test/model-id", tmpdir)

            assert len(captured_paths) == 1
            assert captured_paths[0] == tmpdir
            assert adapted.model is returned_model
            assert adapted.base_model_id == "test/model-id"
            assert adapted.strategy_name == "qlora"
            assert adapted.is_peft_model is True
            assert adapted.lora_config is None


class TestCreateQloraStrategy:
    """Tests for the create_qlora_strategy factory function."""

    def test_create_qlora_strategy_returns_instance(self) -> None:
        """Test that factory creates a QLoRAStrategy instance."""
        strategy = create_qlora_strategy()
        expected = QLoRAStrategy()
        assert type(strategy) is type(expected)
        assert strategy.name() == "qlora"

    def test_create_qlora_strategy_returns_new_instances(self) -> None:
        """Test that factory creates new instances each time."""
        s1 = create_qlora_strategy()
        s2 = create_qlora_strategy()
        assert s1 is not s2
