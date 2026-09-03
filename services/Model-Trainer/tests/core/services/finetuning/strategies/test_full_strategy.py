"""Tests for the FullFineTuneStrategy."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.services.finetuning.strategies._test_hooks import Hooks, reset_hooks
from model_trainer.core.services.finetuning.strategies.full import (
    FullFineTuneStrategy,
    create_full_strategy,
)
from model_trainer.core.types import LMModelProto
from tests.core.services.finetuning.testing import FakeModel


def make_test_config() -> ModelTrainConfig:
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
        "device": "cpu",
        "precision": "fp32",
        "data_num_workers": 0,
        "data_pin_memory": False,
        "early_stopping_patience": 3,
        "test_split_ratio": 0.1,
        "finetune_lr_cap": 0.0001,
        "loss_mask_prefix_separator": None,
        "finetuning_strategy": "full",
        "hub_model_id": None,
        "lora": None,
        "cartridge": None,
        "quantization": None,
        "gguf_export": None,
    }


@pytest.fixture(autouse=True)
def _reset_hooks() -> Generator[None, None, None]:
    """Reset hooks before and after each test."""
    reset_hooks()
    yield
    reset_hooks()


class TestFullFineTuneStrategyBasics:
    """Tests for basic FullFineTuneStrategy functionality."""

    def test_name_returns_full(self) -> None:
        """Test that name() returns 'full'."""
        strategy = FullFineTuneStrategy()
        assert strategy.name() == "full"

    def test_capabilities_correct(self) -> None:
        """Test that capabilities are correctly configured."""
        strategy = FullFineTuneStrategy()
        caps = strategy.capabilities()
        assert caps["supports_quantization"] is False
        assert caps["supports_gradient_checkpointing"] is True
        assert caps["requires_peft"] is False
        assert caps["trainable_param_fraction"] == 1.0


class TestFullFineTuneStrategyAdapt:
    """Tests for FullFineTuneStrategy.adapt()."""

    def test_adapt_returns_adapted_model(self) -> None:
        """Test that adapt() returns correctly configured AdaptedModel."""
        strategy = FullFineTuneStrategy()
        model = FakeModel("base")
        cfg = make_test_config()

        adapted = strategy.adapt(model, "test/model-id", cfg)

        assert adapted.model is model
        assert adapted.base_model_id == "test/model-id"
        assert adapted.strategy_name == "full"
        assert adapted.is_peft_model is False
        assert adapted.lora_config is None

    def test_adapt_calls_gradient_checkpointing_when_hook_set(self) -> None:
        """Test that adapt() enables gradient checkpointing when hook is set."""
        checkpointed: list[LMModelProto] = []

        def fake_enable_checkpointing(model: LMModelProto) -> None:
            checkpointed.append(model)

        Hooks.enable_gradient_checkpointing = fake_enable_checkpointing

        strategy = FullFineTuneStrategy()
        model = FakeModel("base")
        cfg = make_test_config()

        strategy.adapt(model, "test/model", cfg)

        assert len(checkpointed) == 1
        assert checkpointed[0] is model

    def test_adapt_skips_gradient_checkpointing_when_hook_not_set(self) -> None:
        """Test that adapt() works when gradient checkpointing hook is None."""
        # Hooks are reset by fixture, so enable_gradient_checkpointing is None
        strategy = FullFineTuneStrategy()
        model = FakeModel("base")
        cfg = make_test_config()

        # Should not raise
        adapted = strategy.adapt(model, "test/model", cfg)
        assert adapted.model is model


class TestFullFineTuneStrategySave:
    """Tests for FullFineTuneStrategy.save_adapted()."""

    def test_save_adapted_calls_save_pretrained(self) -> None:
        """Test that save_adapted() calls model.save_pretrained."""
        strategy = FullFineTuneStrategy()
        model = FakeModel("trained")
        cfg = make_test_config()
        adapted = strategy.adapt(model, "test/model", cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = str(Path(tmpdir) / "output")
            strategy.save_adapted(adapted, out_dir)
            # FakeModel records the save path
            assert model._save_path == out_dir


class TestFullFineTuneStrategyLoad:
    """Tests for FullFineTuneStrategy.load_adapted()."""

    def test_load_adapted_raises_when_path_not_found(self) -> None:
        """Test that load_adapted() raises FileNotFoundError for missing path."""
        strategy = FullFineTuneStrategy()
        base_model = FakeModel("base")

        with pytest.raises(FileNotFoundError, match="Model path not found"):
            strategy.load_adapted(base_model, "test/model", "/nonexistent/path")

    def test_load_adapted_returns_adapted_model(self) -> None:
        """Test that load_adapted() returns correctly configured AdaptedModel."""
        returned_model = FakeModel("loaded-model")
        captured_paths: list[str] = []

        def fake_load_full_model(model_path: str) -> LMModelProto:
            captured_paths.append(model_path)
            return returned_model

        Hooks.load_full_model = fake_load_full_model

        strategy = FullFineTuneStrategy()
        base_model = FakeModel("base")

        with tempfile.TemporaryDirectory() as tmpdir:
            adapted = strategy.load_adapted(base_model, "test/model-id", tmpdir)

            assert len(captured_paths) == 1
            assert captured_paths[0] == tmpdir
            assert adapted.model is returned_model
            assert adapted.base_model_id == "test/model-id"
            assert adapted.strategy_name == "full"
            assert adapted.is_peft_model is False
            assert adapted.lora_config is None


class TestCreateFullStrategy:
    """Tests for the create_full_strategy factory function."""

    def test_create_full_strategy_returns_instance(self) -> None:
        """Test that factory creates a FullFineTuneStrategy instance."""
        strategy = create_full_strategy()
        expected = FullFineTuneStrategy()
        assert type(strategy) is type(expected)
        assert strategy.name() == "full"

    def test_create_full_strategy_returns_new_instances(self) -> None:
        """Test that factory creates new instances each time."""
        s1 = create_full_strategy()
        s2 = create_full_strategy()
        assert s1 is not s2
