"""Tests for the LoRAStrategy."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from model_trainer.core.contracts.model import LoraConfig, ModelTrainConfig
from model_trainer.core.services.finetuning.strategies._test_hooks import (
    Hooks,
    reset_hooks,
)
from model_trainer.core.services.finetuning.strategies.lora import (
    LoRAStrategy,
    _require_lora_config,
    create_lora_strategy,
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


def make_test_config(lora: LoraConfig | None = None) -> ModelTrainConfig:
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
        "finetuning_strategy": "lora",
        "hub_model_id": "meta/llama-7b",
        "lora": lora,
        "quantization": None,
        "unsloth": None,
    }


@pytest.fixture(autouse=True)
def _reset_hooks() -> Generator[None, None, None]:
    """Reset hooks before and after each test."""
    reset_hooks()
    yield
    reset_hooks()


class TestRequireLoraConfig:
    """Tests for _require_lora_config helper."""

    def test_returns_lora_config_when_valid(self) -> None:
        """Test that valid config is returned."""
        lora_cfg = make_lora_config()
        cfg = make_test_config(lora=lora_cfg)
        result = _require_lora_config(cfg)
        assert result["r"] == 16
        assert result["enabled"] is True

    def test_raises_when_lora_is_none(self) -> None:
        """Test that ValueError is raised when lora is None."""
        cfg = make_test_config(lora=None)
        with pytest.raises(ValueError, match="LoRA strategy requires lora config"):
            _require_lora_config(cfg)

    def test_raises_when_lora_disabled(self) -> None:
        """Test that ValueError is raised when lora.enabled is False."""
        lora_cfg = make_lora_config()
        lora_cfg["enabled"] = False
        cfg = make_test_config(lora=lora_cfg)
        with pytest.raises(ValueError, match=r"lora\.enabled=True"):
            _require_lora_config(cfg)


class TestLoRAStrategyBasics:
    """Tests for basic LoRAStrategy functionality."""

    def test_name_returns_lora(self) -> None:
        """Test that name() returns 'lora'."""
        strategy = LoRAStrategy()
        assert strategy.name() == "lora"

    def test_capabilities_correct(self) -> None:
        """Test that capabilities are correctly configured."""
        strategy = LoRAStrategy()
        caps = strategy.capabilities()
        assert caps["supports_quantization"] is False
        assert caps["supports_gradient_checkpointing"] is True
        assert caps["requires_peft"] is True
        assert caps["requires_unsloth"] is False
        assert caps["trainable_param_fraction"] == 0.01


class TestLoRAStrategyAdapt:
    """Tests for LoRAStrategy.adapt()."""

    def test_adapt_raises_when_lora_config_missing(self) -> None:
        """Test that adapt() raises ValueError when lora config is missing."""
        strategy = LoRAStrategy()
        model = FakeModel("base")
        cfg = make_test_config(lora=None)

        with pytest.raises(ValueError, match="LoRA strategy requires lora config"):
            strategy.adapt(model, "test/model", cfg)

    def test_adapt_raises_when_peft_hook_not_set(self) -> None:
        """Test that adapt() raises RuntimeError when PEFT hook not set."""
        strategy = LoRAStrategy()
        model = FakeModel("base")
        cfg = make_test_config(lora=make_lora_config())

        with pytest.raises(RuntimeError, match="PEFT hook not configured"):
            strategy.adapt(model, "test/model", cfg)

    def test_adapt_returns_adapted_model(self) -> None:
        """Test that adapt() returns correctly configured AdaptedModel."""
        created_models: list[tuple[LMModelProto, int, int]] = []
        returned_model = FakeModel("peft-model")

        def fake_create_peft(
            model: LMModelProto,
            *,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            target_modules: tuple[str, ...],
            bias: str,
        ) -> LMModelProto:
            created_models.append((model, r, lora_alpha))
            return returned_model

        Hooks.create_peft_model = fake_create_peft

        strategy = LoRAStrategy()
        model = FakeModel("base")
        lora_cfg = make_lora_config()
        cfg = make_test_config(lora=lora_cfg)

        adapted = strategy.adapt(model, "test/model-id", cfg)

        assert len(created_models) == 1
        assert created_models[0][0] is model
        assert created_models[0][1] == 16  # r
        assert created_models[0][2] == 32  # lora_alpha
        assert adapted.model is returned_model
        assert adapted.base_model_id == "test/model-id"
        assert adapted.strategy_name == "lora"
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

        strategy = LoRAStrategy()
        model = FakeModel("base")
        cfg = make_test_config(lora=make_lora_config())

        strategy.adapt(model, "test/model", cfg)

        assert len(checkpointed) == 1
        assert checkpointed[0] is model

    def test_adapt_passes_all_lora_params(self) -> None:
        """Test that adapt() passes all LoRA parameters correctly."""
        captured_r: list[int] = []
        captured_alpha: list[int] = []
        captured_dropout: list[float] = []
        captured_modules: list[tuple[str, ...]] = []
        captured_bias: list[str] = []

        def fake_create_peft(
            model: LMModelProto,
            *,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            target_modules: tuple[str, ...],
            bias: str,
        ) -> LMModelProto:
            captured_r.append(r)
            captured_alpha.append(lora_alpha)
            captured_dropout.append(lora_dropout)
            captured_modules.append(target_modules)
            captured_bias.append(bias)
            return FakeModel("peft")

        Hooks.create_peft_model = fake_create_peft

        strategy = LoRAStrategy()
        lora_cfg: LoraConfig = {
            "enabled": True,
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ("q_proj", "k_proj", "v_proj"),
            "bias": "lora_only",
        }
        cfg = make_test_config(lora=lora_cfg)

        strategy.adapt(FakeModel(), "test/model", cfg)

        assert captured_r[0] == 8
        assert captured_alpha[0] == 16
        assert captured_dropout[0] == 0.05
        assert captured_modules[0] == ("q_proj", "k_proj", "v_proj")
        assert captured_bias[0] == "lora_only"


class TestLoRAStrategySave:
    """Tests for LoRAStrategy.save_adapted()."""

    def test_save_adapted_raises_when_hook_not_set(self) -> None:
        """Test that save_adapted() raises RuntimeError when save hook not set."""

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

        Hooks.create_peft_model = fake_create_peft

        strategy = LoRAStrategy()
        cfg = make_test_config(lora=make_lora_config())
        adapted = strategy.adapt(FakeModel(), "test/model", cfg)

        with pytest.raises(RuntimeError, match="PEFT save hook not configured"):
            strategy.save_adapted(adapted, "/tmp/output")

    def test_save_adapted_calls_hook(self) -> None:
        """Test that save_adapted() calls the save hook."""
        saved: list[tuple[LMModelProto, str]] = []

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
            saved.append((model, out_dir))

        Hooks.create_peft_model = fake_create_peft
        Hooks.save_peft_model = fake_save_peft

        strategy = LoRAStrategy()
        cfg = make_test_config(lora=make_lora_config())
        adapted = strategy.adapt(FakeModel(), "test/model", cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = str(Path(tmpdir) / "adapters")
            strategy.save_adapted(adapted, out_dir)

            assert len(saved) == 1
            assert saved[0][1] == out_dir

    def test_save_adapted_creates_directory(self) -> None:
        """Test that save_adapted() creates output directory."""

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
            pass

        Hooks.create_peft_model = fake_create_peft
        Hooks.save_peft_model = fake_save_peft

        strategy = LoRAStrategy()
        cfg = make_test_config(lora=make_lora_config())
        adapted = strategy.adapt(FakeModel(), "test/model", cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = str(Path(tmpdir) / "nested" / "adapters")
            strategy.save_adapted(adapted, out_dir)
            assert Path(out_dir).exists()


class TestLoRAStrategyLoad:
    """Tests for LoRAStrategy.load_adapted()."""

    def test_load_adapted_raises_when_path_not_found(self) -> None:
        """Test that load_adapted() raises FileNotFoundError for missing path."""
        strategy = LoRAStrategy()
        base_model = FakeModel("base")

        with pytest.raises(FileNotFoundError, match="Adapter path not found"):
            strategy.load_adapted(base_model, "test/model", "/nonexistent/path")

    def test_load_adapted_raises_when_hook_not_set(self) -> None:
        """Test that load_adapted() raises RuntimeError when load hook not set."""
        strategy = LoRAStrategy()
        base_model = FakeModel("base")

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(RuntimeError, match="PEFT load hook not configured"),
        ):
            strategy.load_adapted(base_model, "test/model", tmpdir)

    def test_load_adapted_returns_adapted_model(self) -> None:
        """Test that load_adapted() returns correctly configured AdaptedModel."""
        returned_model = FakeModel("loaded-adapter")
        captured_paths: list[str] = []

        def fake_load_peft(model: LMModelProto, adapter_path: str) -> LMModelProto:
            captured_paths.append(adapter_path)
            return returned_model

        Hooks.load_peft_model = fake_load_peft

        strategy = LoRAStrategy()
        base_model = FakeModel("base")

        with tempfile.TemporaryDirectory() as tmpdir:
            adapted = strategy.load_adapted(base_model, "test/model-id", tmpdir)

            assert len(captured_paths) == 1
            assert captured_paths[0] == tmpdir
            assert adapted.model is returned_model
            assert adapted.base_model_id == "test/model-id"
            assert adapted.strategy_name == "lora"
            assert adapted.is_peft_model is True
            assert adapted.lora_config is None  # Config not preserved


class TestCreateLoraStrategy:
    """Tests for the create_lora_strategy factory function."""

    def test_create_lora_strategy_returns_instance(self) -> None:
        """Test that factory creates a LoRAStrategy instance."""
        strategy = create_lora_strategy()
        expected = LoRAStrategy()
        assert type(strategy) is type(expected)
        assert strategy.name() == "lora"

    def test_create_lora_strategy_returns_new_instances(self) -> None:
        """Test that factory creates new instances each time."""
        s1 = create_lora_strategy()
        s2 = create_lora_strategy()
        assert s1 is not s2
