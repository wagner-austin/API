"""Tests for the UnslothStrategy."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from model_trainer.core.contracts.model import (
    LoraConfig,
    ModelTrainConfig,
    UnslothConfig,
)
from model_trainer.core.services.finetuning.strategies._test_hooks import (
    Hooks,
    reset_hooks,
)
from model_trainer.core.services.finetuning.strategies.unsloth import (
    UnslothStrategy,
    _require_lora_config,
    _require_unsloth_config,
    create_unsloth_strategy,
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


def make_unsloth_config() -> UnslothConfig:
    """Create a valid UnslothConfig for testing."""
    return {
        "enabled": True,
        "max_seq_length": 2048,
        "dtype": "float16",
    }


def make_test_config(
    lora: LoraConfig | None = None,
    unsloth: UnslothConfig | None = None,
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
        "finetuning_strategy": "unsloth",
        "hub_model_id": "unsloth/llama-3-8b-bnb-4bit",
        "lora": lora,
        "quantization": None,
        "unsloth": unsloth,
        "gguf_export": None,
    }


@pytest.fixture(autouse=True)
def _reset_hooks() -> Generator[None, None, None]:
    """Reset hooks before and after each test."""
    reset_hooks()
    yield
    reset_hooks()


class TestRequireLoraConfigUnsloth:
    """Tests for _require_lora_config helper in Unsloth module."""

    def test_returns_lora_config_when_valid(self) -> None:
        """Test that valid config is returned."""
        lora_cfg = make_lora_config()
        cfg = make_test_config(lora=lora_cfg, unsloth=make_unsloth_config())
        result = _require_lora_config(cfg)
        assert result["r"] == 16

    def test_raises_when_lora_is_none(self) -> None:
        """Test that ValueError is raised when lora is None."""
        cfg = make_test_config(lora=None, unsloth=make_unsloth_config())
        with pytest.raises(ValueError, match="Unsloth strategy requires lora config"):
            _require_lora_config(cfg)

    def test_raises_when_lora_disabled(self) -> None:
        """Test that ValueError is raised when lora.enabled is False."""
        lora_cfg = make_lora_config()
        lora_cfg["enabled"] = False
        cfg = make_test_config(lora=lora_cfg, unsloth=make_unsloth_config())
        with pytest.raises(ValueError, match=r"lora\.enabled=True"):
            _require_lora_config(cfg)


class TestRequireUnslothConfig:
    """Tests for _require_unsloth_config helper."""

    def test_returns_unsloth_config_when_valid(self) -> None:
        """Test that valid config is returned."""
        unsloth_cfg = make_unsloth_config()
        cfg = make_test_config(lora=make_lora_config(), unsloth=unsloth_cfg)
        result = _require_unsloth_config(cfg)
        assert result["max_seq_length"] == 2048

    def test_returns_config_with_null_dtype(self) -> None:
        """Test that config with null dtype is returned."""
        unsloth_cfg: UnslothConfig = {
            "enabled": True,
            "max_seq_length": 4096,
            "dtype": None,
        }
        cfg = make_test_config(lora=make_lora_config(), unsloth=unsloth_cfg)
        result = _require_unsloth_config(cfg)
        assert result["dtype"] is None

    def test_raises_when_unsloth_is_none(self) -> None:
        """Test that ValueError is raised when unsloth is None."""
        cfg = make_test_config(lora=make_lora_config(), unsloth=None)
        with pytest.raises(ValueError, match="Unsloth strategy requires unsloth config"):
            _require_unsloth_config(cfg)

    def test_raises_when_unsloth_disabled(self) -> None:
        """Test that ValueError is raised when unsloth.enabled is False."""
        unsloth_cfg = make_unsloth_config()
        unsloth_cfg["enabled"] = False
        cfg = make_test_config(lora=make_lora_config(), unsloth=unsloth_cfg)
        with pytest.raises(ValueError, match=r"unsloth\.enabled=True"):
            _require_unsloth_config(cfg)


class TestUnslothStrategyBasics:
    """Tests for basic UnslothStrategy functionality."""

    def test_name_returns_unsloth(self) -> None:
        """Test that name() returns 'unsloth'."""
        strategy = UnslothStrategy()
        assert strategy.name() == "unsloth"

    def test_capabilities_correct(self) -> None:
        """Test that capabilities are correctly configured."""
        strategy = UnslothStrategy()
        caps = strategy.capabilities()
        assert caps["supports_quantization"] is True
        assert caps["supports_gradient_checkpointing"] is True
        assert caps["requires_peft"] is False
        assert caps["requires_unsloth"] is True
        assert caps["trainable_param_fraction"] == 0.01


class TestUnslothStrategyAdapt:
    """Tests for UnslothStrategy.adapt()."""

    def test_adapt_raises_when_lora_config_missing(self) -> None:
        """Test that adapt() raises ValueError when lora config is missing."""
        strategy = UnslothStrategy()
        model = FakeModel("base")
        cfg = make_test_config(lora=None, unsloth=make_unsloth_config())

        with pytest.raises(ValueError, match="Unsloth strategy requires lora config"):
            strategy.adapt(model, "test/model", cfg)

    def test_adapt_raises_when_unsloth_config_missing(self) -> None:
        """Test that adapt() raises ValueError when unsloth config is missing."""
        strategy = UnslothStrategy()
        model = FakeModel("base")
        cfg = make_test_config(lora=make_lora_config(), unsloth=None)

        with pytest.raises(ValueError, match="Unsloth strategy requires unsloth config"):
            strategy.adapt(model, "test/model", cfg)

    def test_adapt_raises_when_unsloth_hook_not_set(self) -> None:
        """Test that adapt() raises RuntimeError when Unsloth hook not set."""
        strategy = UnslothStrategy()
        model = FakeModel("base")
        cfg = make_test_config(lora=make_lora_config(), unsloth=make_unsloth_config())

        with pytest.raises(RuntimeError, match="Unsloth hook not configured"):
            strategy.adapt(model, "test/model", cfg)

    def test_adapt_returns_adapted_model(self) -> None:
        """Test that adapt() returns correctly configured AdaptedModel."""
        returned_model = FakeModel("unsloth-peft-model")
        captured_params: list[int] = []

        def fake_apply_unsloth_peft(
            model: LMModelProto,
            *,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            target_modules: tuple[str, ...],
        ) -> LMModelProto:
            captured_params.append(r)
            return returned_model

        Hooks.apply_unsloth_peft = fake_apply_unsloth_peft

        strategy = UnslothStrategy()
        model = FakeModel("unsloth-base")
        lora_cfg = make_lora_config()
        cfg = make_test_config(lora=lora_cfg, unsloth=make_unsloth_config())

        adapted = strategy.adapt(model, "test/model-id", cfg)

        assert len(captured_params) == 1
        assert captured_params[0] == 16  # r value from make_lora_config
        assert adapted.model is returned_model
        assert adapted.base_model_id == "test/model-id"
        assert adapted.strategy_name == "unsloth"
        assert adapted.is_peft_model is True
        assert adapted.lora_config is lora_cfg

    def test_adapt_passes_all_lora_params(self) -> None:
        """Test that adapt() passes all LoRA parameters correctly."""
        captured_r: list[int] = []
        captured_alpha: list[int] = []
        captured_dropout: list[float] = []
        captured_modules: list[tuple[str, ...]] = []

        def fake_apply_unsloth_peft(
            model: LMModelProto,
            *,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            target_modules: tuple[str, ...],
        ) -> LMModelProto:
            captured_r.append(r)
            captured_alpha.append(lora_alpha)
            captured_dropout.append(lora_dropout)
            captured_modules.append(target_modules)
            return FakeModel("peft")

        Hooks.apply_unsloth_peft = fake_apply_unsloth_peft

        strategy = UnslothStrategy()
        lora_cfg: LoraConfig = {
            "enabled": True,
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.0,
            "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
            "bias": "all",
        }
        cfg = make_test_config(lora=lora_cfg, unsloth=make_unsloth_config())

        strategy.adapt(FakeModel(), "test/model", cfg)

        assert captured_r[0] == 32
        assert captured_alpha[0] == 64
        assert captured_dropout[0] == 0.0
        assert captured_modules[0] == ("q_proj", "k_proj", "v_proj", "o_proj")


class TestUnslothStrategySave:
    """Tests for UnslothStrategy.save_adapted()."""

    def test_save_adapted_raises_when_hook_not_set(self) -> None:
        """Test that save_adapted() raises RuntimeError when save hook not set."""

        def fake_apply_unsloth_peft(
            model: LMModelProto,
            *,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            target_modules: tuple[str, ...],
        ) -> LMModelProto:
            return FakeModel("peft")

        Hooks.apply_unsloth_peft = fake_apply_unsloth_peft

        strategy = UnslothStrategy()
        cfg = make_test_config(lora=make_lora_config(), unsloth=make_unsloth_config())
        adapted = strategy.adapt(FakeModel(), "test/model", cfg)

        with pytest.raises(RuntimeError, match="PEFT save hook not configured"):
            strategy.save_adapted(adapted, "/tmp/output")

    def test_save_adapted_calls_hook_and_creates_directory(self) -> None:
        """Test that save_adapted() calls hook and creates directory."""
        saved: list[str] = []

        def fake_apply_unsloth_peft(
            model: LMModelProto,
            *,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            target_modules: tuple[str, ...],
        ) -> LMModelProto:
            return FakeModel("peft")

        def fake_save_peft(model: LMModelProto, out_dir: str) -> None:
            saved.append(out_dir)

        Hooks.apply_unsloth_peft = fake_apply_unsloth_peft
        Hooks.save_peft_model = fake_save_peft

        strategy = UnslothStrategy()
        cfg = make_test_config(lora=make_lora_config(), unsloth=make_unsloth_config())
        adapted = strategy.adapt(FakeModel(), "test/model", cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = str(Path(tmpdir) / "nested" / "adapters")
            strategy.save_adapted(adapted, out_dir)

            assert len(saved) == 1
            assert saved[0] == out_dir
            assert Path(out_dir).exists()


class TestUnslothStrategyLoad:
    """Tests for UnslothStrategy.load_adapted()."""

    def test_load_adapted_raises_when_path_not_found(self) -> None:
        """Test that load_adapted() raises FileNotFoundError for missing path."""
        strategy = UnslothStrategy()
        base_model = FakeModel("base")

        with pytest.raises(FileNotFoundError, match="Adapter path not found"):
            strategy.load_adapted(base_model, "test/model", "/nonexistent/path")

    def test_load_adapted_raises_when_hook_not_set(self) -> None:
        """Test that load_adapted() raises RuntimeError when load hook not set."""
        strategy = UnslothStrategy()
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

        strategy = UnslothStrategy()
        base_model = FakeModel("unsloth-base")

        with tempfile.TemporaryDirectory() as tmpdir:
            adapted = strategy.load_adapted(base_model, "test/model-id", tmpdir)

            assert len(captured_paths) == 1
            assert captured_paths[0] == tmpdir
            assert adapted.model is returned_model
            assert adapted.base_model_id == "test/model-id"
            assert adapted.strategy_name == "unsloth"
            assert adapted.is_peft_model is True
            assert adapted.lora_config is None


class TestCreateUnslothStrategy:
    """Tests for the create_unsloth_strategy factory function."""

    def test_create_unsloth_strategy_returns_instance(self) -> None:
        """Test that factory creates an UnslothStrategy instance."""
        strategy = create_unsloth_strategy()
        expected = UnslothStrategy()
        assert type(strategy) is type(expected)
        assert strategy.name() == "unsloth"

    def test_create_unsloth_strategy_returns_new_instances(self) -> None:
        """Test that factory creates new instances each time."""
        s1 = create_unsloth_strategy()
        s2 = create_unsloth_strategy()
        assert s1 is not s2
