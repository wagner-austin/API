"""Tests for finetuning contracts (AdaptedModel and related types)."""

from __future__ import annotations

from model_trainer.core.contracts.finetuning import (
    AdaptedModel,
    StrategyCapabilities,
    StrategyName,
)
from model_trainer.core.contracts.model import LoraConfig
from tests.core.services.finetuning.testing import FakeModel


class TestAdaptedModel:
    """Tests for the AdaptedModel wrapper class."""

    def test_create_full_adapted_model(self) -> None:
        """Test creating AdaptedModel for full fine-tuning."""
        model = FakeModel("base")
        adapted = AdaptedModel(
            model=model,
            base_model_id="test/model-123",
            strategy_name=StrategyName.FULL,
            is_peft_model=False,
            lora_config=None,
        )
        assert adapted.model is model
        assert adapted.base_model_id == "test/model-123"
        assert adapted.strategy_name is StrategyName.FULL
        assert adapted.is_peft_model is False
        assert adapted.lora_config is None

    def test_create_lora_adapted_model(self) -> None:
        """Test creating AdaptedModel for LoRA fine-tuning."""
        model = FakeModel("peft")
        lora_cfg: LoraConfig = {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ("q_proj", "v_proj"),
            "bias": "none",
        }
        adapted = AdaptedModel(
            model=model,
            base_model_id="meta/llama-7b",
            strategy_name=StrategyName.LORA,
            is_peft_model=True,
            lora_config=lora_cfg,
        )
        assert adapted.model is model
        assert adapted.base_model_id == "meta/llama-7b"
        assert adapted.strategy_name is StrategyName.LORA
        assert adapted.is_peft_model is True
        lora_config = adapted.lora_config
        assert lora_config is lora_cfg
        assert lora_config["r"] == 16

    def test_create_qlora_adapted_model(self) -> None:
        """Test creating AdaptedModel for QLoRA fine-tuning."""
        model = FakeModel("quantized-peft")
        lora_cfg: LoraConfig = {
            "enabled": True,
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ("q_proj", "k_proj", "v_proj"),
            "bias": "lora_only",
        }
        adapted = AdaptedModel(
            model=model,
            base_model_id="meta/llama-7b",
            strategy_name=StrategyName.QLORA,
            is_peft_model=True,
            lora_config=lora_cfg,
        )
        assert adapted.strategy_name is StrategyName.QLORA
        assert adapted.is_peft_model is True
        lora_config = adapted.lora_config
        assert lora_config is lora_cfg
        assert lora_config["bias"] == "lora_only"

    def test_adapted_model_attributes_are_mutable(self) -> None:
        """Test that AdaptedModel attributes can be updated."""
        model = FakeModel("original")
        adapted = AdaptedModel(
            model=model,
            base_model_id="test/model",
            strategy_name=StrategyName.FULL,
            is_peft_model=False,
            lora_config=None,
        )
        new_model = FakeModel("updated")
        adapted.model = new_model
        assert adapted.model is new_model
        assert adapted.model.name == "updated"

    def test_adapted_model_with_none_lora_config(self) -> None:
        """Test that lora_config=None is valid for non-PEFT models."""
        adapted = AdaptedModel(
            model=FakeModel(),
            base_model_id="test/model",
            strategy_name=StrategyName.FULL,
            is_peft_model=False,
            lora_config=None,
        )
        assert adapted.lora_config is None

    def test_adapted_model_with_peft_but_none_lora_config(self) -> None:
        """Test PEFT model with None lora_config (after loading without config)."""
        # This is valid - when loading adapters, we may not have the original config
        adapted = AdaptedModel(
            model=FakeModel(),
            base_model_id="test/model",
            strategy_name=StrategyName.LORA,
            is_peft_model=True,
            lora_config=None,
        )
        assert adapted.is_peft_model is True
        assert adapted.lora_config is None


class TestStrategyCapabilities:
    """Tests for StrategyCapabilities TypedDict."""

    def test_full_strategy_capabilities(self) -> None:
        """Test capabilities for full fine-tuning."""
        caps: StrategyCapabilities = {
            "supports_quantization": False,
            "supports_gradient_checkpointing": True,
            "requires_peft": False,
            "trainable_param_fraction": 1.0,
        }
        assert caps["supports_quantization"] is False
        assert caps["supports_gradient_checkpointing"] is True
        assert caps["requires_peft"] is False
        assert caps["trainable_param_fraction"] == 1.0

    def test_lora_strategy_capabilities(self) -> None:
        """Test capabilities for LoRA strategy."""
        caps: StrategyCapabilities = {
            "supports_quantization": False,
            "supports_gradient_checkpointing": True,
            "requires_peft": True,
            "trainable_param_fraction": 0.01,
        }
        assert caps["requires_peft"] is True
        assert caps["trainable_param_fraction"] == 0.01

    def test_qlora_strategy_capabilities(self) -> None:
        """Test capabilities for QLoRA strategy."""
        caps: StrategyCapabilities = {
            "supports_quantization": True,
            "supports_gradient_checkpointing": True,
            "requires_peft": True,
            "trainable_param_fraction": 0.01,
        }
        assert caps["supports_quantization"] is True
        assert caps["requires_peft"] is True
