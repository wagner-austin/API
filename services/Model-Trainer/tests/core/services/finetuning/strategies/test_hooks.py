"""Tests for finetuning strategy hooks."""

from __future__ import annotations

from model_trainer.core.services.finetuning.strategies._test_hooks import (
    Hooks,
    reset_hooks,
)
from model_trainer.core.types import LMModelProto
from tests.core.services.finetuning.testing import FakeModel


class TestHooksClass:
    """Tests for the Hooks container class."""

    def test_all_hooks_initially_none(self) -> None:
        """Test that all hooks are None by default after reset."""
        reset_hooks()
        assert Hooks.create_peft_model is None
        assert Hooks.save_peft_model is None
        assert Hooks.load_peft_model is None
        assert Hooks.load_quantized_model is None
        assert Hooks.load_unsloth_model is None
        assert Hooks.apply_unsloth_peft is None
        assert Hooks.enable_gradient_checkpointing is None
        assert Hooks.load_full_model is None

    def test_set_create_peft_model_hook(self) -> None:
        """Test setting the create_peft_model hook."""
        reset_hooks()
        returned_model = FakeModel("peft-r16")
        captured_r: list[int] = []

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
            return returned_model

        Hooks.create_peft_model = fake_create_peft
        hook = Hooks.create_peft_model
        assert hook is fake_create_peft

        result = hook(
            FakeModel(),
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=("q_proj",),
            bias="none",
        )
        assert len(captured_r) == 1
        assert captured_r[0] == 16
        assert result is returned_model
        reset_hooks()

    def test_set_save_peft_model_hook(self) -> None:
        """Test setting the save_peft_model hook."""
        reset_hooks()
        saved_paths: list[str] = []

        def fake_save_peft(model: LMModelProto, out_dir: str) -> None:
            saved_paths.append(out_dir)

        Hooks.save_peft_model = fake_save_peft
        hook = Hooks.save_peft_model
        assert hook is fake_save_peft

        hook(FakeModel(), "/test/path")
        assert saved_paths == ["/test/path"]
        reset_hooks()

    def test_set_load_peft_model_hook(self) -> None:
        """Test setting the load_peft_model hook."""
        reset_hooks()
        returned_model = FakeModel("loaded-adapter")
        captured_paths: list[str] = []

        def fake_load_peft(model: LMModelProto, adapter_path: str) -> LMModelProto:
            captured_paths.append(adapter_path)
            return returned_model

        Hooks.load_peft_model = fake_load_peft
        hook = Hooks.load_peft_model
        assert hook is fake_load_peft

        result = hook(FakeModel(), "/adapters")
        assert len(captured_paths) == 1
        assert captured_paths[0] == "/adapters"
        assert result is returned_model
        reset_hooks()

    def test_set_load_quantized_model_hook(self) -> None:
        """Test setting the load_quantized_model hook."""
        reset_hooks()
        returned_model = FakeModel("quantized-model")
        captured_model_ids: list[str] = []
        captured_4bit: list[bool] = []

        def fake_load_quantized(
            model_id: str,
            *,
            load_in_4bit: bool,
            load_in_8bit: bool,
            bnb_4bit_compute_dtype: str,
            bnb_4bit_quant_type: str,
        ) -> LMModelProto:
            captured_model_ids.append(model_id)
            captured_4bit.append(load_in_4bit)
            return returned_model

        Hooks.load_quantized_model = fake_load_quantized
        hook = Hooks.load_quantized_model
        assert hook is fake_load_quantized

        result = hook(
            "test-model",
            load_in_4bit=True,
            load_in_8bit=False,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
        )
        assert len(captured_model_ids) == 1
        assert captured_model_ids[0] == "test-model"
        assert captured_4bit[0] is True
        assert result is returned_model
        reset_hooks()

    def test_set_load_unsloth_model_hook(self) -> None:
        """Test setting the load_unsloth_model hook."""
        reset_hooks()
        returned_model = FakeModel("unsloth-model")
        captured_model_ids: list[str] = []
        captured_seq_lengths: list[int] = []

        def fake_load_unsloth(
            model_id: str,
            *,
            max_seq_length: int,
            dtype: str | None,
            load_in_4bit: bool,
        ) -> LMModelProto:
            captured_model_ids.append(model_id)
            captured_seq_lengths.append(max_seq_length)
            return returned_model

        Hooks.load_unsloth_model = fake_load_unsloth
        hook = Hooks.load_unsloth_model
        assert hook is fake_load_unsloth

        result = hook(
            "test-model",
            max_seq_length=2048,
            dtype="float16",
            load_in_4bit=True,
        )
        assert len(captured_model_ids) == 1
        assert captured_model_ids[0] == "test-model"
        assert captured_seq_lengths[0] == 2048
        assert result is returned_model
        reset_hooks()

    def test_set_apply_unsloth_peft_hook(self) -> None:
        """Test setting the apply_unsloth_peft hook."""
        reset_hooks()
        returned_model = FakeModel("unsloth-peft")
        captured_r: list[int] = []

        def fake_apply_unsloth_peft(
            model: LMModelProto,
            *,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            target_modules: tuple[str, ...],
        ) -> LMModelProto:
            captured_r.append(r)
            return returned_model

        Hooks.apply_unsloth_peft = fake_apply_unsloth_peft
        hook = Hooks.apply_unsloth_peft
        assert hook is fake_apply_unsloth_peft

        result = hook(
            FakeModel(),
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=("q_proj",),
        )
        assert len(captured_r) == 1
        assert captured_r[0] == 16
        assert result is returned_model
        reset_hooks()

    def test_set_enable_gradient_checkpointing_hook(self) -> None:
        """Test setting the enable_gradient_checkpointing hook."""
        reset_hooks()
        checkpointed_models: list[LMModelProto] = []

        def fake_enable_checkpointing(model: LMModelProto) -> None:
            checkpointed_models.append(model)

        Hooks.enable_gradient_checkpointing = fake_enable_checkpointing
        hook = Hooks.enable_gradient_checkpointing
        assert hook is fake_enable_checkpointing

        model = FakeModel()
        hook(model)
        assert len(checkpointed_models) == 1
        assert checkpointed_models[0] is model
        reset_hooks()

    def test_set_load_full_model_hook(self) -> None:
        """Test setting the load_full_model hook."""
        reset_hooks()
        returned_model = FakeModel("full-model")
        captured_paths: list[str] = []

        def fake_load_full_model(model_path: str) -> LMModelProto:
            captured_paths.append(model_path)
            return returned_model

        Hooks.load_full_model = fake_load_full_model
        hook = Hooks.load_full_model
        assert hook is fake_load_full_model

        result = hook("/path/to/model")
        assert len(captured_paths) == 1
        assert captured_paths[0] == "/path/to/model"
        assert result is returned_model
        reset_hooks()


class TestResetHooks:
    """Tests for the reset_hooks function."""

    def test_reset_hooks_clears_all(self) -> None:
        """Test that reset_hooks clears all hooks by setting and then resetting."""

        # Set all hooks using properly typed functions
        def fake_create_peft(
            model: LMModelProto,
            *,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            target_modules: tuple[str, ...],
            bias: str,
        ) -> LMModelProto:
            return model

        def fake_save_peft(model: LMModelProto, out_dir: str) -> None:
            pass

        def fake_load_peft(model: LMModelProto, adapter_path: str) -> LMModelProto:
            return model

        def fake_load_quantized(
            model_id: str,
            *,
            load_in_4bit: bool,
            load_in_8bit: bool,
            bnb_4bit_compute_dtype: str,
            bnb_4bit_quant_type: str,
        ) -> LMModelProto:
            return FakeModel()

        def fake_load_unsloth(
            model_id: str,
            *,
            max_seq_length: int,
            dtype: str | None,
            load_in_4bit: bool,
        ) -> LMModelProto:
            return FakeModel()

        def fake_apply_unsloth_peft(
            model: LMModelProto,
            *,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            target_modules: tuple[str, ...],
        ) -> LMModelProto:
            return model

        def fake_enable_checkpointing(model: LMModelProto) -> None:
            pass

        def fake_load_full_model(model_path: str) -> LMModelProto:
            return FakeModel()

        Hooks.create_peft_model = fake_create_peft
        Hooks.save_peft_model = fake_save_peft
        Hooks.load_peft_model = fake_load_peft
        Hooks.load_quantized_model = fake_load_quantized
        Hooks.load_unsloth_model = fake_load_unsloth
        Hooks.apply_unsloth_peft = fake_apply_unsloth_peft
        Hooks.enable_gradient_checkpointing = fake_enable_checkpointing
        Hooks.load_full_model = fake_load_full_model

        # Verify all are set to the expected functions
        assert Hooks.create_peft_model is fake_create_peft
        assert Hooks.save_peft_model is fake_save_peft
        assert Hooks.load_peft_model is fake_load_peft
        assert Hooks.load_quantized_model is fake_load_quantized
        assert Hooks.load_unsloth_model is fake_load_unsloth
        assert Hooks.apply_unsloth_peft is fake_apply_unsloth_peft
        assert Hooks.enable_gradient_checkpointing is fake_enable_checkpointing
        assert Hooks.load_full_model is fake_load_full_model

        # Reset
        reset_hooks()

        # Verify all are None
        assert Hooks.create_peft_model is None
        assert Hooks.save_peft_model is None
        assert Hooks.load_peft_model is None
        assert Hooks.load_quantized_model is None
        assert Hooks.load_unsloth_model is None
        assert Hooks.apply_unsloth_peft is None
        assert Hooks.enable_gradient_checkpointing is None
        assert Hooks.load_full_model is None

    def test_reset_hooks_idempotent(self) -> None:
        """Test that calling reset_hooks twice has same effect."""
        reset_hooks()
        reset_hooks()

        assert Hooks.create_peft_model is None
        assert Hooks.save_peft_model is None
        assert Hooks.load_peft_model is None
        assert Hooks.load_quantized_model is None
        assert Hooks.load_unsloth_model is None
        assert Hooks.apply_unsloth_peft is None
        assert Hooks.enable_gradient_checkpointing is None
        assert Hooks.load_full_model is None
