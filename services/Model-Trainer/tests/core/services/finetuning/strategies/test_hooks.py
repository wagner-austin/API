"""Tests for finetuning strategy hooks."""

from __future__ import annotations

from model_trainer.core.services.finetuning.strategies._test_hooks import (
    Hooks,
    _default_create_peft_model,
    _default_enable_gradient_checkpointing,
    _default_load_full_model,
    _default_load_peft_model,
    _default_save_peft_model,
    reset_hooks,
)
from model_trainer.core.types import LMModelProto
from tests.core.services.finetuning.testing import FakeModel


class TestHooksClass:
    """Tests for the Hooks container class."""

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


class TestHooksAreBoundAtImport:
    """The container is usable with no wiring step."""

    def test_every_hook_holds_its_production_implementation(self) -> None:
        """No caller has to initialize the container before using it."""
        reset_hooks()
        assert Hooks.create_peft_model is _default_create_peft_model
        assert Hooks.save_peft_model is _default_save_peft_model
        assert Hooks.load_peft_model is _default_load_peft_model
        assert Hooks.enable_gradient_checkpointing is _default_enable_gradient_checkpointing
        assert Hooks.load_full_model is _default_load_full_model

    def test_reset_restores_a_replaced_hook(self) -> None:
        """A fake is replaced by the production implementation, not by None."""

        def fake_load_full_model(model_path: str) -> LMModelProto:
            return FakeModel()

        Hooks.load_full_model = fake_load_full_model
        assert Hooks.load_full_model is fake_load_full_model
        reset_hooks()
        assert Hooks.load_full_model is _default_load_full_model
