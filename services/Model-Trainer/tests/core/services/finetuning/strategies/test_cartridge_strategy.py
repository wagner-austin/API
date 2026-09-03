"""The cartridge strategy: registration, config, geometry, and real training.

``TestAgainstARealTransformer`` is the one that matters. Everything above it
can pass against a fake reporting whatever shape it likes; only a real GPT-2
can show that a prefix built to a discovered geometry reaches attention, that
training it reduces the loss, and that not one base weight moves. The wrapper's
own behaviour is covered in ``test_cartridge_model``.
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator, Sequence
from pathlib import Path

import pytest
import torch
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.contracts.cartridge import (
    CARTRIDGE_WEIGHTS_NAME,
    CartridgeGeometry,
    trainable_parameter_count,
)
from model_trainer.core.contracts.model import CartridgeConfig, ModelTrainConfig
from model_trainer.core.services.finetuning import default_registry
from model_trainer.core.services.finetuning.strategies._test_hooks import Hooks, reset_hooks
from model_trainer.core.services.finetuning.strategies.cartridge import (
    CartridgeStrategy,
    create_cartridge_strategy,
    measure_geometry,
    require_cache_capable,
)
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.finetuning.strategies.cartridge_slots import initialise_slots
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.services.training.base_trainer_core import _get_optimizer_for_config
from model_trainer.core.types import KVCacheProto, TracedLMModelProto
from tests.core.services.finetuning.testing import FakeCacheCapableModel, FakeModel

#: The tiny rung's own shape: two layers, two heads, 128 wide.
_TINY_GEOMETRY = CartridgeGeometry(num_layers=2, num_kv_heads=2, head_dim=64, num_slots=4)


@pytest.fixture(autouse=True)
def _reset_hooks() -> Generator[None, None, None]:
    """Reset hooks before and after each test."""
    reset_hooks()
    yield
    reset_hooks()


def make_cartridge_config(
    *, enabled: bool = True, num_slots: int = 4, init_seed: int = 7
) -> CartridgeConfig:
    """Build a cartridge config for testing.

    Args:
        enabled: Whether the strategy is enabled.
        num_slots: Prefix positions.
        init_seed: Seed for the draw.

    Returns:
        The config.
    """
    return CartridgeConfig(enabled=enabled, num_slots=num_slots, init_seed=init_seed)


def make_train_config(cartridge: CartridgeConfig | None) -> ModelTrainConfig:
    """Build a minimal training config carrying a cartridge section.

    Args:
        cartridge: The cartridge config, or None.

    Returns:
        The training config.
    """
    return {
        "model_family": "hf_lm",
        "model_size": "small",
        "max_seq_len": 64,
        "num_epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.1,
        "tokenizer_id": None,
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
        "finetuning_strategy": "cartridge",
        "hub_model_id": "gpt2",
        "lora": None,
        "cartridge": cartridge,
        "quantization": None,
        "gguf_export": None,
    }


def tiny_gpt2() -> TracedLMModelProto:
    """Build a real but very small GPT-2.

    The same builder every forward measurement in this package uses, so what
    is tested here is the model the rest of the suite reasons about rather
    than a second definition of "small model". Constructed rather than
    downloaded, so no network and no cache are involved, while still a genuine
    transformer with real attention and a real key-value cache.

    Returns:
        The model, in eval mode on the CPU.
    """
    model, _ = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return model


def tokens(*, batch_size: int, length: int) -> torch.Tensor:
    """Build a token batch inside the tiny rung's vocabulary.

    Args:
        batch_size: Rows.
        length: Positions per row.

    Returns:
        Token ids.
    """
    return torch.arange(batch_size * length, dtype=torch.long).reshape(batch_size, length) % 512


def adapt(base: TracedLMModelProto, *, num_slots: int = 4) -> CartridgeModel:
    """Adapt a model through the real strategy and return the wrapper.

    Args:
        base: The transformer to prepend to.
        num_slots: Prefix positions.

    Returns:
        The cartridge-wrapped model.

    Raises:
        TypeError: If the strategy returned something else.
    """
    adapted = CartridgeStrategy().adapt(
        base, "gpt2", make_train_config(make_cartridge_config(num_slots=num_slots))
    )
    wrapper = adapted.model
    if not isinstance(wrapper, CartridgeModel):
        raise TypeError("the cartridge strategy must produce a CartridgeModel")
    return wrapper


def train_and_report(model: CartridgeModel, *, steps: int) -> tuple[float, float]:
    """Run real optimizer steps on a fixed batch.

    Args:
        model: The cartridge-wrapped model.
        steps: How many steps to run.

    Returns:
        The first and final losses.
    """
    batch = tokens(batch_size=2, length=6)
    optimiser = _get_optimizer_for_config("adamw")(model.parameters(), lr=0.1)
    first_loss = 0.0
    final_loss = 0.0
    for step in range(steps):
        optimiser.zero_grad()
        out = model.forward(input_ids=batch, labels=batch)
        torch.autograd.backward([out.loss])
        optimiser.step()
        final_loss = out.loss.item()
        if step == 0:
            first_loss = final_loss
    return first_loss, final_loss


class TestRegistration:
    """The fourth strategy, reachable the way the other three are."""

    def test_it_is_registered_under_its_name(self) -> None:
        """A strategy the registry cannot produce is unreachable from a request."""
        assert "cartridge" in default_registry().list_strategies()

    def test_the_registry_produces_it(self) -> None:
        """Through the real factory, not a direct construction."""
        assert default_registry().get("cartridge").name() == "cartridge"

    def test_the_factory_makes_a_new_instance_each_time(self) -> None:
        """Two runs must not share mutable strategy state."""
        assert create_cartridge_strategy() is not create_cartridge_strategy()

    def test_its_capabilities_are_exactly_these(self) -> None:
        """Pinned whole, so a change to any one of them is deliberate."""
        assert default_registry().get_capabilities("cartridge") == {
            "supports_quantization": False,
            "supports_gradient_checkpointing": False,
            "requires_peft": False,
            "trainable_param_fraction": 0.017,
        }

    def test_it_alone_refuses_gradient_checkpointing(self) -> None:
        """Measured, not assumed -- the other three all report True.

        Asserted against them rather than in isolation, because the value that
        would have been copied is the one this contradicts.
        """
        registry = default_registry()
        others = [
            registry.get_capabilities("full")["supports_gradient_checkpointing"],
            registry.get_capabilities("lora")["supports_gradient_checkpointing"],
            registry.get_capabilities("qlora")["supports_gradient_checkpointing"],
        ]
        assert others == [True, True, True]
        assert not registry.get_capabilities("cartridge")["supports_gradient_checkpointing"]


class TestConfigRequirements:
    """A cartridge run has to say how big its cartridge is."""

    def test_a_missing_config_is_refused(self) -> None:
        """There is no defensible default slot count."""
        with pytest.raises(AppError) as excinfo:
            CartridgeStrategy().adapt(tiny_gpt2(), "gpt2", make_train_config(None))
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_CONFIG_MISSING

    def test_a_disabled_config_is_refused(self) -> None:
        """The strategy and its config disagree, and guessing picks a wrong run."""
        config = make_train_config(make_cartridge_config(enabled=False))
        with pytest.raises(AppError) as excinfo:
            CartridgeStrategy().adapt(tiny_gpt2(), "gpt2", config)
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_CONFIG_MISSING


class TestNarrowingToACacheCapableModel:
    """Not every language model can host a prefix."""

    def test_a_model_without_the_cached_call_is_refused(self) -> None:
        """FakeModel satisfies LMModelProto and is not callable with a cache."""
        with pytest.raises(AppError) as excinfo:
            require_cache_capable(FakeModel())
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_MODEL_REPORTS_NO_CACHE

    def test_a_cache_capable_model_passes_through_unchanged(self) -> None:
        """Narrowing must return the same object, not a wrapper."""
        model = FakeCacheCapableModel(num_layers=2, num_kv_heads=4, head_dim=8)
        assert require_cache_capable(model) is model

    def test_a_real_transformer_is_cache_capable(self) -> None:
        """The check must admit the models it exists to admit."""
        model = tiny_gpt2()
        assert id(require_cache_capable(model)) == id(model)


class TestGeometryMeasurement:
    """Measuring a model through the injected probe."""

    def test_it_reports_what_the_model_cached(self) -> None:
        """Driven through the hook, so the reading is what is under test."""
        model = FakeCacheCapableModel(num_layers=3, num_kv_heads=2, head_dim=16)
        assert measure_geometry(model, num_slots=5) == CartridgeGeometry(
            num_layers=3, num_kv_heads=2, head_dim=16, num_slots=5
        )

    def test_a_model_reporting_an_empty_cache_is_refused(self) -> None:
        """A transformer with no attention layers cannot carry a prefix."""
        model = FakeCacheCapableModel(num_layers=0, num_kv_heads=2, head_dim=4)
        with pytest.raises(AppError) as excinfo:
            measure_geometry(model, num_slots=4)
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_MODEL_REPORTS_NO_CACHE

    def test_the_probe_asks_for_a_cache_and_computes_no_loss(self) -> None:
        """Without use_cache there is nothing to measure; labels would compute
        a loss that is immediately discarded."""
        model = FakeCacheCapableModel(num_layers=1, num_kv_heads=1, head_dim=2)
        measure_geometry(model, num_slots=1)
        assert model.calls[0]["use_cache"] is True
        assert model.calls[0]["labels"] is None

    def test_the_probe_sends_a_single_token(self) -> None:
        """Only the cache's shape is wanted, so a longer probe is wasted work."""
        model = FakeCacheCapableModel(num_layers=1, num_kv_heads=1, head_dim=2)
        measure_geometry(model, num_slots=1)
        assert tuple(model.calls[0]["input_ids"].shape) == (1, 1)


class TestSaveAndLoad:
    """A cartridge on disk, and the model it may be attached to."""

    def test_a_saved_cartridge_reloads_identically(self) -> None:
        """Through the real saver and loader, against a real model."""
        strategy = CartridgeStrategy()
        adapted = strategy.adapt(tiny_gpt2(), "gpt2", make_train_config(make_cartridge_config()))
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy.save_adapted(adapted, tmpdir)
            reloaded = strategy.load_adapted(tiny_gpt2(), "gpt2", tmpdir)
            assert all(
                torch.equal(a.detach(), b.detach())
                for a, b in zip(
                    adapted.model.parameters(), reloaded.model.parameters(), strict=True
                )
            )

    def test_a_reloaded_cartridge_keeps_its_slot_count(self) -> None:
        """The caller's choice travels with the artifact."""
        strategy = CartridgeStrategy()
        adapted = strategy.adapt(
            tiny_gpt2(), "gpt2", make_train_config(make_cartridge_config(num_slots=6))
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy.save_adapted(adapted, tmpdir)
            restored = strategy.load_adapted(tiny_gpt2(), "gpt2", tmpdir).model
            if not isinstance(restored, CartridgeModel):
                raise TypeError("loading must produce a CartridgeModel")
            assert restored.geometry["num_slots"] == 6

    def test_attaching_to_a_differently_shaped_model_is_refused(self) -> None:
        """A prefix is one model's own keys and values; on another it is noise.

        Saved from a real two-layer GPT-2 and offered to a model reporting
        three layers, so the refusal runs through the real load path.
        """
        strategy = CartridgeStrategy()
        adapted = strategy.adapt(tiny_gpt2(), "gpt2", make_train_config(make_cartridge_config()))
        deeper = FakeCacheCapableModel(num_layers=3, num_kv_heads=2, head_dim=64)
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy.save_adapted(adapted, tmpdir)
            with pytest.raises(AppError) as excinfo:
                strategy.load_adapted(deeper, "gpt2-deep", tmpdir)
            assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_MISMATCH

    def test_a_directory_with_no_cartridge_is_refused(self) -> None:
        """An empty directory must not read as an empty cartridge."""
        strategy = CartridgeStrategy()
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(FileNotFoundError):
            strategy.load_adapted(tiny_gpt2(), "gpt2", tmpdir)

    def test_a_manifest_without_its_weights_is_refused(self) -> None:
        """Half an artifact is not a cartridge."""
        strategy = CartridgeStrategy()
        adapted = strategy.adapt(tiny_gpt2(), "gpt2", make_train_config(make_cartridge_config()))
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy.save_adapted(adapted, tmpdir)
            (Path(tmpdir) / CARTRIDGE_WEIGHTS_NAME).unlink()
            with pytest.raises(FileNotFoundError):
                strategy.load_adapted(tiny_gpt2(), "gpt2", tmpdir)


class TestAgainstARealTransformer:
    """The claims only a real model can establish.

    Every assertion here is about behaviour a fake could report falsely: that
    a prefix built to a discovered geometry reaches attention at all, that
    training it reduces the loss, and that the base comes out of a run
    byte-identical to how it went in.
    """

    def test_the_geometry_matches_the_models_own_shape(self) -> None:
        """Read off the model rather than from any config table."""
        assert adapt(tiny_gpt2()).geometry == _TINY_GEOMETRY

    def test_adapting_freezes_every_base_weight(self) -> None:
        """A single unfrozen weight would be updated by any optimizer built here."""
        base = tiny_gpt2()
        adapt(base)
        assert all(not parameter.requires_grad for _, parameter in base.named_parameters())

    def test_the_prefix_changes_what_the_model_predicts(self) -> None:
        """Otherwise the cartridge is attached and inert.

        Two cartridges drawn from different seeds must produce different
        losses on the same input; if they did not, nothing training learned
        could reach the model's predictions.
        """
        batch = tokens(batch_size=2, length=6)
        cache_capable = require_cache_capable(tiny_gpt2())
        first = CartridgeModel(base=cache_capable, slots=initialise_slots(_TINY_GEOMETRY, seed=1))
        second = CartridgeModel(base=cache_capable, slots=initialise_slots(_TINY_GEOMETRY, seed=2))
        first_loss = first.forward(input_ids=batch, labels=batch).loss.item()
        second_loss = second.forward(input_ids=batch, labels=batch).loss.item()
        assert first_loss != pytest.approx(second_loss)

    def test_training_reduces_the_loss_and_moves_every_slot(self) -> None:
        """The whole point: a cartridge learns.

        The assertion is not that the loss merely changed but that it fell,
        and that every trainable block was updated -- a block that never moves
        is capacity the run pays attention cost for and gets nothing from.
        """
        model = adapt(tiny_gpt2())
        before = [tensor.detach().clone() for tensor in model.parameters()]
        first_loss, final_loss = train_and_report(model, steps=5)
        assert final_loss < first_loss
        moved = sum(
            0 if torch.equal(was, now.detach()) else 1
            for was, now in zip(before, model.parameters(), strict=True)
        )
        assert moved == len(before)

    def test_training_leaves_every_base_weight_byte_identical(self) -> None:
        """The claim that separates this from every other strategy here.

        Checked weight by weight after real optimizer steps, and paired with
        the loss falling so the test cannot pass by not training at all.
        """
        base = tiny_gpt2()
        model = adapt(base)
        snapshot = {name: parameter.detach().clone() for name, parameter in base.named_parameters()}
        first_loss, final_loss = train_and_report(model, steps=3)
        assert final_loss < first_loss
        assert all(
            torch.equal(snapshot[name], parameter.detach())
            for name, parameter in base.named_parameters()
        )

    def test_a_run_that_switches_to_training_mode_still_learns(self) -> None:
        """The sequence the training loop actually performs.

        ``train()`` reaches the frozen base, because its dropout and its
        layer-norm behaviour still have to be in training mode for the steps
        that follow to be the steps a run takes. Covered together with the
        learning it precedes, rather than as a mode flag nobody reads.
        """
        model = adapt(tiny_gpt2())
        model.train()
        first_loss, final_loss = train_and_report(model, steps=3)
        assert final_loss < first_loss

    def test_the_optimizer_cannot_reach_the_base_at_all(self) -> None:
        """Structural, not conventional: the base is absent from parameters().

        A frozen weight that appeared here would still be held by the
        optimizer, and one stray requires_grad would start training it.
        """
        base = tiny_gpt2()
        model = adapt(base)
        exposed = {id(tensor) for tensor in model.parameters()}
        base_ids = {id(parameter) for _, parameter in base.named_parameters()}
        assert exposed.isdisjoint(base_ids)

    def test_the_trainable_count_matches_what_is_exposed(self) -> None:
        """The contract's arithmetic against the tensors that actually exist."""
        model = adapt(tiny_gpt2(), num_slots=4)
        counted = sum(int(tensor.detach().numel()) for tensor in model.parameters())
        assert counted == trainable_parameter_count(model.geometry)

    def test_a_larger_cartridge_holds_proportionally_more(self) -> None:
        """The slot count is the capacity knob and this is what it buys."""
        small = adapt(tiny_gpt2(), num_slots=4)
        large = adapt(tiny_gpt2(), num_slots=8)
        assert trainable_parameter_count(large.geometry) == 2 * trainable_parameter_count(
            small.geometry
        )

    def test_batches_of_different_widths_both_produce_a_finite_loss(self) -> None:
        """The expansion is per-call, so a changing batch size must not break it."""
        model = adapt(tiny_gpt2())
        losses = [
            model.forward(
                input_ids=tokens(batch_size=size, length=5),
                labels=tokens(batch_size=size, length=5),
            ).loss.item()
            for size in (1, 4)
        ]
        assert all(value > 0.0 for value in losses)

    def test_every_row_of_a_batch_reaches_the_one_cartridge(self) -> None:
        """The blocks are expanded views, so a batch accumulates into one object.

        A four-row batch must produce a different gradient than its first row
        alone; had expansion copied, only one row's signal would arrive.
        """
        batch = tokens(batch_size=4, length=5)

        batched_model = adapt(tiny_gpt2())
        optimiser = _get_optimizer_for_config("adamw")(batched_model.parameters(), lr=0.1)
        param_before = batched_model.parameters()[0].detach().clone()
        before = batched_model.forward(input_ids=batch, labels=batch)
        first_loss = before.loss.item()
        torch.autograd.backward([before.loss])
        batched_grad = batched_model.parameters()[0].grad
        optimiser.step()
        param_after = batched_model.parameters()[0].detach()
        final_loss = batched_model.forward(input_ids=batch, labels=batch).loss.item()

        single_model = adapt(tiny_gpt2())
        torch.autograd.backward([single_model.forward(input_ids=batch[:1], labels=batch[:1]).loss])
        single_grad = single_model.parameters()[0].grad

        if batched_grad is None or single_grad is None:
            raise TypeError("the cartridge must receive a gradient")
        assert not torch.equal(batched_grad, single_grad)
        assert float((param_after - param_before).abs().sum().item()) > 0.0
        assert final_loss < first_loss


class TestTheHookSurface:
    """The injection points, asserted as a set so a new one cannot appear silently.

    THREE, not six. The cartridge work started with six and three of them were
    injection points onto things that were already injectable: two called the
    MODEL, which every caller supplies and every fake here replaces directly,
    and one forwarded to a public pure function. A hook that nothing can
    usefully replace is indirection with a protocol attached, so they were
    removed rather than left as surface nobody uses.

    What remains are the three real boundaries: the dynamic ``transformers``
    import that builds a cache, and the two that touch the filesystem.
    """

    def test_the_cartridge_hooks_are_exactly_the_three_boundaries(self) -> None:
        """Pinned as an equality over the cartridge names, not a subset.

        A subset check passes when a hook is added, which is the drift this
        exists to catch: the next unnecessary injection point should fail here
        and have to justify itself.
        """
        declared = {name for name in dir(Hooks) if not name.startswith("_")}
        assert {name for name in declared if "cartridge" in name or "prefix" in name} == {
            "build_prefix_cache",
            "save_cartridge",
            "load_cartridge",
        }

    def test_reset_restores_a_swapped_hook(self) -> None:
        """A test that swapped one must not leak it into the next."""

        def _fake_builder(blocks: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> KVCacheProto:
            _ = blocks
            raise NotImplementedError("never called")

        original = Hooks.build_prefix_cache
        Hooks.build_prefix_cache = _fake_builder
        reset_hooks()
        assert Hooks.build_prefix_cache is original
