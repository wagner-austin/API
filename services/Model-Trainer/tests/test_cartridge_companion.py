"""Companioned cartridge training: the freeze, the draw, and the seed contract.

The intervention this class exists for -- training a cartridge with a frozen
stranger present so composition stops being an untrained capability -- is a
GPU measurement (board task ``bc29dc3e``). What a test can and must pin is
the machinery's contract: the companion is untouchable by the optimizer, the
presence draw consumes the global generator uniformly, training remains a
function of its seed with the companion machinery included, and the refusals
refuse.
"""

from __future__ import annotations

import pytest
import torch
from platform_core.errors import AppError

from model_trainer.core.contracts.cartridge import CartridgeGeometry
from model_trainer.core.services.finetuning.strategies.cartridge import (
    measure_geometry,
    require_cache_capable,
)
from model_trainer.core.services.finetuning.strategies.cartridge_model import (
    CartridgeModel,
    CompanionedCartridgeModel,
)
from model_trainer.core.services.finetuning.strategies.cartridge_slots import (
    CartridgeSlots,
    initialise_slots,
)
from model_trainer.core.services.model.cartridge_companioned import (
    train_cartridge_with_companion,
)
from model_trainer.core.services.model.cartridge_measurement import train_cartridge
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import CacheCapableLMProto

_VOCAB = PROBE_SHAPES["tiny"]["vocab_size"]


def _base() -> CacheCapableLMProto:
    """Build the deterministic tiny GPT-2 the probe suite uses.

    Returns:
        The base model, cache-capable.
    """
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return require_cache_capable(model)


def _corpus(seed: int, rows: int) -> list[torch.Tensor]:
    """Draw a small deterministic corpus of 8-token windows.

    Args:
        seed: Seed for the draw.
        rows: How many windows.

    Returns:
        One (1, 8) id tensor per window.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return [
        torch.randint(0, _VOCAB, (1, 8), generator=generator, dtype=torch.long) for _ in range(rows)
    ]


def _slot_bytes(slots_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Clone a state dict for later exact comparison.

    Args:
        slots_state: The tensors to snapshot.

    Returns:
        Detached clones, keyed identically.
    """
    return {name: tensor.detach().clone() for name, tensor in slots_state.items()}


class TestConstruction:
    def test_a_zero_probability_is_refused_as_a_dead_knob(self) -> None:
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        with pytest.raises(ValueError, match="dead knob"):
            CompanionedCartridgeModel(
                base=base,
                slots=initialise_slots(geometry, seed=1),
                companion=initialise_slots(geometry, seed=2),
                companion_probability=0.0,
            )

    def test_a_probability_above_one_is_refused(self) -> None:
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        with pytest.raises(ValueError, match="outside"):
            CompanionedCartridgeModel(
                base=base,
                slots=initialise_slots(geometry, seed=1),
                companion=initialise_slots(geometry, seed=2),
                companion_probability=1.5,
            )

    def test_a_companion_for_another_model_is_refused(self) -> None:
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        foreign = CartridgeGeometry(
            num_layers=geometry["num_layers"] + 1,
            num_kv_heads=geometry["num_kv_heads"],
            head_dim=geometry["head_dim"],
            num_slots=2,
        )
        with pytest.raises(AppError, match="differently shaped model"):
            CompanionedCartridgeModel(
                base=base,
                slots=initialise_slots(geometry, seed=1),
                companion=initialise_slots(foreign, seed=2),
                companion_probability=1.0,
            )

    def test_a_companion_of_a_different_slot_count_is_accepted(self) -> None:
        """The slot count is the caller's choice, exactly as compose() treats it."""
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        wider = measure_geometry(base, num_slots=4)
        model = CompanionedCartridgeModel(
            base=base,
            slots=initialise_slots(geometry, seed=1),
            companion=initialise_slots(wider, seed=2),
            companion_probability=1.0,
        )
        assert model.geometry["num_slots"] == 2


class TestForward:
    def test_the_presence_draw_consumes_the_generator_even_at_probability_one(self) -> None:
        """The p-sweep's arms must share one RNG-consumption pattern.

        In eval mode dropout draws nothing, so the only global-generator
        consumer in a companioned forward is the presence draw. If the draw
        were skipped at probability 1.0, the two models below would leave the
        generator in the same state and the follow-up draws would agree.
        """
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        item = _corpus(seed=11, rows=1)[0]

        torch.manual_seed(99)
        plain = CartridgeModel(base=base, slots=initialise_slots(geometry, seed=1))
        plain.eval()
        plain_out = plain.forward(input_ids=item, labels=item)
        after_plain = float(torch.rand(()))

        torch.manual_seed(99)
        companioned = CompanionedCartridgeModel(
            base=base,
            slots=initialise_slots(geometry, seed=1),
            companion=initialise_slots(geometry, seed=2),
            companion_probability=1.0,
        )
        companioned.eval()
        companioned_out = companioned.forward(input_ids=item, labels=item)
        after_companioned = float(torch.rand(()))

        # Same trainee slots, same input: the companion's presence is the only
        # difference, and it adds attention targets, so the loss must move.
        assert float(companioned_out.loss.item()) != float(plain_out.loss.item())
        assert after_plain != after_companioned

    def test_an_absent_companion_forwards_exactly_as_the_plain_model(self) -> None:
        """Below-threshold draws fall through to the parent forward.

        A tiny probability plus a pinned generator state that draws above it
        exercises the fall-through branch; the logits must equal the plain
        model's on the same input, because an absent companion must leave no
        trace.
        """
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        item = _corpus(seed=13, rows=1)[0]

        plain = CartridgeModel(base=base, slots=initialise_slots(geometry, seed=1))
        plain.eval()
        expected = plain.forward(input_ids=item, labels=item)

        companioned = CompanionedCartridgeModel(
            base=base,
            slots=initialise_slots(geometry, seed=1),
            companion=initialise_slots(geometry, seed=2),
            companion_probability=1e-9,
        )
        companioned.eval()
        torch.manual_seed(7)
        actual = companioned.forward(input_ids=item, labels=item)

        assert float(actual.loss.item()) == pytest.approx(float(expected.loss.item()))

    def test_construction_moves_the_companion_onto_the_base_device(self) -> None:
        """The constructor owns device coherence, not the caller.

        A noise provider draws on the CPU and nothing else moves the draw;
        on a CUDA base the first forward would torch.cat across devices.
        This suite runs on CPU where the assertion is trivially satisfiable,
        so what it pins is that the move HAPPENS in the constructor: the
        companion's tensors end on the base's device whatever device the
        provider drew them on.
        """
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        companion = initialise_slots(geometry, seed=2)

        model = CompanionedCartridgeModel(
            base=base,
            slots=initialise_slots(geometry, seed=1),
            companion=companion,
            companion_probability=1.0,
        )

        base_device = next(iter(base.named_parameters()))[1].device
        for name, tensor in companion.state_dict().items():
            assert tensor.device == base_device, name
        assert model.geometry["num_slots"] == 2

    def test_to_moves_the_companion_with_the_model(self) -> None:
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        model = CompanionedCartridgeModel(
            base=base,
            slots=initialise_slots(geometry, seed=1),
            companion=initialise_slots(geometry, seed=2),
            companion_probability=1.0,
        )
        assert model.to("cpu") is model


class TestCompanionedScalingDegenerate:
    def test_no_other_cartridges_composes_the_first_alone(self) -> None:
        """The empty companioned composition is the companioned first, exactly.

        No plan asks for a one-compartment composition; this covers the
        zero-others branch directly so it is measured rather than dead.
        """
        from model_trainer.core.services.model.cartridge_companioned import (
            measure_companioned_scaling,
        )

        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        companion = initialise_slots(geometry, seed=5)

        def fixed_companion(seed: int) -> CartridgeSlots:
            """The same frozen companion for every replicate.

            Args:
                seed: Ignored; the degenerate case needs no per-seed draw.

            Returns:
                The one companion.
            """
            return companion

        alone, composed, untrained_composed, cross = measure_companioned_scaling(
            base,
            first_train=_corpus(seed=41, rows=4),
            other_trains=(),
            held_out=_corpus(seed=42, rows=3),
            arm="solo-n1",
            num_slots=2,
            seeds=(7, 8, 9),
            epochs=1,
            learning_rate=0.05,
            companion_for_seed=fixed_companion,
            companion_probability=1.0,
        )

        assert cross == ()
        assert composed["gains"] == alone["gains"]
        assert untrained_composed["gains"] == alone["gains"]
        assert alone["arm"] == "solo-n1-alone"


class TestTraining:
    def test_the_companion_is_byte_identical_after_training(self) -> None:
        """The freeze is the load-bearing property: only the trainee learns."""
        base = _base()
        companion = train_cartridge(
            base, _corpus(seed=21, rows=4), num_slots=2, seed=5, epochs=1, learning_rate=0.05
        )
        before = _slot_bytes(companion.state_dict())

        trained = train_cartridge_with_companion(
            base,
            _corpus(seed=22, rows=4),
            num_slots=2,
            seed=6,
            epochs=1,
            learning_rate=0.05,
            companion=companion,
            companion_probability=1.0,
        )

        after = companion.state_dict()
        assert sorted(after) == sorted(before)
        for name, tensor in after.items():
            assert torch.equal(tensor, before[name]), name
        assert set(trained.state_dict()) == set(before)

    def test_the_trainee_actually_moves(self) -> None:
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        companion = initialise_slots(geometry, seed=5)
        drawn_before = _slot_bytes(initialise_slots(geometry, seed=6).state_dict())

        trained = train_cartridge_with_companion(
            base,
            _corpus(seed=23, rows=4),
            num_slots=2,
            seed=6,
            epochs=1,
            learning_rate=0.05,
            companion=companion,
            companion_probability=1.0,
        )

        moved = [
            name
            for name, tensor in trained.state_dict().items()
            if not torch.equal(tensor, drawn_before[name])
        ]
        assert sorted(moved) == sorted(drawn_before)

    def test_training_is_a_function_of_its_seed_with_the_companion_included(self) -> None:
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        companion = initialise_slots(geometry, seed=5)

        first = train_cartridge_with_companion(
            base,
            _corpus(seed=24, rows=4),
            num_slots=2,
            seed=6,
            epochs=1,
            learning_rate=0.05,
            companion=companion,
            companion_probability=0.5,
        )
        second = train_cartridge_with_companion(
            base,
            _corpus(seed=24, rows=4),
            num_slots=2,
            seed=6,
            epochs=1,
            learning_rate=0.05,
            companion=companion,
            companion_probability=0.5,
        )

        for name, tensor in first.state_dict().items():
            assert torch.equal(tensor, second.state_dict()[name]), name

    def test_the_companion_changes_what_is_learned(self) -> None:
        """Same seed, same corpus: companioned training must differ from plain.

        If it did not, the machinery would be decoration and the sweep would
        measure nothing.
        """
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        companion = initialise_slots(geometry, seed=5)

        plain = train_cartridge(
            base, _corpus(seed=25, rows=4), num_slots=2, seed=6, epochs=1, learning_rate=0.05
        )
        companioned = train_cartridge_with_companion(
            base,
            _corpus(seed=25, rows=4),
            num_slots=2,
            seed=6,
            epochs=1,
            learning_rate=0.05,
            companion=companion,
            companion_probability=1.0,
        )

        differing = [
            name
            for name, tensor in companioned.state_dict().items()
            if not torch.equal(tensor, plain.state_dict()[name])
        ]
        assert sorted(differing) == sorted(plain.state_dict())
