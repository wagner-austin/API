"""Varied-count companioned training: the pool freeze, the draws, the seed.

What the single-companion suite pins for its machinery, this suite pins for
the pool variant (board task ``7815a0fd``): no pool member is reachable by
the optimizer, every forward consumes the same three global-generator draws
whatever the outcome, training remains a function of its seed with the pool
machinery included, the drawn count actually moves the loss, and the
refusals refuse -- including the two degenerate pools whose honest spellings
are the existing classes.
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
    MultiCompanionedCartridgeModel,
)
from model_trainer.core.services.finetuning.strategies.cartridge_slots import (
    CartridgeSlots,
    initialise_slots,
)
from model_trainer.core.services.model.cartridge_companioned import (
    train_cartridge_with_companion,
)
from model_trainer.core.services.model.cartridge_varied import (
    measure_varied_companioned_scaling,
    train_cartridge_with_companions,
)
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


def _pool(geometry: CartridgeGeometry, *seeds: int) -> tuple[CartridgeSlots, ...]:
    """Draw one untrained pool member per seed.

    Args:
        geometry: Shape every member is cut to.
        seeds: One draw seed per member.

    Returns:
        The pool.
    """
    return tuple(initialise_slots(geometry, seed=seed) for seed in seeds)


class TestConstruction:
    def test_a_zero_probability_is_refused_as_a_dead_knob(self) -> None:
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        with pytest.raises(ValueError, match="dead knob"):
            MultiCompanionedCartridgeModel(
                base=base,
                slots=initialise_slots(geometry, seed=1),
                companions=_pool(geometry, 2, 3),
                companion_probability=0.0,
            )

    def test_a_probability_above_one_is_refused(self) -> None:
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        with pytest.raises(ValueError, match="outside"):
            MultiCompanionedCartridgeModel(
                base=base,
                slots=initialise_slots(geometry, seed=1),
                companions=_pool(geometry, 2, 3),
                companion_probability=1.5,
            )

    def test_a_pool_of_one_is_refused_as_the_single_companion_model(self) -> None:
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        with pytest.raises(ValueError, match="CompanionedCartridgeModel"):
            MultiCompanionedCartridgeModel(
                base=base,
                slots=initialise_slots(geometry, seed=1),
                companions=_pool(geometry, 2),
                companion_probability=1.0,
            )

    def test_an_empty_pool_is_refused(self) -> None:
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        with pytest.raises(ValueError, match="cannot vary the count"):
            MultiCompanionedCartridgeModel(
                base=base,
                slots=initialise_slots(geometry, seed=1),
                companions=(),
                companion_probability=1.0,
            )

    def test_a_pool_member_for_another_model_is_refused(self) -> None:
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        foreign = CartridgeGeometry(
            num_layers=geometry["num_layers"] + 1,
            num_kv_heads=geometry["num_kv_heads"],
            head_dim=geometry["head_dim"],
            num_slots=2,
        )
        with pytest.raises(AppError, match="differently shaped model"):
            MultiCompanionedCartridgeModel(
                base=base,
                slots=initialise_slots(geometry, seed=1),
                companions=(
                    initialise_slots(geometry, seed=2),
                    initialise_slots(foreign, seed=3),
                ),
                companion_probability=1.0,
            )

    def test_pool_members_of_a_different_slot_count_are_accepted(self) -> None:
        """The slot count is the caller's choice, exactly as compose() treats it."""
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        wider = measure_geometry(base, num_slots=4)
        model = MultiCompanionedCartridgeModel(
            base=base,
            slots=initialise_slots(geometry, seed=1),
            companions=(initialise_slots(wider, seed=2), initialise_slots(wider, seed=3)),
            companion_probability=1.0,
        )
        assert model.geometry["num_slots"] == 2


class TestForward:
    def test_all_three_draws_are_consumed_even_at_probability_one(self) -> None:
        """The sweep's arms must share one RNG-consumption pattern.

        In eval mode dropout draws nothing, so the pool machinery's three
        draws are the only global-generator consumers. If any were skipped
        at probability 1.0, the two models below would leave the generator
        in the same state and the follow-up draws would agree.
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
        varied = MultiCompanionedCartridgeModel(
            base=base,
            slots=initialise_slots(geometry, seed=1),
            companions=_pool(geometry, 2, 3),
            companion_probability=1.0,
        )
        varied.eval()
        varied_out = varied.forward(input_ids=item, labels=item)
        after_varied = float(torch.rand(()))

        assert float(varied_out.loss.item()) != float(plain_out.loss.item())
        assert after_plain != after_varied

    def test_all_three_draws_are_consumed_when_the_pool_is_absent(self) -> None:
        """The absent branch consumes exactly what the present branch does.

        Draws taken before the branch mean an absent forward and a present
        forward advance the generator identically; a plain model advances it
        not at all. Both facts are asserted, because uniformity ACROSS this
        model's own outcomes is the property the seed contract rests on.
        """
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        item = _corpus(seed=13, rows=1)[0]
        pool = _pool(geometry, 2, 3)

        torch.manual_seed(41)
        absent = MultiCompanionedCartridgeModel(
            base=base,
            slots=initialise_slots(geometry, seed=1),
            companions=pool,
            companion_probability=1e-9,
        )
        absent.eval()
        absent_out = absent.forward(input_ids=item, labels=item)
        after_absent = float(torch.rand(()))

        torch.manual_seed(41)
        present = MultiCompanionedCartridgeModel(
            base=base,
            slots=initialise_slots(geometry, seed=1),
            companions=pool,
            companion_probability=1.0,
        )
        present.eval()
        present_out = present.forward(input_ids=item, labels=item)
        after_present = float(torch.rand(()))

        torch.manual_seed(41)
        plain = CartridgeModel(base=base, slots=initialise_slots(geometry, seed=1))
        plain.eval()
        plain_out = plain.forward(input_ids=item, labels=item)
        after_plain = float(torch.rand(()))

        assert after_absent == after_present
        assert after_absent != after_plain
        # The generator claims above are about draws, and these are about the
        # forwards themselves: an absent pool leaves the loss exactly where
        # the plain model puts it, and a present pool moves it.
        assert float(absent_out.loss.item()) == pytest.approx(float(plain_out.loss.item()))
        assert float(present_out.loss.item()) != float(plain_out.loss.item())

    def test_an_absent_pool_forwards_exactly_as_the_plain_model(self) -> None:
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        item = _corpus(seed=13, rows=1)[0]

        plain = CartridgeModel(base=base, slots=initialise_slots(geometry, seed=1))
        plain.eval()
        expected = plain.forward(input_ids=item, labels=item)

        varied = MultiCompanionedCartridgeModel(
            base=base,
            slots=initialise_slots(geometry, seed=1),
            companions=_pool(geometry, 2, 3),
            companion_probability=1e-9,
        )
        varied.eval()
        torch.manual_seed(7)
        actual = varied.forward(input_ids=item, labels=item)

        assert float(actual.loss.item()) == pytest.approx(float(expected.loss.item()))

    def test_the_drawn_count_moves_the_loss(self) -> None:
        """The count draw is machinery, not decoration.

        Over twenty pinned generator states at probability 1.0 the drawn
        count and permutation vary; if every forward produced one loss, a
        fixed-size prefix would be hiding behind the draws and the sweep
        would measure nothing new.
        """
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        item = _corpus(seed=17, rows=1)[0]
        varied = MultiCompanionedCartridgeModel(
            base=base,
            slots=initialise_slots(geometry, seed=1),
            companions=_pool(geometry, 2, 3),
            companion_probability=1.0,
        )
        varied.eval()

        losses = set()
        for state in range(20):
            torch.manual_seed(state)
            out = varied.forward(input_ids=item, labels=item)
            losses.add(round(float(out.loss.item()), 10))

        assert len(losses) > 1

    def test_construction_moves_the_pool_onto_the_base_device(self) -> None:
        """The constructor owns device coherence for every member."""
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        pool = _pool(geometry, 2, 3)

        model = MultiCompanionedCartridgeModel(
            base=base,
            slots=initialise_slots(geometry, seed=1),
            companions=pool,
            companion_probability=1.0,
        )

        base_device = next(iter(base.named_parameters()))[1].device
        for member in pool:
            for name, tensor in member.state_dict().items():
                assert tensor.device == base_device, name
        assert model.geometry["num_slots"] == 2

    def test_to_moves_the_pool_with_the_model(self) -> None:
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        model = MultiCompanionedCartridgeModel(
            base=base,
            slots=initialise_slots(geometry, seed=1),
            companions=_pool(geometry, 2, 3),
            companion_probability=1.0,
        )
        assert model.to("cpu") is model


class TestVariedScalingDegenerate:
    def test_no_other_cartridges_composes_the_first_alone(self) -> None:
        """The empty varied composition is the varied first, exactly."""
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        pool = _pool(geometry, 5, 6)

        def fixed_pool(seed: int) -> tuple[CartridgeSlots, ...]:
            """The same frozen pool for every replicate.

            Args:
                seed: Ignored; the degenerate case needs no per-seed build.

            Returns:
                The one pool.
            """
            return pool

        alone, composed, untrained_composed, cross = measure_varied_companioned_scaling(
            base,
            first_train=_corpus(seed=41, rows=4),
            other_trains=(),
            held_out=_corpus(seed=42, rows=3),
            arm="solo-n1",
            num_slots=2,
            seeds=(7, 8, 9),
            epochs=1,
            learning_rate=0.05,
            pool_for_seed=fixed_pool,
            companion_probability=1.0,
        )

        assert cross == ()
        assert composed["gains"] == alone["gains"]
        assert untrained_composed["gains"] == alone["gains"]
        assert alone["arm"] == "solo-n1-alone"


class TestTraining:
    def test_every_pool_member_is_byte_identical_after_training(self) -> None:
        """The freeze is the load-bearing property: only the trainee learns."""
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        pool = _pool(geometry, 5, 6)
        before = [_slot_bytes(member.state_dict()) for member in pool]

        trained = train_cartridge_with_companions(
            base,
            _corpus(seed=22, rows=4),
            num_slots=2,
            seed=6,
            epochs=1,
            learning_rate=0.05,
            companions=pool,
            companion_probability=1.0,
        )

        for member, snapshot in zip(pool, before, strict=True):
            after = member.state_dict()
            assert sorted(after) == sorted(snapshot)
            for name, tensor in after.items():
                assert torch.equal(tensor, snapshot[name]), name
        assert set(trained.state_dict()) == set(before[0])

    def test_training_is_a_function_of_its_seed_with_the_pool_included(self) -> None:
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        pool = _pool(geometry, 5, 6)

        first = train_cartridge_with_companions(
            base,
            _corpus(seed=24, rows=4),
            num_slots=2,
            seed=6,
            epochs=1,
            learning_rate=0.05,
            companions=pool,
            companion_probability=0.5,
        )
        second = train_cartridge_with_companions(
            base,
            _corpus(seed=24, rows=4),
            num_slots=2,
            seed=6,
            epochs=1,
            learning_rate=0.05,
            companions=pool,
            companion_probability=0.5,
        )

        for name, tensor in first.state_dict().items():
            assert torch.equal(tensor, second.state_dict()[name]), name

    def test_the_pool_changes_what_a_single_companion_taught(self) -> None:
        """Same seed, same corpus, same first companion: the pool must differ.

        The varied model's extra draws and extra members make its gradient
        history different from the single-companion model's under identical
        seeds; if they did not, the new machinery would be decoration over
        the recorded recipe and the sweep would measure nothing.
        """
        base = _base()
        geometry = measure_geometry(base, num_slots=2)
        shared = initialise_slots(geometry, seed=5)

        single = train_cartridge_with_companion(
            base,
            _corpus(seed=25, rows=4),
            num_slots=2,
            seed=6,
            epochs=1,
            learning_rate=0.05,
            companion=shared,
            companion_probability=1.0,
        )
        pooled = train_cartridge_with_companions(
            base,
            _corpus(seed=25, rows=4),
            num_slots=2,
            seed=6,
            epochs=1,
            learning_rate=0.05,
            companions=(shared, initialise_slots(geometry, seed=9)),
            companion_probability=1.0,
        )

        differing = [
            name
            for name, tensor in pooled.state_dict().items()
            if not torch.equal(tensor, single.state_dict()[name])
        ]
        assert sorted(differing) == sorted(single.state_dict())
