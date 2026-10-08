"""The corpus grid's recipes bound to trait pairs, on a real tiny GPT-2.

Each builder is asserted against the recorded function it binds -- the
replicates it emits must be the ones that function emits for the same
training items -- because the whole claim of the repair families is that the
levers are the RECORDED ones applied to a new substrate. A builder that
trained the same cartridges a different way would still produce a complete
table, and nothing downstream could tell.
"""

from __future__ import annotations

import torch

from model_trainer.core.services.finetuning.strategies.cartridge import (
    measure_geometry,
    require_cache_capable,
)
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.finetuning.strategies.cartridge_slots import (
    CartridgeSlots,
    initialise_slots,
)
from model_trainer.core.services.model.cartridge_measurement import (
    ComposedReplicate,
    composed_replicates,
)
from model_trainer.core.services.model.cartridge_scoring import TraitPair
from model_trainer.core.services.model.cartridge_varied import companioned_replicates
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.services.model.trait_arms import measure_trait_composition, trait_gains
from model_trainer.core.services.model.trait_corpus import training_items
from model_trainer.core.services.model.trait_families import (
    companioned_trait_build,
    measure_trait_companion_cross,
    plain_trait_build,
)
from model_trainer.core.types import CacheCapableLMProto

_SEEDS = (7, 8, 9)


def _base() -> CacheCapableLMProto:
    """Build the deterministic tiny GPT-2 the probe suite uses.

    Returns:
        The base, freshly initialised so every call starts from one state.
    """
    built, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return require_cache_capable(built)


def _pairs(offset: int, count: int) -> list[TraitPair]:
    """Build a trait's pairs, distinct from any other offset's.

    Args:
        offset: Shifts the token ranges, so two traits are different text.
        count: How many pairs.

    Returns:
        The pairs.
    """
    built: list[TraitPair] = []
    for index in range(count):
        generator = torch.Generator()
        generator.manual_seed(offset * 100 + index)
        built.append(
            TraitPair(
                expressing=torch.randint(
                    offset, offset + 60, (1, 6), generator=generator, dtype=torch.long
                ),
                neutral=torch.randint(
                    offset + 120, offset + 180, (1, 6), generator=generator, dtype=torch.long
                ),
            )
        )
    return built


class _FixedPool:
    """One frozen pool per seed, drawn once and served by identity."""

    def __init__(self, base: CacheCapableLMProto, seeds: tuple[int, ...] = _SEEDS) -> None:
        """Draw the pools.

        Args:
            base: The base the members are cut for.
            seeds: The replicates to draw a pool for.
        """
        geometry = measure_geometry(base, num_slots=4)
        self._pools = {
            seed: tuple(initialise_slots(geometry, seed=seed * 10 + member) for member in range(2))
            for seed in seeds
        }

    def pool(self, seed: int) -> tuple[CartridgeSlots, ...]:
        """Serve one replicate's pool.

        Args:
            seed: The replicate's seed.

        Returns:
            Two untrained members.
        """
        return self._pools[seed]


class _Collector:
    """A consumer that keeps every replicate, in the order handed over."""

    def __init__(self, into: list[ComposedReplicate]) -> None:
        """Hold the list.

        Args:
            into: The list to fill.
        """
        self._into = into

    def __call__(self, built: ComposedReplicate, /) -> None:
        """Keep one replicate.

        Args:
            built: The replicate.
        """
        self._into.append(built)


def _collect(into: list[ComposedReplicate]) -> _Collector:
    """Build a consumer that keeps every replicate it is handed.

    Args:
        into: The list to fill.

    Returns:
        The consumer.
    """
    return _Collector(into)


def _slots_equal(left: CartridgeSlots, right: CartridgeSlots) -> bool:
    """Compare two slot blocks tensor for tensor.

    Args:
        left: One block.
        right: The other.

    Returns:
        True when both carry the same tensors under the same names, bitwise.
    """
    left_state, right_state = left.state_dict(), right.state_dict()
    return left_state.keys() == right_state.keys() and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )


class TestThePlainBuilder:
    """It IS the recorded naive recipe, fed the expressing members."""

    def test_it_emits_what_the_recorded_function_emits(self) -> None:
        """Same items, same seeds: bitwise-equal cartridges in every slot."""
        primary, other = _pairs(1, 4), _pairs(3, 4)
        through_binding: list[ComposedReplicate] = []
        plain_trait_build(
            _base(),
            first_train=primary,
            other_trains=[other],
            num_slots=4,
            seeds=_SEEDS,
            seed_stride=len(_SEEDS),
            epochs=2,
            learning_rate=0.05,
        )(_collect(through_binding))
        direct: list[ComposedReplicate] = []
        composed_replicates(
            _base(),
            first_train=training_items(primary),
            other_trains=[training_items(other)],
            num_slots=4,
            seeds=_SEEDS,
            seed_stride=len(_SEEDS),
            epochs=2,
            learning_rate=0.05,
            consume=_collect(direct),
        )
        assert [built["seed"] for built in through_binding] == list(_SEEDS)
        for left, right in zip(through_binding, direct, strict=True):
            assert _slots_equal(left["alone"], right["alone"])
            assert _slots_equal(left["composed"], right["composed"])
            assert _slots_equal(left["untrained_composed"], right["untrained_composed"])


class TestABlockDrawsWhatAStraightRunDraws:
    """The property seed-block sharding rests on, for both recipes."""

    def test_a_plain_block_is_the_straight_runs_replicates(self) -> None:
        """Seeds 10-12 measured alone equal seeds 10-12 of a straight 7-12 run.

        Bitwise, every slot, because the stride is the plan's six rather than
        the block's three: the partner of seed 10 draws 16 either way. A
        stride taken from the call would draw 13 in the block -- seed 13's own
        primary draw in a longer plan -- and the record would hold cartridges
        no straight run would ever have trained.
        """
        primary, other = _pairs(1, 4), _pairs(3, 4)
        plan_seeds: tuple[int, ...] = (7, 8, 9, 10, 11, 12)
        later_block: tuple[int, ...] = (10, 11, 12)
        straight: list[ComposedReplicate] = []
        plain_trait_build(
            _base(),
            first_train=primary,
            other_trains=[other],
            num_slots=4,
            seeds=plan_seeds,
            seed_stride=len(plan_seeds),
            epochs=2,
            learning_rate=0.05,
        )(_collect(straight))
        block: list[ComposedReplicate] = []
        plain_trait_build(
            _base(),
            first_train=primary,
            other_trains=[other],
            num_slots=4,
            seeds=later_block,
            seed_stride=len(plan_seeds),
            epochs=2,
            learning_rate=0.05,
        )(_collect(block))
        for left, right in zip(straight[3:], block, strict=True):
            assert left["seed"] == right["seed"]
            assert _slots_equal(left["composed"], right["composed"])
            assert _slots_equal(left["untrained_composed"], right["untrained_composed"])

    def test_a_companioned_block_is_the_straight_runs_replicates(self) -> None:
        """The same property through the companioned recipe and its pools."""
        primary, other = _pairs(1, 4), _pairs(3, 4)
        plan_seeds: tuple[int, ...] = (7, 8, 9, 10, 11, 12)
        later_block: tuple[int, ...] = (10, 11, 12)
        built: list[list[ComposedReplicate]] = [[], []]
        for into, seeds in zip(built, (plan_seeds, later_block), strict=True):
            base = _base()
            companioned_trait_build(
                base,
                first_train=primary,
                other_trains=[other],
                num_slots=4,
                seeds=seeds,
                seed_stride=len(plan_seeds),
                epochs=2,
                learning_rate=0.05,
                pool_for_seed=_FixedPool(base, plan_seeds).pool,
                companion_probability=0.5,
            )(_collect(into))
        straight, block = built
        for left, right in zip(straight[3:], block, strict=True):
            assert left["seed"] == right["seed"]
            assert _slots_equal(left["composed"], right["composed"])


class TestTheCompanionedBuilder:
    """It IS the recorded companioned recipe, and the pool changes what it teaches."""

    def test_it_emits_what_the_recorded_function_emits(self) -> None:
        """Same items, same pools, same seeds: bitwise-equal cartridges."""
        primary, other = _pairs(1, 4), _pairs(3, 4)
        bound_base = _base()
        through_binding: list[ComposedReplicate] = []
        companioned_trait_build(
            bound_base,
            first_train=primary,
            other_trains=[other],
            num_slots=4,
            seeds=_SEEDS,
            seed_stride=len(_SEEDS),
            epochs=2,
            learning_rate=0.05,
            pool_for_seed=_FixedPool(bound_base).pool,
            companion_probability=0.5,
        )(_collect(through_binding))
        direct_base = _base()
        direct: list[ComposedReplicate] = []
        companioned_replicates(
            direct_base,
            first_train=training_items(primary),
            other_trains=[training_items(other)],
            num_slots=4,
            seeds=_SEEDS,
            seed_stride=len(_SEEDS),
            epochs=2,
            learning_rate=0.05,
            pool_for_seed=_FixedPool(direct_base).pool,
            companion_probability=0.5,
            consume=_collect(direct),
        )
        for left, right in zip(through_binding, direct, strict=True):
            assert _slots_equal(left["alone"], right["alone"])
            assert _slots_equal(left["composed"], right["composed"])

    def test_the_pool_changes_the_trained_cartridge(self) -> None:
        """A companioned build that matched the plain one would be no recipe."""
        primary, other = _pairs(1, 4), _pairs(3, 4)
        plain: list[ComposedReplicate] = []
        plain_trait_build(
            _base(),
            first_train=primary,
            other_trains=[other],
            num_slots=4,
            seeds=_SEEDS,
            seed_stride=len(_SEEDS),
            epochs=2,
            learning_rate=0.05,
        )(_collect(plain))
        base = _base()
        companioned: list[ComposedReplicate] = []
        companioned_trait_build(
            base,
            first_train=primary,
            other_trains=[other],
            num_slots=4,
            seeds=_SEEDS,
            seed_stride=len(_SEEDS),
            epochs=2,
            learning_rate=0.05,
            pool_for_seed=_FixedPool(base).pool,
            companion_probability=0.5,
        )(_collect(companioned))
        assert not _slots_equal(plain[0]["alone"], companioned[0]["alone"])

    def test_a_companioned_cell_scores_into_the_trait_arms(self) -> None:
        """The scorer takes either builder and names the cell's arms the same way."""
        primary, other = _pairs(1, 6), _pairs(3, 6)
        base = _base()
        cell = measure_trait_composition(
            base,
            build=companioned_trait_build(
                base,
                first_train=primary[:4],
                other_trains=[other[:4]],
                num_slots=4,
                seeds=_SEEDS,
                seed_stride=len(_SEEDS),
                epochs=2,
                learning_rate=0.05,
                pool_for_seed=_FixedPool(base).pool,
                companion_probability=0.5,
            ),
            partners=1,
            held_out=primary[4:],
            arm="bullets-diverse-n2",
        )
        assert cell["composed"]["expression"]["arm"] == "bullets-diverse-n2-composed-expression"
        assert cell["cross"][0]["coherence"]["seeds"] == _SEEDS


class TestTheCompanionCross:
    """Every pool member, scored alone, on the primary trait's pairs."""

    def test_each_member_arm_is_that_member_scored_alone(self) -> None:
        """The per-seed gains are exactly the member's own reading, not a mean of others."""
        base = _base()
        pools = _FixedPool(base)
        held_out = _pairs(1, 2)
        arms = measure_trait_companion_cross(
            base,
            pool_for_seed=pools.pool,
            pool_size=2,
            seeds=_SEEDS,
            held_out=held_out,
            arm="bullets-companion-cross",
        )
        assert [arm["expression"]["arm"] for arm in arms] == [
            "bullets-companion-cross-0-expression",
            "bullets-companion-cross-1-expression",
        ]
        gains = trait_gains(CartridgeModel(base=base, slots=pools.pool(8)[1]), held_out)
        assert arms[1]["expression"]["gains"][1] == gains["expression"]
        assert arms[1]["coherence"]["gains"][1] == gains["coherence"]
        assert arms[1]["style"]["gains"][1] == gains["style"]
