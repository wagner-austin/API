"""The arms, their controls, and the rows they put in a record.

THE SPLIT BETWEEN THESE TWO HALVES IS DELIBERATE. The measurement functions
run against a real tiny GPT-2, because what they assert is that training and
composition actually happen and are attributable to a seed. The assembly
functions are exercised against CONSTRUCTED arms, because the rule worth
checking there -- that a retention ratio is absent when the solo arm did not
improve -- needs an arm that failed, and forcing a real cartridge to fail its
own trait deterministically is harder than the failure itself.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.run_record import Observation

from model_trainer.core.contracts.replicated_measurement import replicate
from model_trainer.core.services.model.cartridge_scoring import TraitPair
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.services.model.steering_vectors import require_steerable
from model_trainer.core.services.model.trait_arms import (
    TraitArm,
    TraitCompositionArms,
    measure_trait_composition,
    measure_trait_solo,
    measure_trait_steering,
    steering_observations,
    trait_arm_observations,
    trait_cell_observations,
)
from model_trainer.core.types import CacheCapableLMProto, SteerableLMProto

_SITE = "transformer.h.0.mlp.c_proj"

#: Three is the fewest replicates a gain may be built from, and the arms
#: refuse below it, so every measurement here runs exactly three.
_SEEDS = (7, 8, 9)


def _model() -> SteerableLMProto:
    """Build the tiny probe model, narrowed to steerable.

    Returns:
        The model, which is also cache-capable and so serves both kinds of arm.
    """
    built, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return require_steerable(built)


def _pairs(offset: int, count: int = 4) -> list[TraitPair]:
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


def _arm(name: str, expression: tuple[float, ...], coherence: tuple[float, ...]) -> TraitArm:
    """Build one arm from chosen numbers.

    Args:
        name: The arm's name, without a reading suffix.
        expression: Per-seed expression gains.
        coherence: Per-seed coherence gains.

    Returns:
        The arm.
    """
    return TraitArm(
        expression=replicate(f"{name}-expression", list(zip(_SEEDS, expression, strict=True))),
        coherence=replicate(f"{name}-coherence", list(zip(_SEEDS, coherence, strict=True))),
    )


def _names(observations: Sequence[Observation]) -> set[str]:
    """Collect observation names.

    Args:
        observations: The rows.

    Returns:
        Their names.
    """
    return {row["name"] for row in observations}


@pytest.fixture(name="measured", scope="module")
def _measured() -> tuple[TraitArm, TraitArm]:
    """Run the solo cell once for the assertions that read it.

    Module-scoped because it trains three cartridges: every claim below is
    about one run's outcome, and retraining per assertion would multiply the
    cost without testing anything more.

    Returns:
        ``(trained, untrained)``.
    """
    base: CacheCapableLMProto = _model()
    pairs = _pairs(1, count=6)
    return measure_trait_solo(
        base,
        train=pairs[:4],
        held_out=pairs[4:],
        arm="bullets-solo",
        num_slots=4,
        seeds=_SEEDS,
        epochs=2,
        learning_rate=0.05,
    )


@pytest.fixture(name="cell", scope="module")
def _cell() -> TraitCompositionArms:
    """Run one n2 cell once.

    Returns:
        The cell.
    """
    base: CacheCapableLMProto = _model()
    primary = _pairs(1, count=6)
    other = _pairs(3, count=6)
    return measure_trait_composition(
        base,
        first_train=primary[:4],
        other_trains=[other[:4]],
        held_out=primary[4:],
        arm="bullets-n2",
        num_slots=4,
        seeds=_SEEDS,
        epochs=2,
        learning_rate=0.05,
    )


class TestTheSoloArmAndItsControl:
    """The precondition arm: did the trait land in the prefix at all?"""

    def test_both_readings_are_replicated_over_the_same_seeds(
        self, measured: tuple[TraitArm, TraitArm]
    ) -> None:
        """Expression and coherence must be pairable seed by seed.

        Args:
            measured: The solo cell.
        """
        trained, untrained = measured
        for arm in (trained, untrained):
            assert arm["expression"]["seeds"] == _SEEDS
            assert arm["coherence"]["seeds"] == _SEEDS

    def test_the_arms_are_named_apart(self, measured: tuple[TraitArm, TraitArm]) -> None:
        """The control's rows must not land under the trained arm's name.

        Args:
            measured: The solo cell.
        """
        trained, untrained = measured
        assert trained["expression"]["arm"] == "bullets-solo-expression"
        assert untrained["expression"]["arm"] == "bullets-solo-untrained-expression"

    def test_training_moves_the_arm_away_from_its_control(
        self, measured: tuple[TraitArm, TraitArm]
    ) -> None:
        """Without this the cell would be measuring the prefix's mere presence.

        Args:
            measured: The solo cell.
        """
        trained, untrained = measured
        assert trained["expression"]["mean"] != untrained["expression"]["mean"]

    def test_a_single_seed_is_refused(self) -> None:
        """A one-seed gain has no spread, so nothing can call it a gain."""
        pairs = _pairs(2, count=6)
        with pytest.raises(AppError) as excinfo:
            measure_trait_solo(
                _model(),
                train=pairs[:4],
                held_out=pairs[4:],
                arm="bullets-solo",
                num_slots=4,
                seeds=(7,),
                epochs=1,
                learning_rate=0.05,
            )
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_MEASUREMENT_UNREPLICATED


class TestTheComposedCell:
    """One trait with others in front of it, and the controls that explain it."""

    def test_one_cross_arm_per_other_trait(self, cell: TraitCompositionArms) -> None:
        """The leakage detector is per partner, not one number for all of them.

        Args:
            cell: The composed cell.
        """
        assert len(cell["cross"]) == 1
        assert cell["cross"][0]["expression"]["arm"] == "bullets-n2-cross-0-expression"

    def test_every_arm_carries_both_readings(self, cell: TraitCompositionArms) -> None:
        """An expression number with no coherence beside it cannot be read.

        Args:
            cell: The composed cell.
        """
        for arm in (cell["alone"], cell["composed"], cell["untrained_composed"], *cell["cross"]):
            assert arm["expression"]["seeds"] == _SEEDS
            assert arm["coherence"]["seeds"] == _SEEDS

    def test_the_untrained_control_differs_from_the_trained_composition(
        self, cell: TraitCompositionArms
    ) -> None:
        """The two attribute the cost differently: structure against content.

        If they were equal the composed loss would be entirely structural, and
        the run could not tell a longer prefix from interference. They are
        measured rather than assumed, so this asserts they are two numbers.

        Args:
            cell: The composed cell.
        """
        assert (
            cell["composed"]["expression"]["mean"]
            != (cell["untrained_composed"]["expression"]["mean"])
        )

    def test_the_alone_arm_is_the_same_configuration_as_a_solo_arm(
        self, cell: TraitCompositionArms
    ) -> None:
        """Its cartridge is drawn and trained exactly as the solo cell's is.

        So the composed cell carries its own denominator rather than borrowing
        the solo cell's, which is what lets a retention be read within one
        cell even if the two cells ran on different days.

        Args:
            cell: The composed cell.
        """
        assert cell["alone"]["expression"]["arm"] == "bullets-n2-alone-expression"
        assert len(cell["alone"]["expression"]["gains"]) == len(_SEEDS)


class TestTheSteeringArm:
    """Deterministic by construction, and reported as such."""

    def test_it_reports_one_reading_over_every_pair(self) -> None:
        """No seeds: a contrastive vector is a mean over a fixed pair set."""
        model = _model()
        primary = _pairs(1, count=6)
        reading = measure_trait_steering(
            model,
            trait_trains=[primary[:4]],
            held_out=primary[4:],
            arm="bullets-steer-n1",
            module_name=_SITE,
            strength=10.0,
        )
        assert reading["arm"] == "bullets-steer-n1"
        assert reading["expression"]["items"] == 2
        assert reading["coherence"]["items"] == 2

    def test_composing_two_traits_is_a_different_reading(self) -> None:
        """The composed arm must not silently be the solo arm again.

        This is the comparison the whole steering arm exists for, so an n2
        that equalled n1 would mean the second direction never entered.
        """
        model = _model()
        primary = _pairs(1, count=6)
        other = _pairs(3, count=6)
        solo = measure_trait_steering(
            model,
            trait_trains=[primary[:4]],
            held_out=primary[4:],
            arm="bullets-steer-n1",
            module_name=_SITE,
            strength=10.0,
        )
        composed = measure_trait_steering(
            model,
            trait_trains=[primary[:4], other[:4]],
            held_out=primary[4:],
            arm="bullets-steer-n2",
            module_name=_SITE,
            strength=10.0,
        )
        assert composed["expression"]["mean_treatment"] != solo["expression"]["mean_treatment"]

    def test_it_is_deterministic(self) -> None:
        """Two runs of one configuration must agree exactly.

        No judge model and no draw, so identical is the correct bar; anything
        weaker would let the arm's own variation be read as an effect.
        """
        model = _model()
        primary = _pairs(1, count=6)
        first = measure_trait_steering(
            model,
            trait_trains=[primary[:4]],
            held_out=primary[4:],
            arm="a",
            module_name=_SITE,
            strength=10.0,
        )
        second = measure_trait_steering(
            model,
            trait_trains=[primary[:4]],
            held_out=primary[4:],
            arm="a",
            module_name=_SITE,
            strength=10.0,
        )
        assert first == second


class TestTheRowsAnArmPutsInTheRecord:
    """Assembly, against constructed arms so every branch is reachable."""

    def test_an_arm_emits_both_readings_means_spreads_and_per_seed_gains(self) -> None:
        """One function emits both, so an arm cannot reach a record half-named."""
        rows = trait_arm_observations(_arm("x", (0.3, 0.4, 0.5), (-0.1, -0.2, -0.3)))
        names = _names(rows)
        assert {"x-expression_mean", "x-expression_spread"} <= names
        assert {"x-coherence_mean", "x-coherence_spread"} <= names
        assert {f"x-expression_seed{seed}_gain" for seed in _SEEDS} <= names

    def test_a_cell_reports_retention_when_the_solo_arm_improved(self) -> None:
        """The number the composition question is asked in."""
        cell = TraitCompositionArms(
            alone=_arm("c-alone", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
            composed=_arm("c-composed", (0.5, 0.5, 0.5), (0.0, 0.0, 0.0)),
            untrained_composed=_arm("c-untrained", (0.2, 0.2, 0.2), (0.0, 0.0, 0.0)),
            cross=(),
        )
        rows = trait_cell_observations("c", cell)
        retention = [row for row in rows if row["name"] == "c_expression_retention"]
        assert len(retention) == 1
        assert retention[0]["value"] == pytest.approx(0.5)

    def test_a_cell_omits_retention_when_the_solo_arm_did_not_improve(self) -> None:
        """A ratio against a non-gain has a sign and a size that mean nothing.

        A cartridge that failed to express its own trait is a RESULT this grid
        exists to find, so the raw arms carry it and the absent retention
        reads as "alone did not improve" -- checkable from the alone mean's
        sign in the same record.
        """
        cell = TraitCompositionArms(
            alone=_arm("c-alone", (-0.1, -0.2, -0.3), (0.0, 0.0, 0.0)),
            composed=_arm("c-composed", (-0.2, -0.3, -0.4), (0.0, 0.0, 0.0)),
            untrained_composed=_arm("c-untrained", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            cross=(),
        )
        names = _names(trait_cell_observations("c", cell))
        assert "c_expression_retention" not in names
        assert "c-alone-expression_mean" in names

    def test_a_cells_cross_arms_are_named_and_present(self) -> None:
        """Leakage is measured per partner, so every partner gets rows."""
        cell = TraitCompositionArms(
            alone=_arm("c-alone", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
            composed=_arm("c-composed", (0.5, 0.5, 0.5), (0.0, 0.0, 0.0)),
            untrained_composed=_arm("c-untrained", (0.2, 0.2, 0.2), (0.0, 0.0, 0.0)),
            cross=(_arm("c-cross-0", (0.0, 0.1, 0.2), (0.0, 0.0, 0.0)),),
        )
        names = _names(trait_cell_observations("c", cell))
        assert "c-cross-0-expression_mean" in names

    def test_a_steering_row_is_named_once_rather_than_mean(self) -> None:
        """The suffix is the only place the absence of a spread survives.

        A row called ``_mean`` beside rows that are means would invite a
        reader to compare its spread with theirs, and it has none.
        """
        model = _model()
        primary = _pairs(1, count=6)
        rows = steering_observations(
            measure_trait_steering(
                model,
                trait_trains=[primary[:4]],
                held_out=primary[4:],
                arm="s",
                module_name=_SITE,
                strength=10.0,
            )
        )
        names = _names(rows)
        assert "s-expression_once" in names
        assert "s-expression_mean" not in names
        assert {"s-expression_items", "s-expression_improved", "s-expression_p_value"} <= names
        assert "s-coherence_once" in names
