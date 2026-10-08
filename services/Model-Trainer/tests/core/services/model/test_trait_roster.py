"""The seed gate: a derived SEI, a pilot's spread, and the seeds that resolve it.

Pure arithmetic over the real power module, so the expected values are
computed here from first principles -- the t critical value and the
derivation -- rather than by calling the function under test twice.
"""

from __future__ import annotations

import math

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.power_distributions import t_critical
from tests._trait_sweep_support import TINY_TRAIT_PLAN

from model_trainer.core.contracts.trait_plan import TraitPlan
from model_trainer.core.services.model.trait_roster import require_resolvable_seeds


def _rows(plan: TraitPlan) -> dict[str, float]:
    """Run the gate and map its rows by name.

    Args:
        plan: The plan to gate.

    Returns:
        Every row's value, keyed by name.
    """
    return {row["name"]: row["value"] for row in require_resolvable_seeds(plan)}


class TestTheDerivation:
    """The SEI is the acted-on retention carried into nats by the solo gain."""

    def test_the_sei_is_the_retention_times_the_alone_gain(self) -> None:
        """0.0526 of a 2.5-nat solo gain is 0.1315 nats of composed expression."""
        rows = _rows({**TINY_TRAIT_PLAN, "pilot_alone_gain": 2.5})
        assert rows["smallest_effect_of_interest"] == pytest.approx(0.0526 * 2.5)
        assert rows["acted_on_retention"] == 0.0526
        assert rows["pilot_alone_gain"] == 2.5

    def test_the_mde_is_at_the_declared_seeds_not_the_pilots(self) -> None:
        """Five declared seeds, a three-difference pilot: the MDE uses five."""
        plan: TraitPlan = {**TINY_TRAIT_PLAN, "seeds": (7, 8, 9, 10, 11)}
        rows = _rows(plan)
        assert rows["pilot_paired_sd"] == pytest.approx(0.01)
        assert rows["seed_paired_mde"] == pytest.approx(t_critical(4, 0.05) * 0.01 / math.sqrt(5))


class TestTheRefusal:
    """Too few seeds for the pilot's spread is refused, naming the count."""

    def test_three_seeds_cannot_resolve_a_noisy_pilot(self) -> None:
        """At an sd of 0.1, three seeds' MDE is 0.248 against an SEI of 0.0526."""
        plan: TraitPlan = {**TINY_TRAIT_PLAN, "pilot_paired_differences": (0.0, 0.1, 0.2)}
        with pytest.raises(AppError) as excinfo:
            require_resolvable_seeds(plan)
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_QA_UNDERPOWERED
        assert "composes over 3 seed(s)" in excinfo.value.message
        assert "paired sd of 0.1000 nats" in excinfo.value.message

    def test_the_required_count_clears_the_gate(self) -> None:
        """The count the refusal names is enough, and one fewer is not.

        At sd 0.1 and SEI 0.0526 the smallest n with t(n-1) * 0.1 / sqrt(n)
        at or below 0.0526 is found by direct search here, independently of
        the gate's own search.
        """
        sd, sei = 0.1, 0.0526
        needed = next(
            n for n in range(3, 200) if t_critical(n - 1, 0.05) * sd / math.sqrt(n) <= sei
        )
        noisy: TraitPlan = {**TINY_TRAIT_PLAN, "pilot_paired_differences": (0.0, 0.1, 0.2)}
        rows = _rows({**noisy, "seeds": tuple(range(needed))})
        assert rows["required_seeds"] == float(needed)
        with pytest.raises(AppError):
            require_resolvable_seeds({**noisy, "seeds": tuple(range(needed - 1))})
