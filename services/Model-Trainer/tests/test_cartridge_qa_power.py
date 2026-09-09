"""The precondition that would have refused the retracted headline.

Every number asserted here is derived from the real
:mod:`platform_core.minimum_detectable_effect`, not from this module's own
output, and the historical case is pinned as a test of its own: 32 items, a
claimed difference of 0.0417, refused.
"""

from __future__ import annotations

import pytest
from platform_core.error_codes import ModelTrainerErrorCode
from platform_core.errors import AppError
from platform_core.power_distributions import McNemarTest

from model_trainer.core.services.model.cartridge_qa_plans import QA_PLANS, QaPlan
from model_trainer.core.services.model.cartridge_qa_power import (
    require_resolvable_question_set,
    resolvable_floor,
    smallest_rejecting_discordant,
)


def _plan(effect: float, alpha: float = 0.05, test: McNemarTest = McNemarTest.MID_P) -> QaPlan:
    """Build a plan that differs from a real one only where a test needs it.

    Args:
        effect: The plan's declared smallest effect of interest.
        alpha: Two-sided significance level.
        test: Which McNemar variant the plan is judged under.

    Returns:
        A complete :class:`QaPlan`.
    """
    return QaPlan(
        model_id="gpt2",
        window=256,
        held_out_stride=4,
        num_slots=128,
        max_seq_len=896,
        seeds=(7, 8, 9),
        epochs=12,
        learning_rate=0.01,
        distractor_count=3,
        max_items=120,
        smallest_effect_of_interest=effect,
        alpha=alpha,
        mcnemar_test=test,
        bm25_k1=1.5,
        bm25_b=0.75,
        retrieved_chunks=5,
    )


class TestTheSmallestRejectingDiscordantCount:
    """The constant the whole floor rests on, against the real power module."""

    @pytest.mark.parametrize(
        ("alpha", "test", "expected"),
        [
            (0.05, McNemarTest.MID_P, 5),
            (0.05, McNemarTest.EXACT, 6),
            (0.01, McNemarTest.MID_P, 7),
            (0.01, McNemarTest.EXACT, 8),
        ],
    )
    def test_matches_the_power_module(self, alpha: float, test: McNemarTest, expected: int) -> None:
        """Mid-p rejects sooner than exact, which is why it is the default.

        Args:
            alpha: Two-sided significance level.
            test: Which McNemar variant.
            expected: The count measured against the real power module.
        """
        assert smallest_rejecting_discordant(alpha, test) == expected

    def test_an_unreachable_alpha_is_refused_rather_than_searched_forever(self) -> None:
        """A ceiling that fails loudly beats a loop that does not terminate."""
        with pytest.raises(AppError) as excinfo:
            smallest_rejecting_discordant(1e-30, McNemarTest.MID_P)

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_QA_UNDERPOWERED


class TestTheFloor:
    """What a question set of a given size could ever resolve."""

    @pytest.mark.parametrize(
        ("item_count", "expected"),
        [(32, 5 / 32), (120, 5 / 120), (235, 5 / 235), (2627, 5 / 2627)],
    )
    def test_the_floor_is_the_rejecting_count_over_the_items(
        self, item_count: int, expected: float
    ) -> None:
        """Args:
        item_count: Items in the question set.
        expected: The floor those items imply.
        """
        assert resolvable_floor(item_count, 0.05, McNemarTest.MID_P) == pytest.approx(expected)

    def test_more_items_resolve_smaller_differences(self) -> None:
        """The direction the whole gate depends on."""
        coarse = resolvable_floor(32, 0.05, McNemarTest.MID_P)
        fine = resolvable_floor(2627, 0.05, McNemarTest.MID_P)

        assert fine < coarse

    def test_an_empty_question_set_is_refused_rather_than_divided_by(self) -> None:
        """Zero items is a corpus failure surfacing here, not a floor."""
        with pytest.raises(AppError) as excinfo:
            resolvable_floor(0, 0.05, McNemarTest.MID_P)

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_QA_UNDERPOWERED


class TestTheHistoricalCase:
    """2026-09-09, and the refusal that would have prevented it."""

    def test_thirty_two_items_cannot_resolve_the_retracted_difference(self) -> None:
        """The published claim was 0.0417 at 774M and 0.0521 at 1.5B.

        Both sit below the 0.1562 that 32 items can resolve, so no split of
        that question set would have supported either.
        """
        with pytest.raises(AppError) as excinfo:
            require_resolvable_question_set(_plan(0.0417), 32)

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_QA_UNDERPOWERED

    def test_the_refusal_names_the_item_count_that_would_be_enough(self) -> None:
        """A refusal without a target is an obstacle rather than a next step."""
        with pytest.raises(AppError) as excinfo:
            require_resolvable_question_set(_plan(0.05), 32)

        assert "100 items" in str(excinfo.value)

    def test_the_refusal_states_the_claim_in_items_before_rates(self) -> None:
        """A rate looks like a finding at any size; a count does not.

        The retracted headline read as a result because it was written
        "+0.0417" rather than "1.3 items of 32", so the refusal leads with
        the counts and gives the rates afterwards.
        """
        with pytest.raises(AppError) as excinfo:
            require_resolvable_question_set(_plan(0.0417), 32)

        message = str(excinfo.value)
        assert "1.3 item(s) of 32" in message
        assert "5 of 32" in message

    def test_the_corpus_that_was_actually_needed_passes(self) -> None:
        """235 items clear 0.05, which is why the bigger wiki was the fix."""
        floor = require_resolvable_question_set(_plan(0.05), 235)

        assert floor == pytest.approx(5 / 235)

    def test_a_plan_hunting_a_large_effect_may_legitimately_use_few_items(self) -> None:
        """The gate refuses unresolvable CLAIMS, not small question sets.

        Cartridge-versus-base moved about 0.27 at these rungs. A plan that
        says so is asking something 32 items can answer, and refusing it
        would make the gate a blanket ban on small corpora rather than a
        check that a claim is answerable.
        """
        floor = require_resolvable_question_set(_plan(0.20), 32)

        assert floor == pytest.approx(5 / 32)


class TestTheShippedPlans:
    """The declarations the registry actually carries."""

    @pytest.mark.parametrize("name", sorted(QA_PLANS))
    def test_every_plan_declares_what_it_is_hunting(self, name: str) -> None:
        """A plan that cannot say this cannot be told it failed to find one.

        Args:
            name: Plan name in the registry.
        """
        plan = QA_PLANS[name]

        assert 0.0 < plan["smallest_effect_of_interest"] < 1.0
        assert 0.0 < plan["alpha"] < 1.0
        assert plan["mcnemar_test"] in tuple(McNemarTest)

    @pytest.mark.parametrize("name", sorted(QA_PLANS))
    def test_every_plan_needs_two_hundred_and_fifty_items_at_its_declared_effect(
        self, name: str
    ) -> None:
        """0.02 under mid-p at alpha 0.05 means 250 items, for every plan.

        Pinned so that lowering a plan's declared effect without growing its
        corpus fails here rather than at the end of a GPU run.

        THIS TEST CAUGHT THE CHANGE IT WAS WRITTEN FOR, which is why the
        number moved rather than the pin being deleted. It read 100 items at
        0.05 until 2026-09-09, when the SEI was corrected to 0.02 -- the
        lowest non-noise effect the plan table's own anchor list cites, where
        0.05 had been roughly the middle of that list and described as its
        floor. Lowering the effect without growing the corpus is exactly what
        happened, and this assertion is where it surfaced.

        The consequence is deliberate and is stated in the table's comment:
        at 250 items EVERY plan in the registry is now refused, including the
        two api-wiki plans added the same morning to fix the power problem
        (235 raw and 224 reshaped items against the 250 required). The corpus
        is what moves next. Raising the SEI back so the existing corpus
        clears it would make this test pass and mean nothing.

        Args:
            name: Plan name in the registry.
        """
        plan = QA_PLANS[name]

        with pytest.raises(AppError):
            require_resolvable_question_set(plan, 249)
        assert require_resolvable_question_set(plan, 250) == pytest.approx(0.02)
