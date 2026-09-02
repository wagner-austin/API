"""The paired statistics, including the trap the spec warned about.

The load-bearing property is that only DISCORDANT items move the p-value.
Two arms that pass and fail exactly the same files carry no evidence of a
difference no matter how many files there are, and a test that reported
otherwise would let a sweep claim an improvement it never measured.
"""

from __future__ import annotations

from code_style_eval.contracts.outcomes import CheckOutcome, ItemOutcome, PairedCounts
from code_style_eval.core.scoring import (
    exact_mcnemar_p,
    mid_p_mcnemar_p,
    net_improvement,
    paired_counts,
    pass_rate,
    per_checker_rates,
)


def _outcome(item_id: str, arm: str, *, ruff: bool, mypy: bool, guards: bool) -> ItemOutcome:
    """Build an outcome with the three checkers set explicitly.

    Args:
        item_id: The item.
        arm: The arm.
        ruff: Whether ruff passed.
        mypy: Whether mypy passed.
        guards: Whether the guards passed.

    Returns:
        The outcome.
    """
    checks = (
        CheckOutcome(checker="ruff", passed=ruff, exit_code=0 if ruff else 1, detail=""),
        CheckOutcome(checker="mypy", passed=mypy, exit_code=0 if mypy else 1, detail=""),
        CheckOutcome(checker="guards", passed=guards, exit_code=0 if guards else 1, detail=""),
    )
    return ItemOutcome(
        item_id=item_id,
        arm=arm,
        checks=checks,
        all_passed=ruff and mypy and guards,
    )


def _passing(item_id: str, arm: str) -> ItemOutcome:
    """An outcome where every checker passed.

    Args:
        item_id: The item.
        arm: The arm.

    Returns:
        The outcome.
    """
    return _outcome(item_id, arm, ruff=True, mypy=True, guards=True)


def _failing(item_id: str, arm: str) -> ItemOutcome:
    """An outcome where the guards failed.

    Args:
        item_id: The item.
        arm: The arm.

    Returns:
        The outcome.
    """
    return _outcome(item_id, arm, ruff=True, mypy=True, guards=False)


class TestPassRate:
    """The combined rate over items."""

    def test_all_passing_is_one(self) -> None:
        """Three of three."""
        outcomes = [_passing(f"a{i}.py", "base") for i in range(3)]

        assert pass_rate(outcomes) == 1.0

    def test_mixed_is_the_fraction(self) -> None:
        """One of four."""
        outcomes = [_passing("a.py", "base"), *(_failing(f"b{i}.py", "base") for i in range(3))]

        assert pass_rate(outcomes) == 0.25

    def test_an_empty_sweep_is_zero_rather_than_a_crash(self) -> None:
        """An empty sweep has no rate; the item count reports the emptiness."""
        assert pass_rate([]) == 0.0


class TestPerCheckerRates:
    """A model can be clean for ruff and wrong for the guards."""

    def test_each_checker_is_scored_separately(self) -> None:
        """The combined rate hides which discipline failed; this does not."""
        outcomes = [
            _outcome("a.py", "base", ruff=True, mypy=True, guards=False),
            _outcome("b.py", "base", ruff=True, mypy=False, guards=False),
        ]

        rates = per_checker_rates(outcomes)

        assert rates["ruff"] == 1.0
        assert rates["mypy"] == 0.5
        assert rates["guards"] == 0.0

    def test_an_empty_sweep_scores_every_checker_zero(self) -> None:
        """Every checker is still named, so a reader sees the full shape."""
        rates = per_checker_rates([])

        assert rates == {"ruff": 0.0, "mypy": 0.0, "guards": 0.0}


class TestPairedCounts:
    """The 2x2 table, over shared items only."""

    def test_the_four_cells_are_counted(self) -> None:
        """One item of each kind."""
        baseline = {
            "both.py": _passing("both.py", "base"),
            "base_only.py": _passing("base_only.py", "base"),
            "cand_only.py": _failing("cand_only.py", "base"),
            "neither.py": _failing("neither.py", "base"),
        }
        candidate = {
            "both.py": _passing("both.py", "cand"),
            "base_only.py": _failing("base_only.py", "cand"),
            "cand_only.py": _passing("cand_only.py", "cand"),
            "neither.py": _failing("neither.py", "cand"),
        }

        counts = paired_counts(baseline, candidate)

        assert counts == PairedCounts(both_passed=1, baseline_only=1, candidate_only=1, neither=1)

    def test_an_item_only_one_arm_produced_is_excluded(self) -> None:
        """A missing generation is a fact about the run, not about the model.

        Counting it against the arm that lacks it would credit whichever arm
        happened to generate more often.
        """
        baseline = {
            "shared.py": _passing("shared.py", "base"),
            "base_extra.py": _passing("base_extra.py", "base"),
        }
        candidate = {"shared.py": _failing("shared.py", "cand")}

        counts = paired_counts(baseline, candidate)

        assert counts["baseline_only"] == 1
        assert counts["both_passed"] == 0
        assert counts["candidate_only"] == 0
        assert counts["neither"] == 0


class TestExactMcNemar:
    """Only discordant pairs carry evidence."""

    def test_no_discordant_pairs_is_p_one(self) -> None:
        """Identical arms carry no evidence of a difference.

        Not a sentinel: if both arms passed and failed exactly the same
        items, the data say nothing about which is better.
        """
        counts = PairedCounts(both_passed=500, baseline_only=0, candidate_only=0, neither=500)

        assert exact_mcnemar_p(counts) == 1.0

    def test_a_large_concordant_count_does_not_manufacture_significance(self) -> None:
        """The trap, stated as a test.

        A thousand items where both arms agree, and one discordant pair, is
        weak evidence. A test that pooled the arms would report a tiny
        p-value off the sample size alone.
        """
        counts = PairedCounts(both_passed=999, baseline_only=0, candidate_only=1, neither=0)

        assert exact_mcnemar_p(counts) == 1.0

    def test_a_perfectly_one_sided_split_is_significant(self) -> None:
        """Ten fixed and none broken is 2 * 0.5**10."""
        counts = PairedCounts(both_passed=0, baseline_only=0, candidate_only=10, neither=0)

        assert exact_mcnemar_p(counts) == 2.0 / 1024.0

    def test_an_even_split_is_p_one(self) -> None:
        """Five fixed and five broken is no evidence either way."""
        counts = PairedCounts(both_passed=0, baseline_only=5, candidate_only=5, neither=0)

        assert exact_mcnemar_p(counts) == 1.0

    def test_the_p_value_is_symmetric(self) -> None:
        """Swapping the arms cannot change the strength of the evidence."""
        forward = PairedCounts(both_passed=3, baseline_only=2, candidate_only=9, neither=4)
        reversed_arms = PairedCounts(both_passed=3, baseline_only=9, candidate_only=2, neither=4)

        assert exact_mcnemar_p(forward) == exact_mcnemar_p(reversed_arms)

    def test_a_p_value_never_exceeds_one(self) -> None:
        """The doubled tail is clamped, which matters at the even split."""
        counts = PairedCounts(both_passed=0, baseline_only=1, candidate_only=1, neither=0)

        assert exact_mcnemar_p(counts) <= 1.0


class TestNetImprovement:
    """The effect size the p-value qualifies."""

    def test_fixed_minus_broken(self) -> None:
        """Seven fixed, two broken, net five."""
        counts = PairedCounts(both_passed=10, baseline_only=2, candidate_only=7, neither=1)

        assert net_improvement(counts) == 5

    def test_a_regression_is_negative(self) -> None:
        """A candidate that breaks more than it fixes says so."""
        counts = PairedCounts(both_passed=0, baseline_only=6, candidate_only=1, neither=0)

        assert net_improvement(counts) == -5


class TestMidP:
    """The variant this package reports, and why it differs from exact."""

    def test_no_discordant_pairs_is_p_one(self) -> None:
        """Same answer as exact: identical arms carry no evidence."""
        counts = PairedCounts(both_passed=9, baseline_only=0, candidate_only=0, neither=9)

        assert mid_p_mcnemar_p(counts) == 1.0

    def test_mid_p_is_the_exact_value_minus_the_point_probability(self) -> None:
        """Ten fixed, none broken: 2/1024 exact, less 1/1024, is 1/1024."""
        counts = PairedCounts(both_passed=0, baseline_only=0, candidate_only=10, neither=0)

        assert exact_mcnemar_p(counts) == 2.0 / 1024.0
        assert mid_p_mcnemar_p(counts) == 1.0 / 1024.0

    def test_an_even_split_uses_its_own_form(self) -> None:
        """With the cells equal the general form would exceed 1.

        Five and five over ten discordant pairs: 1 - C(10,5)/2**11.
        """
        counts = PairedCounts(both_passed=0, baseline_only=5, candidate_only=5, neither=0)

        assert exact_mcnemar_p(counts) == 1.0
        assert mid_p_mcnemar_p(counts) == 1.0 - (252.0 / 1024.0) / 2.0

    def test_mid_p_is_never_more_conservative_than_exact(self) -> None:
        """The property the whole substitution rests on.

        Swept rather than spot-checked: for every table up to 12 discordant
        pairs the mid-p value must be no larger than the exact one, because
        it is that value less a non-negative point probability.
        """
        for baseline_only in range(13):
            for candidate_only in range(13):
                counts = PairedCounts(
                    both_passed=4,
                    baseline_only=baseline_only,
                    candidate_only=candidate_only,
                    neither=4,
                )
                assert mid_p_mcnemar_p(counts) <= exact_mcnemar_p(counts)

    def test_mid_p_stays_in_the_unit_interval(self) -> None:
        """A probability that left [0, 1] would be a reporting bug."""
        for baseline_only in range(13):
            for candidate_only in range(13):
                counts = PairedCounts(
                    both_passed=0,
                    baseline_only=baseline_only,
                    candidate_only=candidate_only,
                    neither=0,
                )
                value = mid_p_mcnemar_p(counts)
                assert 0.0 <= value <= 1.0

    def test_the_conservativeness_gap_changes_a_verdict(self) -> None:
        """The measured reason for preferring mid-p, as a test.

        Nine items fixed against two broken: the exact test returns 0.065 and
        declines to call it at the 0.05 level, the mid-p test returns 0.039
        and calls it. Fagerland et al. found that pattern across 9,595
        scenarios, which is why this package reports the latter.
        """
        counts = PairedCounts(both_passed=3, baseline_only=2, candidate_only=9, neither=4)

        assert exact_mcnemar_p(counts) > 0.05
        assert mid_p_mcnemar_p(counts) < 0.05

    def test_mid_p_is_symmetric(self) -> None:
        """Swapping the arms cannot change the strength of the evidence."""
        forward = PairedCounts(both_passed=3, baseline_only=2, candidate_only=9, neither=4)
        reversed_arms = PairedCounts(both_passed=3, baseline_only=9, candidate_only=2, neither=4)

        assert mid_p_mcnemar_p(forward) == mid_p_mcnemar_p(reversed_arms)
