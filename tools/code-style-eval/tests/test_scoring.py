"""The guard-pass rates, and the table the paired test is handed.

The load-bearing property is that only DISCORDANT items move the p-value.
Two arms that pass and fail exactly the same files carry no evidence of a
difference no matter how many files there are, and a test that reported
otherwise would let a sweep claim an improvement it never measured.

The p-values themselves are not computed here and are not tested here:
:mod:`platform_core.power_distributions` owns both McNemar variants for the
whole monorepo. That they still produce this package's PUBLISHED digits is
held by ``test_published_comparisons.py``, against the committed artifacts
rather than against a fixture, because a fixture written beside the code
agrees with it by construction.
"""

from __future__ import annotations

from code_style_eval.contracts.outcomes import CheckOutcome, ItemOutcome, PairedCounts
from code_style_eval.core.scoring import (
    discordant_split,
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


class TestDiscordantSplit:
    """The projection onto the two numbers a conditional test takes.

    The ARITHMETIC is no longer tested here: it lives in
    :mod:`platform_core.power_distributions`, which owns it for the whole
    monorepo and tests it there. What is this package's, and what these
    tests hold, is which cells of a guard-pass table reach the test at all.
    """

    def test_no_discordant_pairs_is_an_empty_split(self) -> None:
        """Identical arms carry no evidence of a difference.

        Not a sentinel: if both arms passed and failed exactly the same
        items, the data say nothing about which is better, and the split the
        test is handed says so.
        """
        counts = PairedCounts(both_passed=500, baseline_only=0, candidate_only=0, neither=500)

        assert discordant_split(counts) == (0, 0)

    def test_a_large_concordant_count_does_not_reach_the_test(self) -> None:
        """The trap, stated as a test.

        A thousand items where both arms agree, and one discordant pair, is
        weak evidence. A test that pooled the arms would report a tiny
        p-value off the sample size alone, so the concordant cells must be
        dropped by the projection rather than trusted to cancel later.
        """
        counts = PairedCounts(both_passed=999, baseline_only=0, candidate_only=1, neither=0)

        assert discordant_split(counts) == (0, 1)

    def test_the_minority_cell_is_the_one_returned(self) -> None:
        """Two fixed against nine broken hands over 2, not 9."""
        counts = PairedCounts(both_passed=3, baseline_only=9, candidate_only=2, neither=4)

        assert discordant_split(counts) == (2, 11)

    def test_the_split_is_symmetric(self) -> None:
        """Swapping the arms cannot change the strength of the evidence.

        The two-sided p-value is symmetric, and returning the minority is
        what makes that a property of the projection rather than something
        the test has to restore.
        """
        forward = PairedCounts(both_passed=3, baseline_only=2, candidate_only=9, neither=4)
        reversed_arms = PairedCounts(both_passed=3, baseline_only=9, candidate_only=2, neither=4)

        assert discordant_split(forward) == discordant_split(reversed_arms)

    def test_an_even_split_reports_the_shared_value(self) -> None:
        """Five fixed and five broken is a 5-of-10 split, the tie form."""
        counts = PairedCounts(both_passed=0, baseline_only=5, candidate_only=5, neither=0)

        assert discordant_split(counts) == (5, 10)


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
