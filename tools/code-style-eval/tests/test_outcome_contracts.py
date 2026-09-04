"""The outcome codecs a sweep's records round-trip through.

The prompt-construction half of this file moved to
:mod:`platform_core.continuation_task` when the generator and the scorer
became two packages that must agree on it.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from code_style_eval.contracts.outcomes import (
    CHECKERS,
    CheckOutcome,
    ItemOutcome,
    PairedCounts,
    as_checker,
    decode_check_outcome,
    decode_item_outcome,
    decode_paired_counts,
    encode_check_outcome,
    encode_item_outcome,
    encode_paired_counts,
)


class TestCheckerNarrowing:
    """The checker set is closed."""

    @pytest.mark.parametrize("checker", CHECKERS)
    def test_every_declared_checker_narrows(self, checker: str) -> None:
        """Iterating CHECKERS keeps this honest as the set grows.

        Args:
            checker: The checker name.
        """
        assert as_checker(checker, "checker") == checker

    def test_an_unknown_checker_is_refused(self) -> None:
        """A typo must not reach the runner."""
        with pytest.raises(JSONTypeError, match="checker"):
            _ = as_checker("pylint", "checker")


class TestOutcomeCodecs:
    """Every record round-trips, and a self-contradicting one is refused."""

    def test_a_check_outcome_round_trips(self) -> None:
        """Exit code survives, because a crash and a clean run differ."""
        outcome = CheckOutcome(checker="mypy", passed=False, exit_code=2, detail="boom")

        assert decode_check_outcome(encode_check_outcome(outcome)) == outcome

    def test_an_item_outcome_round_trips(self) -> None:
        """The per-checker rows survive, not just the summary."""
        checks = (
            CheckOutcome(checker="ruff", passed=True, exit_code=0, detail=""),
            CheckOutcome(checker="mypy", passed=True, exit_code=0, detail=""),
            CheckOutcome(checker="guards", passed=True, exit_code=0, detail=""),
        )
        outcome = ItemOutcome(item_id="a.py", arm="base", checks=checks, all_passed=True)

        assert decode_item_outcome(encode_item_outcome(outcome)) == outcome

    def test_a_summary_contradicting_its_rows_is_refused(self) -> None:
        """A record claiming a pass over a failing checker cannot be compared."""
        encoded = encode_item_outcome(
            ItemOutcome(
                item_id="a.py",
                arm="base",
                checks=(CheckOutcome(checker="ruff", passed=False, exit_code=1, detail="x"),),
                all_passed=False,
            )
        )
        encoded["all_passed"] = True

        with pytest.raises(JSONTypeError, match="disagrees with"):
            _ = decode_item_outcome(encoded)

    def test_paired_counts_round_trip(self) -> None:
        """The 2x2 table survives serialization."""
        counts = PairedCounts(both_passed=1, baseline_only=2, candidate_only=3, neither=4)

        assert decode_paired_counts(encode_paired_counts(counts)) == counts
