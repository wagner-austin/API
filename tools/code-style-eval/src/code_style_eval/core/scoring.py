"""Guard-pass rates, and the paired test that compares two arms.

WHY NOT A T-TEST. The outcome per item is a BOOLEAN -- the generated file
either passed every checker or did not. Two arms are scored on the SAME
held-out items, so their results are paired, and the items differ enormously
in difficulty: a short module with two imports is easy for both arms, a
600-line service with nested TypedDicts is hard for both. Pooling those into
two independent samples and running a t-test throws away the pairing, and the
between-item variance it then measures is dominated by which files were
sampled rather than by which model wrote the code.

The paired structure means only the DISCORDANT items carry information about
a difference: items where one arm passed and the other failed. Items both
arms passed, and items both failed, say the same thing about each arm and
cancel. That is McNemar's insight, and the exact form of his test is a
two-sided binomial on the discordant pairs, which is what this module
computes. Exact rather than the chi-square approximation because a guard-pass
sweep over a few hundred held-out files routinely produces single-digit
discordant counts, which is exactly where the approximation is worst.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import comb

from code_style_eval.contracts.outcomes import CHECKERS, ItemOutcome, PairedCounts


def pass_rate(outcomes: Sequence[ItemOutcome]) -> float:
    """Fraction of items where every checker passed.

    Args:
        outcomes: One arm's outcomes.

    Returns:
        The fraction in [0, 1], or 0.0 when there are no items. An empty
        sweep has no rate; reporting 0.0 rather than raising keeps the
        summary printable, and the item count is reported beside it so an
        empty sweep is never mistaken for a total failure.
    """
    if not outcomes:
        return 0.0
    return sum(1 for outcome in outcomes if outcome["all_passed"]) / len(outcomes)


def per_checker_rates(outcomes: Sequence[ItemOutcome]) -> dict[str, float]:
    """Fraction of items each individual checker passed.

    Reported alongside the combined rate because the checkers measure
    different things: a model can be syntactically clean and architecturally
    wrong, and a single combined number hides which.

    Args:
        outcomes: One arm's outcomes.

    Returns:
        A rate per checker name, over the same denominator as
        :func:`pass_rate`.
    """
    rates: dict[str, float] = {}
    for checker in CHECKERS:
        if not outcomes:
            rates[checker] = 0.0
            continue
        passed = sum(
            1
            for outcome in outcomes
            for check in outcome["checks"]
            if check["checker"] == checker and check["passed"]
        )
        rates[checker] = passed / len(outcomes)
    return rates


def paired_counts(
    baseline: Mapping[str, ItemOutcome], candidate: Mapping[str, ItemOutcome]
) -> PairedCounts:
    """Build the 2x2 table over the items both arms were scored on.

    Only items present in BOTH arms are counted. An item one arm never
    produced a completion for is not evidence about the other arm, and
    silently treating it as a failure would credit whichever arm happened to
    generate more often.

    Args:
        baseline: The baseline arm's outcomes, keyed by item id.
        candidate: The candidate arm's outcomes, keyed by item id.

    Returns:
        The four counts.
    """
    shared = sorted(set(baseline) & set(candidate))
    counts = PairedCounts(both_passed=0, baseline_only=0, candidate_only=0, neither=0)
    for item_id in shared:
        base_ok = baseline[item_id]["all_passed"]
        cand_ok = candidate[item_id]["all_passed"]
        if base_ok and cand_ok:
            counts["both_passed"] += 1
        elif base_ok:
            counts["baseline_only"] += 1
        elif cand_ok:
            counts["candidate_only"] += 1
        else:
            counts["neither"] += 1
    return counts


def _binomial_tail(successes: int, trials: int) -> float:
    """Probability of a result at least as extreme, under p = 0.5.

    Args:
        successes: Count in one discordant cell.
        trials: Total discordant pairs.

    Returns:
        The two-sided exact binomial p-value.
    """
    extreme = min(successes, trials - successes)
    tail = sum(comb(trials, k) for k in range(extreme + 1))
    # ``1 << trials`` rather than ``2 ** trials``: the latter is typed Any,
    # because a negative exponent would make it a float. Both are exact
    # integers here, and staying in integer arithmetic until the final
    # division is what keeps the p-value exact rather than accumulated.
    two_sided = 2.0 * tail / float(1 << trials)
    return min(1.0, two_sided)


def exact_mcnemar_p(counts: PairedCounts) -> float:
    """Two-sided exact McNemar p-value for the discordant pairs.

    Args:
        counts: The 2x2 table.

    Returns:
        The p-value. Returns 1.0 when there are no discordant pairs, which is
        the correct answer rather than a sentinel: if the two arms passed and
        failed exactly the same items, the data contain no evidence of any
        difference at all.
    """
    discordant = counts["baseline_only"] + counts["candidate_only"]
    if discordant == 0:
        return 1.0
    return _binomial_tail(counts["candidate_only"], discordant)


def net_improvement(counts: PairedCounts) -> int:
    """Items the candidate fixed minus items it broke.

    Args:
        counts: The 2x2 table.

    Returns:
        A positive number when the candidate passes more items than the
        baseline. This is the effect size the p-value qualifies, and it is
        reported beside it because a significant p on two discordant pairs
        still describes two files.
    """
    return counts["candidate_only"] - counts["baseline_only"]


__all__ = [
    "exact_mcnemar_p",
    "net_improvement",
    "paired_counts",
    "pass_rate",
    "per_checker_rates",
]
