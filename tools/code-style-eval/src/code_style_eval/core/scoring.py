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
cancel. That is McNemar's insight, and his test is a two-sided binomial on
the discordant pairs. Conditional rather than the chi-square approximation
because a guard-pass sweep over a few hundred held-out files routinely
produces single-digit discordant counts, which is where the approximation is
worst.

WHICH VARIANT THIS REPORTS, AND WHY NOT THE EXACT ONE. Both are reported.
The mid-p value is the one to read; the exact conditional value is kept
beside it because it is the guaranteed-level reference the mid-p value is
derived from. Fagerland, Lydersen and Laake measured type I error and power
over 9,595 scenarios and found the exact conditional test overly conservative
in all of them, while the mid-p test never violated the nominal level and was
almost as powerful as the asymptotic test. On a table of 9 items fixed
against 2 broken, exact returns 0.065 and mid-p returns 0.039: the exact test
declines to call an improvement the data support.

WHERE THE ARITHMETIC LIVES, AND WHY IT NO LONGER LIVES HERE. This module
carried its own binomial tail and point probability until 2026-09-09.
:mod:`platform_core.power_distributions` now ships the same two tests for the
whole monorepo, written independently for the power audit, and a second
implementation of a published statistic is a second thing that can drift.
Before removing them the two were compared over EVERY split of every
discordant count up to 60 -- 1,922 tables, both arm orientations -- and agreed
to the bit on both tests, including the tie form (0.84375 at 3:3, which a
version that doubles the tail gets wrong) and the 4:3 table this package
published as 0.7265625. What remains here is :func:`discordant_split`, which
is the part that is genuinely this package's: the projection from a 2x2
guard-pass table onto the two numbers a conditional test takes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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


def discordant_split(counts: PairedCounts) -> tuple[int, int]:
    """Project the 2x2 table onto the two numbers a conditional test takes.

    McNemar conditions on the discordant pairs alone, so the concordant cells
    carry no information about a difference and are dropped here rather than
    inside the test. The MINORITY cell is returned rather than the candidate
    cell: the p-value is two-sided and therefore symmetric under a swap of the
    arms, and naming the smaller cell makes that symmetry a property of the
    projection instead of something the test has to restore.

    Args:
        counts: The 2x2 table.

    Returns:
        The minority discordant count and the total discordant count, in that
        order -- the argument order
        :func:`platform_core.power_distributions.mcnemar_p` takes.
    """
    baseline_only = counts["baseline_only"]
    candidate_only = counts["candidate_only"]
    return min(baseline_only, candidate_only), baseline_only + candidate_only


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
    "discordant_split",
    "net_improvement",
    "paired_counts",
    "pass_rate",
    "per_checker_rates",
]
