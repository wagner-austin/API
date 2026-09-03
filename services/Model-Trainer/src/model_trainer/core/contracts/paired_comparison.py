"""Two arms scored on the SAME items, compared item by item.

WHY PAIRED AND NOT TWO MEANS. Two arms can report the same mean loss and
disagree about every item under it. An aggregate cannot tell those apart, and
the difference is the whole question when the claim is "this intervention
helped": a treatment that improves half the items and ruins the other half
reports the same mean as one that changed nothing, and only the pairing
distinguishes them.

The items are the pairing. Both arms score the SAME tokens in the SAME order,
so item ``n`` in one arm and item ``n`` in the other are the same text, and
their difference is attributable to the arm and to nothing else.

WHAT THE TEST IS. Every item is one of three things: the treatment did better,
the treatment did worse, or they tied. Ties carry no information about
direction, so the test conditions on the DISCORDANT pairs and asks whether
their split could plausibly be a coin flip. That is McNemar's exact conditional
test, and its exactness is the reason to prefer it here: the discordant counts
in a small held-out set are small, and a chi-square approximation is not
trustworthy at those counts.

It is also CONSERVATIVE, and that is stated rather than hidden. Conditioning on
the discordant total makes the attainable p-values discrete, so the realised
type-I error sits below the nominal level rather than at it -- the test rejects
less often than its alpha advertises. For this use that is the right direction
to err: the claim being tested is "the intervention helped", and a test that
under-rejects will fail to credit a real improvement before it credits a false
one.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_float,
    require_int,
    require_str,
)
from typing_extensions import TypedDict


class PairedItemOutcome(TypedDict):
    """One item, scored under both arms.

    Attributes:
        index: Position of the item in the held-out set. Carried so an outcome
            can be traced back to the text that produced it.
        baseline: The item's loss under the control arm.
        treatment: The item's loss under the arm being tested.
    """

    index: int
    baseline: float
    treatment: float


class PairedComparison(TypedDict):
    """What two arms did across a held-out set.

    Attributes:
        items: How many items were scored under both arms.
        mean_baseline: Mean loss under the control arm.
        mean_treatment: Mean loss under the arm being tested.
        improved: Items where the treatment scored a LOWER loss.
        worsened: Items where the treatment scored a higher loss.
        tied: Items scoring identically. Excluded from the test, because a tie
            says nothing about direction.
        p_value: McNemar's exact conditional two-sided p-value over the
            discordant pairs. One when there are none, which is the honest
            answer to "could this split be chance" when there is no split.
        outcomes_digest: Digest of WHICH items improved, in order. Two runs
            can agree on every count here and disagree about which items they
            agreed on; this is what distinguishes them.
    """

    items: int
    mean_baseline: float
    mean_treatment: float
    improved: int
    worsened: int
    tied: int
    p_value: float
    outcomes_digest: str


def outcomes_digest(outcomes: Sequence[PairedItemOutcome]) -> str:
    """Digest WHICH items the treatment improved, and nothing else.

    Deliberately excludes the losses, for the reason the cloze digest excludes
    its scores: two runs of the same comparison on different hardware agree on
    the direction of every item while differing in the last bits of every
    float, so digesting the numbers would report a difference on every
    comparison and mean nothing.

    Args:
        outcomes: Per-item outcomes, in item order.

    Returns:
        Hex digest over the per-item directions.
    """
    directions = "".join(
        "i"
        if outcome["treatment"] < outcome["baseline"]
        else "w"
        if outcome["treatment"] > outcome["baseline"]
        else "t"
        for outcome in outcomes
    )
    return hashlib.sha256(directions.encode("utf-8")).hexdigest()


def exact_mcnemar_p(*, improved: int, worsened: int) -> float:
    """Two-sided exact conditional p-value over discordant pairs.

    Conditions on the discordant total and tests the split against a fair
    coin. Computed with exact integer binomial coefficients rather than a
    normal or chi-square approximation, because the discordant counts on a
    held-out set of tens of items are exactly where those approximations are
    least trustworthy.

    Args:
        improved: Items where the treatment scored lower.
        worsened: Items where the treatment scored higher.

    Returns:
        The p-value, in ``[0.0, 1.0]``. Exactly one when there are no
        discordant pairs: with no evidence of direction there is nothing to
        reject.
    """
    discordant = improved + worsened
    if discordant == 0:
        return 1.0
    extreme = min(improved, worsened)
    tail = sum(math.comb(discordant, k) for k in range(extreme + 1))
    total: int = 1 << discordant
    two_sided = 2.0 * float(tail) / float(total)
    return min(1.0, two_sided)


def summarise_pairs(outcomes: Sequence[PairedItemOutcome]) -> PairedComparison:
    """Reduce per-item outcomes to the comparison they support.

    Args:
        outcomes: Per-item outcomes, in item order.

    Returns:
        The comparison. An empty set reports zero items, zero means and a
        p-value of one, which is what "nothing was measured" should look like
        rather than a division error.
    """
    improved = sum(1 for o in outcomes if o["treatment"] < o["baseline"])
    worsened = sum(1 for o in outcomes if o["treatment"] > o["baseline"])
    count = len(outcomes)
    return PairedComparison(
        items=count,
        mean_baseline=(sum(o["baseline"] for o in outcomes) / count) if count else 0.0,
        mean_treatment=(sum(o["treatment"] for o in outcomes) / count) if count else 0.0,
        improved=improved,
        worsened=worsened,
        tied=count - improved - worsened,
        p_value=exact_mcnemar_p(improved=improved, worsened=worsened),
        outcomes_digest=outcomes_digest(outcomes),
    )


def encode_paired_item_outcome(outcome: PairedItemOutcome) -> JSONObject:
    """Encode one per-item outcome.

    Args:
        outcome: The outcome to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "index": outcome["index"],
        "baseline": outcome["baseline"],
        "treatment": outcome["treatment"],
    }


def decode_paired_item_outcome(value: JSONValue) -> PairedItemOutcome:
    """Decode and validate one per-item outcome.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated outcome.

    Raises:
        JSONTypeError: If the value is not an object, or a field is missing or
            mistyped.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"paired outcome must be a JSON object, got {type(value).__name__}")
    return PairedItemOutcome(
        index=require_int(value, "index"),
        baseline=require_float(value, "baseline"),
        treatment=require_float(value, "treatment"),
    )


def encode_paired_comparison(comparison: PairedComparison) -> JSONObject:
    """Encode a comparison.

    Args:
        comparison: The comparison to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "items": comparison["items"],
        "mean_baseline": comparison["mean_baseline"],
        "mean_treatment": comparison["mean_treatment"],
        "improved": comparison["improved"],
        "worsened": comparison["worsened"],
        "tied": comparison["tied"],
        "p_value": comparison["p_value"],
        "outcomes_digest": comparison["outcomes_digest"],
    }


def decode_paired_comparison(value: JSONValue) -> PairedComparison:
    """Decode and validate a comparison.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated comparison.

    Raises:
        JSONTypeError: If the value is not an object, or a field is missing or
            mistyped.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"paired comparison must be a JSON object, got {type(value).__name__}")
    return PairedComparison(
        items=require_int(value, "items"),
        mean_baseline=require_float(value, "mean_baseline"),
        mean_treatment=require_float(value, "mean_treatment"),
        improved=require_int(value, "improved"),
        worsened=require_int(value, "worsened"),
        tied=require_int(value, "tied"),
        p_value=require_float(value, "p_value"),
        outcomes_digest=require_str(value, "outcomes_digest"),
    )


def decode_paired_item_outcomes(value: JSONValue) -> list[PairedItemOutcome]:
    """Decode a list of per-item outcomes.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated outcomes, in the order read.

    Raises:
        JSONTypeError: If the value is not a list, or any entry is invalid.
    """
    if not isinstance(value, list):
        raise JSONTypeError(f"paired outcomes must be a JSON array, got {type(value).__name__}")
    return [decode_paired_item_outcome(entry) for entry in value]


__all__ = [
    "PairedComparison",
    "PairedItemOutcome",
    "decode_paired_comparison",
    "decode_paired_item_outcome",
    "decode_paired_item_outcomes",
    "encode_paired_comparison",
    "encode_paired_item_outcome",
    "exact_mcnemar_p",
    "outcomes_digest",
    "summarise_pairs",
]
