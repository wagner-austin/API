"""The refusals every power instrument shares.

Split from :mod:`platform_core.minimum_detectable_effect` on 2026-09-09 when
that module passed the 600-line cap, along the layer boundary this package
already uses: ``power_distributions`` holds the mathematics, ``power_types``
holds the shapes, this module holds the refusals, the instruments module
computes, and ``power_records`` serialises.

BOTH REFUSE RATHER THAN CLAMP, for the same reason. A clamped parameter
silently changes what a published number MEANS, and these numbers are read
back months later beside nulls they were meant to qualify.
"""

from __future__ import annotations

from platform_core.error_codes import StatisticalPowerErrorCode
from platform_core.errors import AppError


def require_effect_of_interest(value: float, field: str) -> None:
    """Reject a non-positive smallest-effect-of-interest.

    Args:
        value: Candidate effect size.
        field: Field name, for the message.

    Raises:
        AppError: ``POWER_EFFECT_OF_INTEREST_INVALID`` when not positive.
    """
    if not value > 0.0:
        raise AppError(
            StatisticalPowerErrorCode.POWER_EFFECT_OF_INTEREST_INVALID,
            f"{field} must be positive to make a verdict meaningful; got {value!r}",
        )


def require_rate_floor(floor: float) -> None:
    """Reject a pass-rate floor outside the open unit interval.

    Args:
        floor: Candidate floor.

    Raises:
        AppError: ``POWER_RATE_FLOOR_OUT_OF_RANGE`` when not in ``(0, 1)``.
    """
    if not 0.0 < floor < 1.0:
        raise AppError(
            StatisticalPowerErrorCode.POWER_RATE_FLOOR_OUT_OF_RANGE,
            f"a pass-rate floor must lie in (0, 1) to be beatable; got {floor!r}",
        )


def require_target_power(target_power: float) -> None:
    """Reject a target power outside ``(0, 1)``.

    Its own check rather than reuse of :func:`require_rate_floor` or the
    Clopper-Pearson confidence one: a target POWER is the chance of detecting
    a real effect and a CONFIDENCE is how sure an interval is, so a caller
    passing ``0.95`` meaning either would get no error from a shared
    validator.

    Args:
        target_power: The power the design must reach.

    Raises:
        AppError: ``POWER_TARGET_POWER_OUT_OF_RANGE`` when outside ``(0, 1)``.
            Both endpoints are excluded: power 0 is reached by every design
            and asks nothing, and power 1 is unreachable by any finite design.
    """
    if not 0.0 < target_power < 1.0:
        raise AppError(
            StatisticalPowerErrorCode.POWER_TARGET_POWER_OUT_OF_RANGE,
            f"target power must lie strictly inside (0, 1); got {target_power!r}. "
            "0 is met by every design and 1 by none, so neither states a design goal.",
        )


def require_discordant_rate(discordant_rate: float) -> None:
    """Reject a discordant rate outside ``(0, 1]``.

    THE OPEN LOWER BOUND IS THE POINT. McNemar conditions on the pairs that
    disagree, so a design expecting none of them to disagree has no test to
    size: every sample size gives power equal to zero and the search would
    run to its ceiling before refusing. Refused at the door instead, because
    "your discordant rate is zero" is the actionable sentence and "no sample
    size reaches 80% power" is not.

    ``1.0`` is legal and is not a degenerate case: it says every pair
    disagrees, which is the regime a well-separated comparison lands in.

    Args:
        discordant_rate: Expected share of pairs that disagree.

    Raises:
        AppError: ``POWER_DISCORDANT_RATE_OUT_OF_RANGE`` when outside
            ``(0, 1]``.
    """
    if not 0.0 < discordant_rate <= 1.0:
        raise AppError(
            StatisticalPowerErrorCode.POWER_DISCORDANT_RATE_OUT_OF_RANGE,
            f"the discordant rate must lie in (0, 1]; got {discordant_rate!r}. "
            "At zero no pair disagrees, so there is no McNemar test to size.",
        )


def require_design_split(split: float) -> None:
    """Reject a design split outside ``(1/2, 1]``.

    NOT THE SAME RANGE AS AN OBSERVED SPLIT, deliberately. An observed split
    may legally sit anywhere in ``[0, 1]``, including at or below a half; a
    DESIGN split is the effect the study is built to catch, and one at
    exactly a half is the null. Sizing against the null is not a small
    request that needs a large sample, it is a request with no answer, and
    saying so at the door beats a refusal from the far end of the search that
    reads as "you need more items".

    Below a half is refused rather than mirrored. The two arms are named, so
    a split of 0.3 states that the CANDIDATE loses; mirroring it to 0.7 would
    silently answer a different question from the one asked.

    Args:
        split: Probability that a discordant pair falls to the candidate.

    Raises:
        AppError: ``POWER_DESIGN_SPLIT_OUT_OF_RANGE`` when outside
            ``(1/2, 1]``.
    """
    if not 0.5 < split <= 1.0:
        raise AppError(
            StatisticalPowerErrorCode.POWER_DESIGN_SPLIT_OUT_OF_RANGE,
            f"a design split must lie in (0.5, 1]; got {split!r}. "
            "Exactly 0.5 is the null, which no sample size can reject at better "
            "than alpha, and below 0.5 states that the candidate is the worse arm "
            "-- name the arms the other way round rather than mirroring it here.",
        )


__all__ = [
    "require_design_split",
    "require_discordant_rate",
    "require_effect_of_interest",
    "require_rate_floor",
    "require_target_power",
]
