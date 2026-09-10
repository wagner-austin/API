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


__all__ = [
    "require_effect_of_interest",
    "require_rate_floor",
]
