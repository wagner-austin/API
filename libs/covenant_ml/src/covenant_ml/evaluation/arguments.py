"""Turning the evaluation entry points' string flags into typed settings.

The harness parses every flag as an int, float, string or path, so the
structured ones arrive as strings and are parsed here, where they are
testable without running a benchmark. Every refusal names the flag.
"""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.members import find_member

from ..benchmarking.protocols import TrainerProto
from .intervals import require_confidence
from .learning_curve import require_fractions
from .types import (
    ERR_INVALID_ARGUMENT,
    ERR_INVALID_RESAMPLES,
    AblationGroup,
    ResamplingSettings,
    ResamplingUnit,
)


def parse_unit(raw: str) -> ResamplingUnit:
    """Read ``--unit``.

    Args:
        raw: The flag's value.

    Returns:
        The resampling unit.

    Raises:
        ValueError: If the value names no unit.
    """
    unit = find_member(raw, ResamplingUnit)
    if unit is None:
        admitted = ", ".join(member.value for member in ResamplingUnit)
        raise ValueError(f"[{ERR_INVALID_ARGUMENT}] --unit '{raw}' must be one of {admitted}")
    return unit


def resampling_settings(
    unit: ResamplingUnit,
    block_length: int,
    n_resamples: int,
    resample_seed: int,
    confidence: float,
) -> ResamplingSettings:
    """Validate and bind the resampling flags.

    The block length is checked against the unit when the held-out fold is
    known (:func:`~covenant_ml.evaluation.resampling.build_resampling`); the
    two checks that need no data are made here, before any model is fitted.

    Args:
        unit: What one independent observation is.
        block_length: Rows per block for ``block``, else 1.
        n_resamples: Bootstrap resamples, at least 2.
        resample_seed: Seed of the bootstrap draws.
        confidence: Coverage of every interval.

    Returns:
        The settings.

    Raises:
        ValueError: If ``confidence`` is outside ``(0, 1)`` or fewer than two
            resamples are requested.
    """
    require_confidence(confidence)
    if n_resamples < 2:
        raise ValueError(
            f"[{ERR_INVALID_RESAMPLES}] --n-resamples must be at least 2, got {n_resamples}"
        )
    return {
        "unit": unit,
        "block_length": block_length,
        "n_resamples": n_resamples,
        "resample_seed": resample_seed,
        "confidence": confidence,
    }


def parse_groups(spec: str) -> list[AblationGroup]:
    """Read ``--groups``: groups separated by ``;``, features by ``,``.

    A group is ``name=feature,feature`` or a bare feature name, which is a
    group of one named for the feature: ``X1;liquidity=X3,X4``.

    Args:
        spec: The flag's value.

    Returns:
        The groups, in the order given.

    Raises:
        ValueError: If a group is empty or names an empty feature.
    """
    groups: list[AblationGroup] = []
    for part in spec.split(";"):
        name, separator, members = part.partition("=")
        features = [feature.strip() for feature in (members if separator else name).split(",")]
        label = name.strip()
        if not label or any(not feature for feature in features):
            raise ValueError(
                f"[{ERR_INVALID_ARGUMENT}] --groups part '{part}' must be 'name=f1,f2' "
                f"or a feature name"
            )
        groups.append({"name": label, "features": features})
    return groups


def parse_fractions(spec: str) -> list[float]:
    """Read ``--fractions``: training shares separated by ``,``.

    Args:
        spec: The flag's value, e.g. ``"0.1,0.25,0.5,1.0"``.

    Returns:
        The shares, validated as strictly ascending in ``(0, 1]``.

    Raises:
        ValueError: If a share is not a number, or the shares are invalid.
    """
    fractions: list[float] = []
    for part in spec.split(","):
        text = part.strip()
        digits = text.replace(".", "", 1)
        if not digits.isdigit():
            raise ValueError(
                f"[{ERR_INVALID_ARGUMENT}] --fractions part '{part}' is not a decimal number"
            )
        fractions.append(float(text))
    require_fractions(fractions)
    return fractions


def select_trainer(trainers: Sequence[TrainerProto], name: str) -> TrainerProto:
    """Pick the arm ``--model`` names.

    Args:
        trainers: The available arms.
        name: The flag's value.

    Returns:
        The arm whose model name is ``name``.

    Raises:
        ValueError: If no arm carries that name.
    """
    for trainer in trainers:
        if trainer.model_name.value == name:
            return trainer
    available = ", ".join(trainer.model_name.value for trainer in trainers)
    raise ValueError(f"[{ERR_INVALID_ARGUMENT}] --model '{name}' must be one of {available}")


__all__ = [
    "parse_fractions",
    "parse_groups",
    "parse_unit",
    "resampling_settings",
    "select_trainer",
]
