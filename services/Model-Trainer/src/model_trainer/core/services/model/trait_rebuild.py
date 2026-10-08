"""Rebuilding trait arms from the rows their cells wrote.

THE INVERSE OF :mod:`~model_trainer.core.services.model.trait_arms`'S
EMITTERS, beside nothing else, because a sweep that checkpoints its seeds in
blocks never holds an arm over all of them: it holds rows, one block's at a
time. Every field of an arm is a function of its per-seed gains, so an arm
rebuilt from the rows of every block is the arm a straight run would have
measured, and the record is emitted from the rebuilt arms through the same
functions a straight run uses.

The arm names are spelled here exactly as
:func:`~model_trainer.core.services.model.trait_arms.trait_arm` and
:func:`~model_trainer.core.services.model.trait_arms.measure_trait_composition`
write them; a rebuild that missed one would fail on the absent rows rather
than report a different arm.
"""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.run_record import Observation

from model_trainer.core.contracts.replicated_measurement import replicated_from_observations
from model_trainer.core.services.model.trait_arms import TraitArm, TraitCompositionArms


def rebuild_trait_arm(rows: Sequence[Observation], *, name: str, seeds: Sequence[int]) -> TraitArm:
    """Rebuild one arm's three readings over the given seeds.

    Args:
        rows: Rows holding the arm's per-seed gains, among others.
        name: The arm as it was measured, e.g. ``"bullets-solo"``.
        seeds: Every seed the arm was measured under, in order.

    Returns:
        The arm, identical to the one that wrote the rows.

    Raises:
        AppError: With ``CARTRIDGE_ARM_ROWS_INCOMPLETE`` when a seed's row is
            absent for any reading.
    """
    return TraitArm(
        expression=replicated_from_observations(rows, arm=f"{name}-expression", seeds=seeds),
        coherence=replicated_from_observations(rows, arm=f"{name}-coherence", seeds=seeds),
        style=replicated_from_observations(rows, arm=f"{name}-style", seeds=seeds),
    )


def rebuild_trait_cell(
    rows: Sequence[Observation], *, arm: str, partners: int, seeds: Sequence[int]
) -> TraitCompositionArms:
    """Rebuild one composed cell's arms over the given seeds.

    Args:
        rows: Rows holding every arm of the cell, among others.
        arm: The cell's name, e.g. ``"bullets-n4"``.
        partners: How many traits were composed in front of the primary,
            which is how many cross arms the cell carries.
        seeds: Every seed the cell was measured under, in order.

    Returns:
        The cell's arms, identical to the ones that wrote the rows.

    Raises:
        AppError: With ``CARTRIDGE_ARM_ROWS_INCOMPLETE`` when any arm's
            per-seed row is absent.
    """
    return TraitCompositionArms(
        alone=rebuild_trait_arm(rows, name=f"{arm}-alone", seeds=seeds),
        composed=rebuild_trait_arm(rows, name=f"{arm}-composed", seeds=seeds),
        untrained_composed=rebuild_trait_arm(rows, name=f"{arm}-untrained-composed", seeds=seeds),
        cross=tuple(
            rebuild_trait_arm(rows, name=f"{arm}-cross-{position}", seeds=seeds)
            for position in range(partners)
        ),
    )


__all__ = ["rebuild_trait_arm", "rebuild_trait_cell"]
