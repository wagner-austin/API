"""Phase 1c: project ``TankStateDict`` as a ``Fact[T]``.

``TankStateDict`` carries the full fact metadata flat (``source``, the
four freshness timestamps, plus the Phase 1c ``confidence`` and
``provenance`` fields); :func:`tank_fact` projects it into a true
:class:`~tankpit_bot.facts.fact.Fact` for the layers that consume
Facts. Same flat-carry deviation as
:mod:`tankpit_bot.facts.container_facts`, same rationale.

The projection's ``observed_ms`` is the tank's ``timestamp_ms`` (the
any-source observation gate); consumers needing the finer gates
(wire-seen, position, viewport) read them off the flat dict -- they
are freshness *gates*, not part of the believed value.

This module is intentionally NOT re-exported from
``tankpit_bot.facts`` to keep the import graph acyclic.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from tankpit_bot.facts.fact import Fact, make_fact
from tankpit_bot.state.types.constants import TankLiveness
from tankpit_bot.state.types.tank import TankStateDict


class TankValueDict(TypedDict):
    """The believed value of a tank fact (metadata stripped).

    Attributes:
        tank_id: Unique identifier for this tank.
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        team: Team ID (0-3).
        rank: Military rank (0-7).
        damage_state: Health state (0-3).
        direction: Sprite direction byte.
        name: Player name.
        is_bot: Whether this is a bot player.
        is_self: Whether this is the player's own tank.
        liveness: Lifecycle state (alive / deactivated).
    """

    tank_id: int
    x: int
    y: int
    team: int
    rank: int
    damage_state: int
    direction: int
    name: str
    is_bot: bool
    is_self: bool
    liveness: TankLiveness


def tank_fact(state: TankStateDict) -> Fact[TankValueDict]:
    """Project a tank state into a Fact.

    Args:
        state: Tank state carrying flat fact metadata.

    Returns:
        Fact whose value is the tank's believed identity and kinematics.

    Raises:
        NoUnsourcedFactError: If the state's metadata is incomplete.
        ConfidenceOutOfBoundsError: If confidence is out of range.
        ProvenanceRootednessError: If the provenance is not rooted.
    """
    return make_fact(
        value=TankValueDict(
            tank_id=state["tank_id"],
            x=state["x"],
            y=state["y"],
            team=state["team"],
            rank=state["rank"],
            damage_state=state["damage_state"],
            direction=state["direction"],
            name=state["name"],
            is_bot=state["is_bot"],
            is_self=state["is_self"],
            liveness=state["liveness"],
        ),
        source=state["provenance"]["origin"],
        observed_ms=state["timestamp_ms"],
        confidence=state["confidence"],
        provenance=state["provenance"],
    )


__all__ = [
    "TankValueDict",
    "tank_fact",
]
