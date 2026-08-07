"""Container predicates for the AI system.

Pure functions that evaluate whether containers are pursuable and
whether locked targets should be released. Search functions that find
specific containers live in ``equipment_search.py``. Per-tile scan
coverage lives in :mod:`tankpit_bot.state.scan_coverage`.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.bot.ai.threat_primitives import manhattan_distance
from tankpit_bot.state.types import (
    ContainerStateDict,
    MineStateDict,
    SelfStateDict,
    WorldStateDict,
)
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

log = get_logger(__name__)

_LOCK_RELEASE_MIN_GAP = 10


def is_lock_release_warranted(
    self_state: SelfStateDict,
    locked_x: int,
    locked_y: int,
    candidate_x: int,
    candidate_y: int,
) -> bool:
    """Return True when a candidate is enough closer to drop a locked target.

    Args:
        self_state: Player state for the distance origin.
        locked_x: Locked target X coordinate.
        locked_y: Locked target Y coordinate.
        candidate_x: Fresh candidate X coordinate.
        candidate_y: Fresh candidate Y coordinate.

    Returns:
        True when the candidate is at most half the locked distance and
        at least ``_LOCK_RELEASE_MIN_GAP`` tiles closer.
    """
    sx, sy = self_state["x"], self_state["y"]
    locked_dist = manhattan_distance(sx, sy, locked_x, locked_y)
    candidate_dist = manhattan_distance(sx, sy, candidate_x, candidate_y)
    if candidate_dist * 2 > locked_dist:
        return False
    return locked_dist - candidate_dist >= _LOCK_RELEASE_MIN_GAP


_FUEL_LOCK_VALUE_HYSTERESIS = 2
"""A fuel candidate must be worth at least this multiple of the locked
target's deliverable score to break the lock on value alone.

Prevents lock ping-pong between comparable containers while still
releasing dreg locks: in the flag-13 incident
([[flag-triage-20260729]], run bot-20260729-232252 tick 1055) an
84-fuel remnant held its lock at deficit 207 against a 462-volume
container five tiles away because the release rule was distance-only —
deliverable scores 75 vs 202, ratio 2.7, released under this rule.
"""


def fuel_deliverable_score(
    self_state: SelfStateDict,
    container: ContainerStateDict,
    deficit: int,
) -> int:
    """Return a fuel container's deliverable value net of travel.

    The server clamps a fuel pickup to ``min(volume, headroom)``, so
    volume beyond the current deficit is worthless this trip; distance
    then discounts the trip itself (the same ``volume - distance``
    shape as :func:`~tankpit_bot.bot.ai.equipment_search.find_best_fuel`
    scoring, with the deliverable clamp applied).

    Args:
        self_state: Player state for the distance origin.
        container: Fuel container to score.
        deficit: Current fuel deficit (``capacity - fuel``).

    Returns:
        ``min(volume, deficit) - manhattan_distance(self, container)``.
    """
    deliverable = min(container["volume"], max(deficit, 0))
    return deliverable - manhattan_distance(
        self_state["x"], self_state["y"], container["x"], container["y"]
    )


def is_fuel_lock_release_warranted(
    self_state: SelfStateDict,
    locked_target: ContainerStateDict,
    candidate: ContainerStateDict,
    deficit: int,
) -> bool:
    """Return True when a fuel candidate justifies dropping the locked target.

    Two independent release paths:

    1. The distance rule shared with equipment locks
       (:func:`is_lock_release_warranted` — markedly closer).
    2. The value rule (flag-13 fix): the candidate's deliverable score
       must be at least :data:`_FUEL_LOCK_VALUE_HYSTERESIS` times the
       locked target's (floored at 1, so a non-positive locked score —
       a dreg not worth its own walk — releases for any candidate with
       positive deliverable value).

    Args:
        self_state: Player state for the distance origin.
        locked_target: Currently locked fuel container.
        candidate: Best fresh fuel candidate.
        deficit: Current fuel deficit (``capacity - fuel``).

    Returns:
        True when either release path fires.
    """
    if is_lock_release_warranted(
        self_state,
        locked_target["x"],
        locked_target["y"],
        candidate["x"],
        candidate["y"],
    ):
        return True
    locked_score = fuel_deliverable_score(self_state, locked_target, deficit)
    candidate_score = fuel_deliverable_score(self_state, candidate, deficit)
    return candidate_score >= _FUEL_LOCK_VALUE_HYSTERESIS * max(locked_score, 1)


def is_container_pursuable(
    container: ContainerStateDict,
    *,
    want_fuel: bool,
) -> bool:
    """Return True when a tracked container is worth pursuing at all.

    This is the SINGLE definition of pursuability: candidate selection,
    opportunistic pickups, and lock continuation must all apply it.

    The historical 30 s freshness TTL was removed 2026-07-06. Every
    pursuability consumer is viewport-scoped, and an in-viewport
    container is wire-truthful under the truth layer: the landing 0x5A
    sweep removes silent visible entries, the landing radar's
    omission-prune covers radar-sourced ones, and live 0x43 cache
    updates track consumption while the bot watches. The TTL therefore
    only ever expired REAL loot -- live run 2026-07-06 18:20:55 dropped
    an equipment container revealed 31 s earlier and cascaded into a
    bogus ``out_of_fuel`` session exit at fuel 1100.

    Args:
        container: Tracked container to check.
        want_fuel: True to require fuel, False to require equipment.

    Returns:
        True when the container matches the kind and has no failed
        pickup.
    """
    if container["is_fuel"] != want_fuel:
        return False
    return container["failed_pickups"] == 0


def _viewport_bounds(world: WorldStateDict) -> tuple[int, int, int, int]:
    """Return inclusive observable viewport bounds from world state."""
    return viewport_visible_bounds(world["viewport"])


def hostile_mines(world: WorldStateDict) -> dict[str, MineStateDict]:
    """Return tracked mines that would damage the bot if detonated.

    Tankpit mines do not damage tanks on the placer's team -- friendly
    mines are passable. Every blocking / pathing check should query
    this filtered view instead of ``world["mines"]`` directly so the
    bot doesn't treat its own team's defensive layout as obstacles.

    If ``self_state`` is not yet known, every mine is treated as
    hostile (defensive default).

    Args:
        world: Current world state.

    Returns:
        Mines indexed by ``"x,y"`` key whose team is different from the
        bot's team, or every tracked mine when ``self_state`` is None.
    """
    self_state = world["self_state"]
    if self_state is None:
        return world["mines"]
    self_team = self_state["team"]
    return {key: mine for key, mine in world["mines"].items() if mine["team"] != self_team}


__all__ = [
    "_viewport_bounds",
    "fuel_deliverable_score",
    "hostile_mines",
    "is_container_pursuable",
    "is_fuel_lock_release_warranted",
    "is_lock_release_warranted",
]
