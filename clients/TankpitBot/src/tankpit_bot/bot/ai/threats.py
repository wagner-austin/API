"""Threat analysis: the per-tick viewport scan and locked-target pursuit.

Turns the tank registry into the tick's threat list, and follows a
target that has stopped broadcasting. Target ACQUISITION -- choosing a
new one -- is :mod:`tankpit_bot.bot.ai.threat_acquisition`; the shared
vocabulary is :mod:`tankpit_bot.bot.ai.threat_primitives`.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.humans import (
    DEFAULT_HUMAN_MAX_RANK,
    DEFAULT_HUMAN_MIN_RANK,
    is_human_rank_protected,
)
from tankpit_bot.bot.ai.threat_primitives import (
    _is_enemy,
    _threat_sort_key,
    human_combat_consented,
    manhattan_distance,
)
from tankpit_bot.bot.ai.world_types import (
    EnemyThreatDict,
    make_enemy_threat,
)
from tankpit_bot.protocol.naming import is_human_name
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import (
    VIEWPORT_PRESENCE_TTL_MS,
    SelfStateDict,
    WorldStateDict,
    has_known_position,
)


def analyze_threats(
    ws: WorldService,
    world: WorldStateDict,
    self_state: SelfStateDict,
    now_ms: int,
    *,
    human_min_rank: int = DEFAULT_HUMAN_MIN_RANK,
    human_max_rank: int = DEFAULT_HUMAN_MAX_RANK,
) -> list[EnemyThreatDict]:
    """Analyze all enemy tanks and return sorted threat list.

    Filters to enemy tanks only (different team, not self) confirmed
    alive and observed via viewport-bound wire within
    :data:`VIEWPORT_PRESENCE_TTL_MS` -- the viewport observation gate
    proves the position is current and the tank is actually visible to
    the bot. Computes Manhattan distance from the player and sorts by
    distance ascending.

    Args:
        world: Current world state with tank positions.
        self_state: Player's own state for position and team.
        now_ms: Current tick timestamp for viewport-freshness filtering.

    Returns:
        List of EnemyThreatDict sorted by distance ascending.
    """
    self_x = self_state["x"]
    self_y = self_state["y"]
    self_team = self_state["team"]

    threats: list[EnemyThreatDict] = []
    for tank in world["tanks"].values():
        if not _is_enemy(tank, self_team):
            continue
        # Liveness gate. ``deactivated`` is the corpse window after a
        # kill (empirical ~22 s, 2026-06-20 capture). 0x58 TankRemove
        # is NOT a death signal -- it's tracking removal -- so there
        # is no separate ``removed`` state; removed tanks are simply
        # deleted from the registry and re-added by the next MapData
        # or per-tank wire. The previous explicit ``direction >= 32``
        # corpse-sprite check was replaced by this single liveness
        # filter -- ``apply_tank_observation`` now routes both
        # corpse-direction wire arrivals and 0x41 Deactivation to
        # ``liveness == "deactivated"``.
        if tank["liveness"] != "alive":
            continue
        # Position-confirmation gate: the login roster dump (0x21
        # TankInfo, full map, no coordinates) leaves every tank at the
        # construction default until its first position-bearing wire.
        # Acquiring one would fire at tile (0, 0). The predicate is
        # canonical in ``state/types/tank.py`` -- the guard bans the
        # inline (0, 0) comparison this used to be.
        if not has_known_position(tank):
            continue
        # Viewport-observation gate. ``timestamp_ms`` and
        # ``last_wire_seen_ms`` are both refreshed by global
        # broadcasts (0x4C MapData, 0x2E TankStatusSync) that fire
        # for every alive tank on the map -- they cannot
        # distinguish "in viewport" from "anywhere on the map".
        # ``last_viewport_observation_ms`` only advances when the
        # dispatch layer routes the observation through
        # ``storage_source == "viewport"`` -- the actual viewport-
        # bound proof.
        if now_ms - tank["last_viewport_observation_ms"] > VIEWPORT_PRESENCE_TTL_MS:
            continue
        # Human rank-window protection (user ruling 2026-07-28):
        # humans outside [human_min_rank, human_max_rank] never enter
        # the threat list, so they can be neither locked nor fired at.
        # Practice bots are farmed at any rank.
        if is_human_rank_protected(
            tank["name"],
            tank["rank"],
            min_rank=human_min_rank,
            max_rank=human_max_rank,
        ):
            continue
        # Human-consent contract (2026-07-30): a human who has neither
        # chatted nor shot us never enters the threat list -- no lock,
        # no fire. An attacker consents by attacking (their id lands
        # in the damage book), so defense is never blocked.
        if is_human_name(tank["name"]) and not human_combat_consented(ws, tank["tank_id"]):
            continue
        dist = manhattan_distance(self_x, self_y, tank["x"], tank["y"])
        threats.append(
            make_enemy_threat(
                tank_id=tank["tank_id"],
                x=tank["x"],
                y=tank["y"],
                distance=dist,
                damage_state=tank["damage_state"],
                rank=tank["rank"],
                team=tank["team"],
                name=tank["name"],
                is_bot=tank["is_bot"],
                timestamp_ms=tank["timestamp_ms"],
                last_wire_seen_ms=tank["last_wire_seen_ms"],
                last_position_update_ms=tank["last_position_update_ms"],
                last_aim_x=tank["last_aim_x"],
                last_aim_y=tank["last_aim_y"],
                last_aim_weapon=tank["last_aim_weapon"],
                last_aim_ms=tank["last_aim_ms"],
            )
        )

    threats.sort(key=_threat_sort_key)
    return threats


def find_locked_target_pursuit(
    world: WorldStateDict,
    self_state: SelfStateDict,
    locked_target_id: int,
    killed: dict[str, int],
) -> EnemyThreatDict | None:
    """Build a pursuit threat for a locked target that left the viewport.

    Distinguishes "alive but moved" from "actually dead." When
    ``analyze_threats`` no longer returns the locked target (because
    they left the viewport and the strict viewport-presence gate
    filters them out), this function checks ``world["tanks"]`` for
    the same ``locked_target_id``. If the tank is still alive it
    synthesises an :class:`EnemyThreatDict` at the cached coords so
    HUNT can keep firing homing shots toward the last known position.
    ``SCAN_ON_LANDING`` handles viewport confirmation on arrival,
    so this never re-introduces the phantom-firing bug -- firing
    still requires a strict ``analyze_threats`` hit on the next
    tick when the target is back in viewport.

    Pursuit stops only on authoritative death signals: the locked
    id landing in ``killed`` (kill cooldown applied) or the tank's
    ``liveness`` flipping to ``"deactivated"`` via 0x41. The earlier
    freshness-window gate was removed 2026-06-22 because it tripped
    when the server stopped broadcasting 0x2E TankStatusSync for a
    far-away tank, ending pursuit prematurely (live capture
    18:17:00: 3 pursuit homings hit purple-4 then gate tripped on
    timestamp staleness even though purple-4 was very much alive).
    Ammo on dual / homing shots only decrements on confirmed hit,
    so a target that's truly unreachable burns zero ammo -- there's
    no over-fire cost for indefinite pursuit.

    Args:
        world: Filtered world state (killed tanks already removed).
        self_state: Player's own state.
        locked_target_id: ``combat_target_id`` from current AI state.
        killed: Tank IDs on kill cooldown.

    Returns:
        Pursuit :class:`EnemyThreatDict` synthesised from world state,
        or ``None`` when the locked target is genuinely missing
        (no entry, dead, or no lock).
    """
    if str(locked_target_id) in killed:
        return None
    tank = world["tanks"].get(str(locked_target_id))
    if tank is None:
        return None
    if tank["liveness"] != "alive":
        return None
    if not has_known_position(tank):
        return None
    dist = manhattan_distance(self_state["x"], self_state["y"], tank["x"], tank["y"])
    return make_enemy_threat(
        tank_id=tank["tank_id"],
        x=tank["x"],
        y=tank["y"],
        distance=dist,
        damage_state=tank["damage_state"],
        rank=tank["rank"],
        team=tank["team"],
        name=tank["name"],
        is_bot=tank["is_bot"],
        timestamp_ms=tank["timestamp_ms"],
        last_wire_seen_ms=tank["last_wire_seen_ms"],
        last_position_update_ms=tank["last_position_update_ms"],
        last_aim_x=tank["last_aim_x"],
        last_aim_y=tank["last_aim_y"],
        last_aim_weapon=tank["last_aim_weapon"],
        last_aim_ms=tank["last_aim_ms"],
    )


__all__ = [
    "analyze_threats",
    "find_locked_target_pursuit",
]
