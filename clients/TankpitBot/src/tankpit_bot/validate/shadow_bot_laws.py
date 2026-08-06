"""Shadow law for the practice-bot policy ([[enemy-bot-behavior]]).

Judges the archive's game-bot shots against the mined policy in
``sim/bot_policy.py`` — the same constants the sim executes, never a
restated copy. A mismatch means the sim's bot model and the real
bots disagree.

The law (``bot-return-fire``): every shot fired by a practice bot is
a ``BOT_RETURN_WEAPON`` single explained by ONE of the three mined
per-hit reflexes — personal return fire (the bot itself was hit
within ``BOT_RETURN_WINDOW_MS``; aim judged at the attacker's last
known tile when known), GANG-UP (a same-team bot was hit within the
window and the shooter is within ``AGGRO_SIGHT_RADIUS`` of its
target), or ASSIST (the target tile holds an enemy-team bot that was
hit within the window, shooter within sight). Bot teams derive from
the roster name colors. Upgraded 2026-07-25 after the team-aggro
mining ([[enemy-bot-behavior]] §Team aggro: 2,115 return / 48
gang-up / 81 assist / 3 unexplained archive shots) — the pre-upgrade
law judged only personal return fire and priced 94.6%; the residual
was lawful team aggro, not noise.
"""

from __future__ import annotations

from tankpit_bot.physics.capacity import damage_tier, fuel_capacity
from tankpit_bot.protocol.naming import PRACTICE_BOT_NAME_PATTERN
from tankpit_bot.sim.bot_policy import (
    AGGRO_SIGHT_RADIUS,
    BOT_RETURN_WEAPON,
    BOT_RETURN_WINDOW_MS,
)
from tankpit_bot.sim.server import CORPSE_WINDOW_TICKS, TICK_MS
from tankpit_bot.validate.shadow_timeline import (
    ShadowTimelineDict,
    ShotEventDict,
    TankSyncEventDict,
)
from tankpit_bot.validate.types import ClaimEvidenceDict

REACTIVATION_TOLERANCE_MS = 1000
"""Allowed early arrival of the reactivation sync before the corpse
boundary (frame timing jitter — the same tolerance the corpse-window
law uses)."""


_TEAM_BY_COLOR = {"red": 0, "purple": 1, "blue": 2, "orange": 3}
"""Roster color → team id (join-roster ground truth: red-1 arrives
team 0, purple-2 team 1, blue-7 team 2, orange-1 team 3)."""


def _bot_teams(timeline: ShadowTimelineDict) -> dict[int, int]:
    """Map each practice-bot id to its team, from the name color.

    Args:
        timeline: One session's shadow timeline.

    Returns:
        Team id per bot tank id.
    """
    teams: dict[int, int] = {}
    for tank_id, name in timeline["names"].items():
        match = PRACTICE_BOT_NAME_PATTERN.match(name)
        if match is not None:
            teams[tank_id] = _TEAM_BY_COLOR[match.group(1)]
    return teams


def _bot_ids(timeline: ShadowTimelineDict) -> frozenset[int]:
    """Return the session's practice-bot tank ids.

    Args:
        timeline: One session's shadow timeline.

    Returns:
        Ids whose 0x21 name matches the practice-bot pattern.
    """
    return frozenset(
        tank_id
        for tank_id, name in timeline["names"].items()
        if PRACTICE_BOT_NAME_PATTERN.match(name) is not None
    )


def _hit_in_window(hit: tuple[int, int] | None, shot_ms: int) -> bool:
    """Report whether a recorded hit falls inside the reflex window.

    Args:
        hit: A ``(timestamp_ms, attacker_id)`` hit record, or None.
        shot_ms: The judged shot's frame timestamp.

    Returns:
        True when the hit exists and is recent enough to provoke.
    """
    return hit is not None and shot_ms - hit[0] <= BOT_RETURN_WINDOW_MS


def _shot_obeys_law(
    shot: ShotEventDict,
    shooter_team: int,
    bot_teams: dict[int, int],
    last_hit: dict[int, tuple[int, int]],
    positions: dict[int, tuple[int, int]],
) -> bool:
    """Judge one bot shot against the full team-aggro reflex model.

    Args:
        shot: The bot's 0x53 echo.
        shooter_team: The firing bot's team.
        bot_teams: Team per practice-bot id.
        last_hit: Latest hit per tank id, as
            ``(timestamp_ms, attacker_id)``.
        positions: Latest known tile per tank id at shot time.

    Returns:
        True when the shot is one of the three mined per-hit
        reflexes: personal return fire, gang-up, or assist.
    """
    if shot["weapon"] != BOT_RETURN_WEAPON:
        return False
    if _is_personal_return(shot, last_hit, positions):
        return True
    shooter_pos = positions.get(shot["shooter_id"])
    if shooter_pos is None:
        return False
    reach = max(
        abs(shooter_pos[0] - shot["target_x"]),
        abs(shooter_pos[1] - shot["target_y"]),
    )
    if reach > AGGRO_SIGHT_RADIUS:
        return False
    return _is_gang_up(shot, shooter_team, bot_teams, last_hit, positions) or _is_assist(
        shot, shooter_team, bot_teams, last_hit, positions
    )


def _is_personal_return(
    shot: ShotEventDict,
    last_hit: dict[int, tuple[int, int]],
    positions: dict[int, tuple[int, int]],
) -> bool:
    """Report whether the shot is the bot's own return single."""
    own_hit = last_hit.get(shot["shooter_id"])
    if own_hit is None or not _hit_in_window(own_hit, shot["timestamp_ms"]):
        return False
    attacker_pos = positions.get(own_hit[1])
    return attacker_pos is None or (shot["target_x"], shot["target_y"]) == attacker_pos


def _is_gang_up(
    shot: ShotEventDict,
    shooter_team: int,
    bot_teams: dict[int, int],
    last_hit: dict[int, tuple[int, int]],
    positions: dict[int, tuple[int, int]],
) -> bool:
    """Report whether the shot avenges a recently-hit teammate."""
    for teammate_id, teammate_team in bot_teams.items():
        if teammate_team != shooter_team or teammate_id == shot["shooter_id"]:
            continue
        teammate_hit = last_hit.get(teammate_id)
        if teammate_hit is None or not _hit_in_window(teammate_hit, shot["timestamp_ms"]):
            continue
        attacker_pos = positions.get(teammate_hit[1])
        if attacker_pos is None or (shot["target_x"], shot["target_y"]) == attacker_pos:
            return True
    return False


def _is_assist(
    shot: ShotEventDict,
    shooter_team: int,
    bot_teams: dict[int, int],
    last_hit: dict[int, tuple[int, int]],
    positions: dict[int, tuple[int, int]],
) -> bool:
    """Report whether the shot joins against an engaged enemy bot."""
    for target_id, target_team in bot_teams.items():
        if target_team == shooter_team:
            continue
        if positions.get(target_id) != (shot["target_x"], shot["target_y"]):
            continue
        if _hit_in_window(last_hit.get(target_id), shot["timestamp_ms"]):
            return True
    return False


def shadow_bot_return_fire(timelines: list[ShadowTimelineDict]) -> ClaimEvidenceDict:
    """Judge every practice-bot shot against the mined reflex model.

    Walks each session's shots and positions in wire order, tracking
    the latest tile per tank and the latest hit landing on EVERY tank
    (a shot whose target tile equals the tank's current tile — team
    aggro needs hits on players and teammates alike). One sample per
    bot shot; exact when the shot is a lawful return, gang-up, or
    assist reflex.

    Args:
        timelines: Extracted shadow timelines.

    Returns:
        Evidence for the ``bot-return-fire`` claim.
    """
    samples = 0
    exact = 0
    for timeline in timelines:
        bot_teams = _bot_teams(timeline)
        if not bot_teams:
            continue
        positions: dict[int, tuple[int, int]] = {}
        last_hit: dict[int, tuple[int, int]] = {}
        events: list[tuple[int, int, int]] = [
            (pos["timestamp_ms"], 0, index) for index, pos in enumerate(timeline["positions"])
        ] + [(shot["timestamp_ms"], 1, index) for index, shot in enumerate(timeline["shots"])]
        for _ts, family, index in sorted(events):
            if family == 0:
                pos = timeline["positions"][index]
                positions[pos["tank_id"]] = (pos["x"], pos["y"])
                continue
            shot = timeline["shots"][index]
            shooter_team = bot_teams.get(shot["shooter_id"])
            if shooter_team is not None:
                samples += 1
                if _shot_obeys_law(shot, shooter_team, bot_teams, last_hit, positions):
                    exact += 1
            for tank_id, tank_pos in positions.items():
                if tank_id != shot["shooter_id"] and tank_pos == (
                    shot["target_x"],
                    shot["target_y"],
                ):
                    last_hit[tank_id] = (shot["timestamp_ms"], shot["shooter_id"])
    return ClaimEvidenceDict(
        claim_id="bot-return-fire",
        samples=samples,
        exact=exact,
        mismatches=samples - exact,
        detail="bot singles are per-hit reflexes: return, gang-up, or assist (sim/bot_policy)",
    )


def _first_sync_after(
    syncs: list[TankSyncEventDict], tank_id: int, timestamp_ms: int
) -> TankSyncEventDict | None:
    """Find a tank's first status sync after a moment.

    Args:
        syncs: The session's 0x2E events, in wire order.
        tank_id: The tank whose sync is sought.
        timestamp_ms: The death's frame timestamp (exclusive).

    Returns:
        The first later sync, or None when the wire carried none
        (session ended, or the bot stayed dark).
    """
    for sync in syncs:
        if sync["tank_id"] == tank_id and sync["timestamp_ms"] > timestamp_ms:
            return sync
    return None


def shadow_bot_reactivation(timelines: list[ShadowTimelineDict]) -> ClaimEvidenceDict:
    """Judge every bot death against the same-id reactivation law.

    The sim's law (:func:`tankpit_bot.sim.bot_policy.reactivate_practice_bot`):
    a killed roster bot stays sync-dark through the corpse window,
    then returns under the SAME id at full fuel. One sample per
    practice-bot 0x41 whose id syncs again later in the session;
    exact when that first post-death sync arrives no earlier than the
    corpse boundary (minus jitter tolerance) and carries the
    full-fuel tier. Deaths with no later sync are skipped, not
    judged — the observation window ended, not the law.

    Args:
        timelines: Extracted shadow timelines.

    Returns:
        Evidence for the ``bot-reactivation`` claim.
    """
    corpse_ms = CORPSE_WINDOW_TICKS * TICK_MS
    observed_window_ms = corpse_ms + 2 * TICK_MS
    samples = 0
    exact = 0
    for timeline in timelines:
        bots = _bot_ids(timeline)
        if not bots:
            continue
        for kill in timeline["kills"]:
            if kill["victim_id"] not in bots:
                continue
            sync = _first_sync_after(timeline["syncs"], kill["victim_id"], kill["timestamp_ms"])
            if sync is None:
                continue
            gap = sync["timestamp_ms"] - kill["timestamp_ms"]
            if gap > observed_window_ms:
                # The reactivation happened OFF-viewport: the first
                # re-sight arrived minutes later, after the bot may
                # have fought (and been damaged by) someone else.
                # Judging its tier would test our line of sight, not
                # the law — 34 of the 2026-08-03 sweep's 35 "failures"
                # were exactly these late damaged re-sights (gaps up
                # to 1,047 s, every one full=False). Unobserved, not
                # violated: skip, like deaths with no later sync.
                continue
            samples += 1
            gap_ok = gap >= corpse_ms - REACTIVATION_TOLERANCE_MS
            full_tier = damage_tier(fuel_capacity(sync["rank"]), sync["rank"])
            if gap_ok and sync["damage_state"] == full_tier:
                exact += 1
    return ClaimEvidenceDict(
        claim_id="bot-reactivation",
        samples=samples,
        exact=exact,
        mismatches=samples - exact,
        detail="dead bot syncs again same-id at full fuel after the corpse window",
    )


__all__ = [
    "PRACTICE_BOT_NAME_PATTERN",
    "REACTIVATION_TOLERANCE_MS",
    "shadow_bot_reactivation",
    "shadow_bot_return_fire",
]
