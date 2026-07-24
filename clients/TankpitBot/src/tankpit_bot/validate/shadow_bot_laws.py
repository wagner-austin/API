"""Shadow law for the practice-bot policy ([[enemy-bot-behavior]]).

Judges the archive's game-bot shots against the mined policy in
``sim/bot_policy.py`` — the same constants the sim executes, never a
restated copy. A mismatch means the sim's bot model and the real
bots disagree.

The law (``bot-return-fire``): every shot fired by a practice bot is
a ``BOT_RETURN_WEAPON`` single, fired within ``BOT_RETURN_WINDOW_MS``
of the bot taking a hit, aimed at the attacker's last known tile
(the aim clause is only judged when the attacker's position is
known — position channels are viewport-scoped for far tanks).
Archive fit at mining time (2026-07-24): ~95% exact over 2,247 bot
shots; the residual is hit-attribution noise (multi-attacker fights,
stale positions), the same positive-signed shape the audit's clean
windows show.
"""

from __future__ import annotations

import re

from tankpit_bot.physics.capacity import damage_tier, fuel_capacity
from tankpit_bot.sim.bot_policy import BOT_RETURN_WEAPON, BOT_RETURN_WINDOW_MS
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

BOT_NAME_PATTERN = re.compile(r"^(red|purple|blue|orange)-\d+$")
"""Practice-bot naming from the JS ``sd()`` initializer: team-N."""


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
        if BOT_NAME_PATTERN.match(name) is not None
    )


def _shot_obeys_law(
    shot: ShotEventDict,
    hit: tuple[int, int] | None,
    positions: dict[int, tuple[int, int]],
) -> bool:
    """Judge one bot shot against the return-fire law.

    Args:
        shot: The bot's 0x53 echo.
        hit: The latest hit the bot has taken, as
            ``(timestamp_ms, attacker_id)``, or None.
        positions: Latest known tile per tank id at shot time.

    Returns:
        True when the shot is the mined return single: correct
        weapon, inside the return window, and (when the attacker's
        position is known) aimed at the attacker's tile.
    """
    if shot["weapon"] != BOT_RETURN_WEAPON:
        return False
    if hit is None or shot["timestamp_ms"] - hit[0] > BOT_RETURN_WINDOW_MS:
        return False
    attacker_pos = positions.get(hit[1])
    if attacker_pos is None:
        return True
    return shot["target_x"] == attacker_pos[0] and shot["target_y"] == attacker_pos[1]


def shadow_bot_return_fire(timelines: list[ShadowTimelineDict]) -> ClaimEvidenceDict:
    """Judge every practice-bot shot against the mined policy.

    Walks each session's shots and positions in wire order, tracking
    the latest tile per tank and the latest hit landing on each bot
    (a shot whose target tile equals the bot's current tile). One
    sample per bot shot.

    Args:
        timelines: Extracted shadow timelines.

    Returns:
        Evidence for the ``bot-return-fire`` claim.
    """
    samples = 0
    exact = 0
    for timeline in timelines:
        bots = _bot_ids(timeline)
        if not bots:
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
            for bot_id in bots:
                bot_pos = positions.get(bot_id)
                if (
                    bot_pos is not None
                    and bot_id != shot["shooter_id"]
                    and bot_pos == (shot["target_x"], shot["target_y"])
                ):
                    last_hit[bot_id] = (shot["timestamp_ms"], shot["shooter_id"])
            if shot["shooter_id"] in bots:
                samples += 1
                if _shot_obeys_law(shot, last_hit.get(shot["shooter_id"]), positions):
                    exact += 1
    return ClaimEvidenceDict(
        claim_id="bot-return-fire",
        samples=samples,
        exact=exact,
        mismatches=samples - exact,
        detail="bot shots are provoked singles at the attacker (sim/bot_policy)",
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
            samples += 1
            gap_ok = (
                sync["timestamp_ms"] - kill["timestamp_ms"] >= corpse_ms - REACTIVATION_TOLERANCE_MS
            )
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
    "BOT_NAME_PATTERN",
    "REACTIVATION_TOLERANCE_MS",
    "shadow_bot_reactivation",
    "shadow_bot_return_fire",
]
