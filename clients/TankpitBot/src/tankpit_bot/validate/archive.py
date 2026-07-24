"""Archive validators: firing costs, walk episodes, hit damage, capacity.

All re-derive their claims from wire timelines using the
isolation-window method recorded in ``wiki/log.md`` (2026-07-20,
"Per-weapon firing costs closed from the archive"); window slicing
lives in ``windows``.
"""

from __future__ import annotations

from bisect import bisect_right

from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.physics.costs import (
    DUAL_SHOT_COST,
    HOMING_SHOT_COST,
    MISSILE_SHOT_COST,
    RADAR_COST,
    SINGLE_SHOT_COST,
    WALK_COST_PER_TILE,
)
from tankpit_bot.physics.damage import (
    DUAL_HIT_VICTIM_COST,
    HOMING_HIT_VICTIM_COST,
    MISSILE_HIT_VICTIM_COST,
    SINGLE_HIT_VICTIM_COST,
)
from tankpit_bot.protocol.commands import CMD_RADAR
from tankpit_bot.validate.types import ClaimEvidenceDict
from tankpit_bot.validate.windows import FuelWindowDict, _is_silent_window, _is_walk_window
from tankpit_bot.validate.wire_timeline import WireTimelineDict

WEAPON_SINGLE = 0
WEAPON_DUAL = 1
WEAPON_MISSILE = 2
WEAPON_HOMING = 3

_SHOT_COSTS: dict[int, int] = {
    WEAPON_SINGLE: SINGLE_SHOT_COST,
    WEAPON_DUAL: DUAL_SHOT_COST,
    WEAPON_MISSILE: MISSILE_SHOT_COST,
    WEAPON_HOMING: HOMING_SHOT_COST,
}
_FIRING_CLAIM_IDS: dict[int, str] = {
    WEAPON_SINGLE: "single-shot-cost",
    WEAPON_DUAL: "dual-shot-cost",
    WEAPON_MISSILE: "missile-shot-cost",
    WEAPON_HOMING: "homing-shot-cost",
}
_HIT_COSTS: dict[int, int] = {
    WEAPON_SINGLE: SINGLE_HIT_VICTIM_COST,
    WEAPON_DUAL: DUAL_HIT_VICTIM_COST,
    WEAPON_MISSILE: MISSILE_HIT_VICTIM_COST,
    WEAPON_HOMING: HOMING_HIT_VICTIM_COST,
}
_HIT_CLAIM_IDS: dict[int, str] = {
    WEAPON_SINGLE: "single-hit-victim-cost",
    WEAPON_DUAL: "dual-hit-victim-cost",
    WEAPON_MISSILE: "missile-hit-victim-cost",
    WEAPON_HOMING: "homing-hit-victim-cost",
}


def validate_firing_costs(windows: list[FuelWindowDict]) -> list[ClaimEvidenceDict]:
    """Re-derive the four per-shot firing costs from clean windows.

    A window is clean for weapon ``w`` when it contains exactly one
    own shot (of weapon ``w``) and nothing else. The homing debit may
    split into two -5 steps across sync boundaries, so a -5 homing
    window counts as a sample but not an exact -10 match nor a
    mismatch (per the 2026-07-20 measurement entry).

    Args:
        windows: Fuel windows across the whole archive.

    Returns:
        One evidence record per firing-cost claim.
    """
    samples: dict[int, int] = dict.fromkeys(_SHOT_COSTS, 0)
    exact: dict[int, int] = dict.fromkeys(_SHOT_COSTS, 0)
    mismatches: dict[int, int] = dict.fromkeys(_SHOT_COSTS, 0)
    for window in windows:
        if (
            len(window["own_shot_weapons"]) != 1
            or window["enemy_shot_weapons"]
            or window["walked_tiles"]
            or window["spending_commands"]
            or window["pickups"]
            or window["detonations"]
        ):
            continue
        weapon = window["own_shot_weapons"][0]
        cost = _SHOT_COSTS[weapon]
        samples[weapon] += 1
        if window["delta"] == -cost:
            exact[weapon] += 1
        elif weapon == WEAPON_HOMING and window["delta"] == -(cost // 2):
            continue
        else:
            mismatches[weapon] += 1
    return [
        ClaimEvidenceDict(
            claim_id=_FIRING_CLAIM_IDS[weapon],
            samples=samples[weapon],
            exact=exact[weapon],
            mismatches=mismatches[weapon],
            detail=f"one-shot isolation windows, expected -{_SHOT_COSTS[weapon]}",
        )
        for weapon in sorted(_SHOT_COSTS)
    ]


def validate_walk_cost(windows: list[FuelWindowDict]) -> ClaimEvidenceDict:
    """Re-derive the per-tile walk cost from walk EPISODES.

    Server movement is instantaneous — the full path is billed at the
    echo tick (wiki [[walk-mechanics]], 2026-07-21: 200/200 archive
    episodes carry the whole cost in the echo window). The episode
    still extends through event-free windows to a zero-delta close so
    a debit landing on the window boundary is never split. Only
    SINGLE-ECHO episodes are priced: a second 0x47 inside the episode
    can be a route that never executed (position unchanged at the
    next echo — the source of every tile-overcount in the 2026-07-21
    probe), so multi-echo tile sums are unreliable. The episode's
    delta must equal the walk cost times the echo's step count.
    Episodes cut short by any foreign event are discarded, not judged.

    Args:
        windows: Fuel windows for ONE session, in order.

    Returns:
        Evidence for the walk-cost claim.
    """
    samples = 0
    exact = 0
    mismatches = 0
    index = 0
    while index < len(windows):
        if not _is_walk_window(windows[index]):
            index += 1
            continue
        tiles = 0
        delta = 0
        echoes = 0
        cursor = index
        while cursor < len(windows):
            window = windows[cursor]
            if _is_walk_window(window):
                tiles += window["walked_tiles"]
                echoes += window["move_echoes"]
                delta += window["delta"]
            elif _is_silent_window(window):
                delta += window["delta"]
                if window["delta"] == 0:
                    if echoes == 1:
                        samples += 1
                        if delta == -WALK_COST_PER_TILE * tiles:
                            exact += 1
                        else:
                            mismatches += 1
                    break
            else:
                break
            cursor += 1
        index = cursor + 1
    return ClaimEvidenceDict(
        claim_id="walk-cost",
        samples=samples,
        exact=exact,
        mismatches=mismatches,
        detail="single-echo walk episodes closed by a quiet window, -1 per wire step",
    )


def validate_hit_damage(windows: list[FuelWindowDict]) -> list[ClaimEvidenceDict]:
    """Re-derive single/dual victim costs from lone enemy-shot windows.

    A window with exactly one enemy shot and nothing else must show a
    delta of 0 (the shot targeted someone else or missed) or exactly
    the victim cost for that weapon; anything else is a mismatch.
    Only zero-delta-or-hit windows count as samples of the claim.

    Args:
        windows: Fuel windows across the whole archive.

    Returns:
        One evidence record per hit-damage claim.
    """
    samples: dict[int, int] = dict.fromkeys(_HIT_COSTS, 0)
    exact: dict[int, int] = dict.fromkeys(_HIT_COSTS, 0)
    mismatches: dict[int, int] = dict.fromkeys(_HIT_COSTS, 0)
    for window in windows:
        if (
            len(window["enemy_shot_weapons"]) != 1
            or window["own_shot_weapons"]
            or window["walked_tiles"]
            or window["spending_commands"]
            or window["pickups"]
            or window["detonations"]
        ):
            continue
        weapon = window["enemy_shot_weapons"][0]
        if weapon not in _HIT_COSTS:
            continue
        if window["delta"] == 0:
            continue
        samples[weapon] += 1
        if window["delta"] == -_HIT_COSTS[weapon]:
            exact[weapon] += 1
        else:
            mismatches[weapon] += 1
    return [
        ClaimEvidenceDict(
            claim_id=_HIT_CLAIM_IDS[weapon],
            samples=samples[weapon],
            exact=exact[weapon],
            mismatches=mismatches[weapon],
            detail=f"lone enemy-shot windows with fuel loss, expected -{_HIT_COSTS[weapon]}",
        )
        for weapon in sorted(_HIT_COSTS)
    ]


def validate_fuel_capacity(timelines: list[WireTimelineDict]) -> ClaimEvidenceDict:
    """Check every fuel reading against the rank-derived capacity bound.

    Args:
        timelines: All extracted session timelines.

    Returns:
        Evidence for the fuel-capacity claim: samples are readings in
        rank-known sessions, exact are readings that respect the
        bound, mismatches are readings ABOVE it (impossible if the
        formula holds).
    """
    samples = 0
    exact = 0
    mismatches = 0
    at_cap = 0
    for timeline in timelines:
        rank = timeline["rank"]
        if rank is None:
            continue
        cap = fuel_capacity(rank)
        for reading in timeline["fuel_readings"]:
            samples += 1
            if reading["fuel"] > cap:
                mismatches += 1
            else:
                exact += 1
                if reading["fuel"] == cap:
                    at_cap += 1
    return ClaimEvidenceDict(
        claim_id="fuel-capacity",
        samples=samples,
        exact=exact,
        mismatches=mismatches,
        detail=f"fuel readings vs 1000+100*rank bound; {at_cap} readings AT the cap",
    )


RADAR_CHARGE_GUARD_MS = 3000
"""Backward contamination guard for radar isolation windows.

A prior action's debit can land up to one radar-charge (~3 s) late,
so a window is dirty when any contamination event sits within this
span BEFORE it as well as inside it (wiki/log.md 2026-07-24, "radar
cost archive-isolated": 1,293/1,311 windows exactly -10)."""


def validate_radar_cost(timelines: list[WireTimelineDict]) -> ClaimEvidenceDict:
    """Re-derive the extra-radar scan cost from isolated fuel windows.

    The exact 2026-07-24 mining recipe: a window between consecutive
    fuel readings samples the claim when it contains exactly one sent
    ``CMD_RADAR``, no other sent command of any kind, and no
    contamination — shots either way, pickups, detonations, or
    event-carried fuel readings (0x44/0x64) — inside the window or in
    the :data:`RADAR_CHARGE_GUARD_MS` before it. The window's delta
    must then be exactly ``-RADAR_COST``.

    Args:
        timelines: All extracted session timelines.

    Returns:
        Evidence for the radar-cost claim.
    """
    samples = 0
    exact = 0
    mismatches = 0
    for timeline in timelines:
        radar_times = sorted(
            action["timestamp_ms"]
            for action in timeline["sent_actions"]
            if action["command"] == CMD_RADAR
        )
        other_times = sorted(
            action["timestamp_ms"]
            for action in timeline["sent_actions"]
            if action["command"] != CMD_RADAR
        )
        contamination_times = sorted(
            [shot["timestamp_ms"] for shot in timeline["own_shots"]]
            + [shot["timestamp_ms"] for shot in timeline["enemy_shots"]]
            + timeline["pickup_timestamps"]
            + timeline["detonation_timestamps"]
            + [
                reading["timestamp_ms"]
                for reading in timeline["fuel_readings"]
                if reading["from_event"]
            ]
        )
        readings = timeline["fuel_readings"]
        for index in range(1, len(readings)):
            start_ms = readings[index - 1]["timestamp_ms"]
            end_ms = readings[index]["timestamp_ms"]
            radars = bisect_right(radar_times, end_ms) - bisect_right(radar_times, start_ms)
            others = bisect_right(other_times, end_ms) - bisect_right(other_times, start_ms)
            dirty = bisect_right(contamination_times, end_ms) - bisect_right(
                contamination_times, start_ms - RADAR_CHARGE_GUARD_MS
            )
            if radars != 1 or others or dirty:
                continue
            samples += 1
            if readings[index]["fuel"] - readings[index - 1]["fuel"] == -RADAR_COST:
                exact += 1
            else:
                mismatches += 1
    return ClaimEvidenceDict(
        claim_id="radar-cost",
        samples=samples,
        exact=exact,
        mismatches=mismatches,
        detail="lone-radar fuel windows, 3 s backward guard, -10 per extra scan",
    )


__all__ = [
    "RADAR_CHARGE_GUARD_MS",
    "validate_firing_costs",
    "validate_fuel_capacity",
    "validate_hit_damage",
    "validate_radar_cost",
    "validate_walk_cost",
]
