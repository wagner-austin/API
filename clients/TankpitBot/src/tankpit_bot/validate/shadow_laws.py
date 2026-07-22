"""Shadow-law validators: the sim's laws priced against the archive.

Each validator imports its predictor from the SIM SOURCE — the same
constants and predicates the sim server runs — and re-derives it over
the real capture archive. A mismatch therefore means the sim and the
real server disagree: either a wiki gap or a sim bug, and both demand
investigation. This is the inverse instrument of the seam soaks: the
soaks prove the bot cannot tell the sim from the real server; the
shadow proves the sim cannot be told apart from the archive.

Laws shadowed (v1):

- ``sync-cadence``: per living tank, 0x2E status syncs ride the wire
  at the tick cadence (median inter-sync gap = ``TICK_MS``).
- ``grant-invariants``: every loud 0x67 pickup grants exactly one
  slot, the slot was deficient, counts clip at ``EQUIPMENT_CAP``, and
  uncapped amounts fall inside the measured stack rolls.
- ``kill-mercy-bundle``: an own kill grants the silent multi-slot
  bundle exactly when :func:`kill_grants_mercy` says so (radar zero),
  with amounts inside ``MERCY_BUNDLE_ROLLS``.
- ``corpse-window``: a killed tank's 0x58 TankRemove arrives
  ``CORPSE_WINDOW_TICKS`` ticks after its 0x41.
"""

from __future__ import annotations

from itertools import pairwise
from statistics import median

from tankpit_bot.sim.equipment import (
    EQUIPMENT_CAP,
    MERCY_BUNDLE_ROLLS,
    RADAR_SLOT,
    RADAR_STACK_ROLL,
    WEAPON_STACK_ROLL,
    kill_grants_mercy,
)
from tankpit_bot.sim.server import CORPSE_WINDOW_TICKS, TICK_MS
from tankpit_bot.validate.shadow_timeline import (
    EquipmentGainEventDict,
    InventoryEventDict,
    KillEventDict,
    ShadowTimelineDict,
)
from tankpit_bot.validate.types import ClaimEvidenceDict

SYNC_TOLERANCE_MS = 500
"""Allowed deviation of a tank's median inter-sync gap from the tick."""

SYNC_MIN_GAPS = 5
"""Minimum inter-sync gaps before a tank's cadence is judged."""

PAIRING_WINDOW_MS = TICK_MS
"""Window for pairing a 0x67 with its snapshot / a kill with its bundle
(the corpus delivers both in the very next frame)."""

CORPSE_TOLERANCE_MS = 1000
"""Allowed deviation of the kill->remove gap from the corpse window."""


def shadow_sync_cadence(timelines: list[ShadowTimelineDict]) -> ClaimEvidenceDict:
    """Judge the OTHER-tank sync cadence against the sim tick.

    One sample per non-self tank with at least ``SYNC_MIN_GAPS``
    inter-sync gaps; exact when the tank's MEDIAN gap sits within
    ``SYNC_TOLERANCE_MS`` of ``TICK_MS``. The median is the judge
    because real sessions interleave quiet stretches (viewport exits,
    deaths) with the steady broadcast.

    The SELF tank is excluded by measurement, not convenience: the
    2026-07-22 calibration sweep found inlier medians pinned at
    1981-2010 ms (the 2 s law, dead on), but 23 of ~220 self tanks
    (~10% of sessions) drifted to 3-4 s+ medians — a mode other tanks
    never show (their only outliers are brief-observation noise).
    The self tank's own truth also rides 0x44/0x64/0x49, so its 0x2E
    cadence is evidently not load-bearing and not uniform; the law
    the sim MUST hold is the other-tank broadcast. Recorded as an
    open finding ([[physics-module-roadmap]]).

    Args:
        timelines: Extracted shadow timelines.

    Returns:
        Evidence for the ``sync-cadence`` claim.
    """
    samples = 0
    exact = 0
    for timeline in timelines:
        per_tank: dict[int, list[int]] = {}
        for sync in timeline["syncs"]:
            if sync["tank_id"] == timeline["self_id"]:
                continue
            per_tank.setdefault(sync["tank_id"], []).append(sync["timestamp_ms"])
        for stamps in per_tank.values():
            gaps = [after - before for before, after in pairwise(stamps)]
            if len(gaps) < SYNC_MIN_GAPS:
                continue
            samples += 1
            if abs(median(gaps) - TICK_MS) <= SYNC_TOLERANCE_MS:
                exact += 1
    return ClaimEvidenceDict(
        claim_id="sync-cadence",
        samples=samples,
        exact=exact,
        mismatches=samples - exact,
        detail=f"per-tank median 0x2E gap within {SYNC_TOLERANCE_MS}ms of TICK_MS",
    )


def _next_inventory(
    inventories: list[InventoryEventDict], timestamp_ms: int
) -> InventoryEventDict | None:
    """Find the first inventory snapshot inside the pairing window.

    Args:
        inventories: The session's 0x49 snapshots, in wire order.
        timestamp_ms: The gain's frame timestamp.

    Returns:
        The paired snapshot, or None when the wire carried none in
        time (the pairing itself is not the law under test).
    """
    for snapshot in inventories:
        if timestamp_ms <= snapshot["timestamp_ms"] <= timestamp_ms + PAIRING_WINDOW_MS:
            return snapshot
    return None


def _grant_obeys_invariants(gained: list[int], post: list[int]) -> bool:
    """Check one loud grant against the sim's grant law.

    Args:
        gained: The 0x67 five-slot gained array.
        post: The paired snapshot's counts (post-grant).

    Returns:
        True when the grant matches every invariant: one slot, slot
        was deficient, cap respected, amount a cap-clip or an in-range
        stack roll.
    """
    nonzero = [slot for slot, amount in enumerate(gained) if amount != 0]
    if len(nonzero) != 1:
        return False
    slot = nonzero[0]
    amount = gained[slot]
    pre = post[slot] - amount
    if pre < 0 or pre >= EQUIPMENT_CAP or post[slot] > EQUIPMENT_CAP:
        return False
    if post[slot] == EQUIPMENT_CAP:
        return True
    low, high = RADAR_STACK_ROLL if slot == RADAR_SLOT else WEAPON_STACK_ROLL
    return low <= amount <= high


def shadow_grant_invariants(timelines: list[ShadowTimelineDict]) -> ClaimEvidenceDict:
    """Judge every loud equipment grant against the sim's grant law.

    One sample per loud 0x67 whose 0x49 snapshot follows inside the
    pairing window (``pre = post - gained`` is exact in the corpus);
    unpaired gains are skipped, not judged.

    Args:
        timelines: Extracted shadow timelines.

    Returns:
        Evidence for the ``grant-invariants`` claim.
    """
    samples = 0
    exact = 0
    for timeline in timelines:
        for gain in timeline["gains"]:
            if not gain["show_message"]:
                continue
            snapshot = _next_inventory(timeline["inventories"], gain["timestamp_ms"])
            if snapshot is None:
                continue
            samples += 1
            if _grant_obeys_invariants(gain["gained"], snapshot["counts"]):
                exact += 1
    return ClaimEvidenceDict(
        claim_id="grant-invariants",
        samples=samples,
        exact=exact,
        mismatches=samples - exact,
        detail="one deficient slot, cap 25 clip, stack rolls 5-9 / 2-4",
    )


def _radar_before(inventories: list[InventoryEventDict], timestamp_ms: int) -> int | None:
    """Return the extra-radar count from the last snapshot before a kill.

    Args:
        inventories: The session's 0x49 snapshots, in wire order.
        timestamp_ms: The kill's frame timestamp (exclusive — the
            bundle's own snapshot shares the kill frame).

    Returns:
        The radar count, or None when no snapshot precedes the kill.
    """
    radar: int | None = None
    for snapshot in inventories:
        if snapshot["timestamp_ms"] >= timestamp_ms:
            break
        radar = snapshot["counts"][RADAR_SLOT]
    return radar


def _silent_bundle_after(
    gains: list[EquipmentGainEventDict], timestamp_ms: int
) -> EquipmentGainEventDict | None:
    """Find a silent 0x67 bundle inside the kill's pairing window.

    Args:
        gains: The session's 0x67 events, in wire order.
        timestamp_ms: The kill's frame timestamp.

    Returns:
        The silent bundle, or None.
    """
    for gain in gains:
        if gain["show_message"]:
            continue
        if timestamp_ms <= gain["timestamp_ms"] <= timestamp_ms + PAIRING_WINDOW_MS:
            return gain
    return None


def _bundle_in_rolls(gained: list[int]) -> bool:
    """Check a mercy bundle's amounts against the measured rolls.

    Args:
        gained: The silent 0x67 five-slot gained array.

    Returns:
        True when every slot's amount falls inside its measured range.
    """
    return all(low <= gained[slot] <= high for slot, (low, high) in enumerate(MERCY_BUNDLE_ROLLS))


def _judge_mercy_kill(timeline: ShadowTimelineDict, kill: KillEventDict) -> bool | None:
    """Judge one own kill against the mercy-bundle law.

    Args:
        timeline: The kill's session timeline.
        kill: The own-kill event.

    Returns:
        True/False when the kill is a judged sample (law obeyed /
        violated), None when it cannot be judged (no prior snapshot).
    """
    radar = _radar_before(timeline["inventories"], kill["timestamp_ms"])
    if radar is None:
        return None
    predicted = kill_grants_mercy(radar)
    bundle = _silent_bundle_after(timeline["gains"], kill["timestamp_ms"])
    if bundle is None:
        return not predicted
    return predicted and _bundle_in_rolls(bundle["gained"])


def shadow_mercy_bundle(timelines: list[ShadowTimelineDict]) -> ClaimEvidenceDict:
    """Judge every own kill against the radar-zero mercy-bundle law.

    One sample per own (non-mine) kill with a known prior radar count;
    exact when the silent bundle's presence matches
    :func:`kill_grants_mercy` and, when present, its amounts fall
    inside ``MERCY_BUNDLE_ROLLS``.

    Args:
        timelines: Extracted shadow timelines.

    Returns:
        Evidence for the ``kill-mercy-bundle`` claim.
    """
    samples = 0
    exact = 0
    for timeline in timelines:
        for kill in timeline["kills"]:
            if kill["is_mine_kill"] or kill["killer_id"] != timeline["self_id"]:
                continue
            verdict = _judge_mercy_kill(timeline, kill)
            if verdict is None:
                continue
            samples += 1
            if verdict:
                exact += 1
    return ClaimEvidenceDict(
        claim_id="kill-mercy-bundle",
        samples=samples,
        exact=exact,
        mismatches=samples - exact,
        detail="silent bundle iff radar zero at own kill, amounts in rolls",
    )


def _corpse_gap_ms(timeline: ShadowTimelineDict, kill: KillEventDict) -> int | None:
    """Measure one kill's corpse-removal gap, filtering reuse noise.

    Args:
        timeline: The kill's session timeline.
        kill: The kill event.

    Returns:
        The kill->remove gap in milliseconds, or None when the wire
        carried no removal, the victim quit first (0x29), or the id
        synced again before removal (slot reuse — a dead tank never
        syncs).
    """
    victim = kill["victim_id"]
    removal_ms: int | None = None
    for removal in timeline["removals"]:
        if removal["tank_id"] == victim and removal["timestamp_ms"] > kill["timestamp_ms"]:
            removal_ms = removal["timestamp_ms"]
            break
    if removal_ms is None:
        return None
    for exit_event in timeline["exits"]:
        if (
            exit_event["tank_id"] == victim
            and kill["timestamp_ms"] < exit_event["timestamp_ms"] < removal_ms
        ):
            return None
    for sync in timeline["syncs"]:
        if sync["tank_id"] == victim and kill["timestamp_ms"] < sync["timestamp_ms"] < removal_ms:
            return None
    return removal_ms - kill["timestamp_ms"]


def shadow_corpse_window(timelines: list[ShadowTimelineDict]) -> ClaimEvidenceDict:
    """Judge every kill's corpse removal against the sim's window.

    One sample per kill whose victim's next 0x58 arrives with no
    intervening quit or id-reuse sync; exact when the gap sits within
    ``CORPSE_TOLERANCE_MS`` of ``CORPSE_WINDOW_TICKS * TICK_MS``.

    Args:
        timelines: Extracted shadow timelines.

    Returns:
        Evidence for the ``corpse-window`` claim.
    """
    predicted_ms = CORPSE_WINDOW_TICKS * TICK_MS
    samples = 0
    exact = 0
    for timeline in timelines:
        for kill in timeline["kills"]:
            gap = _corpse_gap_ms(timeline, kill)
            if gap is None:
                continue
            samples += 1
            if abs(gap - predicted_ms) <= CORPSE_TOLERANCE_MS:
                exact += 1
    return ClaimEvidenceDict(
        claim_id="corpse-window",
        samples=samples,
        exact=exact,
        mismatches=samples - exact,
        detail=f"kill->0x58 gap = {predicted_ms}ms (CORPSE_WINDOW_TICKS * TICK_MS)",
    )


__all__ = [
    "CORPSE_TOLERANCE_MS",
    "PAIRING_WINDOW_MS",
    "SYNC_MIN_GAPS",
    "SYNC_TOLERANCE_MS",
    "shadow_corpse_window",
    "shadow_grant_invariants",
    "shadow_mercy_bundle",
    "shadow_sync_cadence",
]
