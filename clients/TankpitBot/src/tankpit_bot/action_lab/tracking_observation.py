"""Pure helpers that build one tracking-observation row.

Lives separately from :mod:`enemy_tracking` so the browser-driven
probe is a thin shell and every routing decision is testable
against synthetic input. The functions here take typed dicts in and
return typed dicts out -- no I/O, no globals, no time. The probe
calls them once per tank per sample with the live world snapshot.
"""

from __future__ import annotations

from tankpit_bot.action_lab.enemy_tracking_types import (
    JSTankBeliefDict,
    OurTankBeliefDict,
    TrackingObservationDict,
)
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.state.types import TankStateDict, WorldStateDict

#: Empty JS belief used when the JS-side identity could not be paired
#: with a registry entry at sample time. The probe records this
#: rather than skipping the row -- a missing entry is itself a
#: measurement.
EMPTY_JS_FIELDS: dict[str, int | float | bool | str | None] = {}


def find_js_tank_entry(
    world_collections: dict[str, list[dict[str, int | float | bool | str | None]]],
    tracked_js_key: str,
    tracked_js_value: str,
) -> dict[str, int | float | bool | str | None] | None:
    """Return the JS tank-registry entry matching ``tracked_js_key``.

    The page-client snapshot's ``world_collections["P.j"]`` holds the
    live JS tank registry (per :mod:`page_client_snapshot`). Each
    item is a dict of primitive fields keyed by the minified
    JS-side property name. ``tracked_js_key`` and
    ``tracked_js_value`` are the (key, stringified-value) pair we
    locked on to at acquisition time -- typically the JS-side tank
    id field -- which gives us a stable identity across samples
    even if the position changes.

    Args:
        world_collections: ``world_collections`` from the page-client
            snapshot.
        tracked_js_key: Minified JS field name whose value uniquely
            identifies the tank within ``P.j``.
        tracked_js_value: Stringified expected value of
            ``tracked_js_key`` for the tracked tank.

    Returns:
        The matching JS-registry entry, or ``None`` when no entry
        matches (the registry no longer lists the tank, or the
        tracked-key pair was never resolved at acquisition time and
        is an empty sentinel).
    """
    if tracked_js_key == "":
        return None
    registry = world_collections.get("P.j")
    if registry is None:
        return None
    for entry in registry:
        raw = entry.get(tracked_js_key)
        if raw is None:
            continue
        if str(raw) == tracked_js_value:
            return entry
    return None


def build_js_belief(
    world_collections: dict[str, list[dict[str, int | float | bool | str | None]]],
    tracked_js_key: str,
    tracked_js_value: str,
) -> JSTankBeliefDict:
    """Build the JS-side belief row for one tank from a snapshot.

    Args:
        world_collections: ``world_collections`` from the page-client
            snapshot.
        tracked_js_key: Minified JS field name whose value uniquely
            identifies the tank within ``P.j``.
        tracked_js_value: Stringified expected value of
            ``tracked_js_key`` for the tracked tank.

    Returns:
        JS belief row. ``present`` is False and ``fields`` is empty
        when the registry entry no longer matches.
    """
    entry = find_js_tank_entry(world_collections, tracked_js_key, tracked_js_value)
    if entry is None:
        return JSTankBeliefDict(present=False, fields=dict(EMPTY_JS_FIELDS))
    return JSTankBeliefDict(present=True, fields=dict(entry))


def build_our_belief(
    *,
    tank_id: int,
    world: WorldStateDict,
    threats: list[EnemyThreatDict],
    sample_timestamp_ms: int,
) -> OurTankBeliefDict:
    """Build our wire-derived belief row for one tank.

    Pairs the registry entry, threat-list membership, and would-be
    lock-fallback verdict in one row. ``locked_target_source``
    captures the gate that fired: ``"threats"`` when ``analyze_threats``
    still includes the tank, ``"world_fallback"`` when only
    ``get_locked_target``'s registry-synthesis path would, and
    ``"none"`` when both paths drop it.

    Args:
        tank_id: Tank id to look up.
        world: World state for this sample.
        threats: ``analyze_threats`` output for this sample.
        sample_timestamp_ms: Sample wall-clock time, used to compute
            wire- and position-age deltas.

    Returns:
        Our belief row for this tank.
    """
    key = str(tank_id)
    tank = world["tanks"].get(key)
    if tank is None:
        return OurTankBeliefDict(
            tank_id=tank_id,
            present=False,
            x=0,
            y=0,
            liveness="",
            last_wire_seen_ms=0,
            last_position_update_ms=0,
            wire_age_ms=0,
            position_age_ms=0,
            is_in_threats=False,
            would_locked_target_return=False,
            locked_target_source="none",
        )
    in_threats = any(threat["tank_id"] == tank_id for threat in threats)
    if in_threats:
        locked_source = "threats"
        would_return = True
    else:
        locked_source = "none"
        would_return = False
    return OurTankBeliefDict(
        tank_id=tank_id,
        present=True,
        x=tank["x"],
        y=tank["y"],
        liveness=tank["liveness"],
        last_wire_seen_ms=tank["last_wire_seen_ms"],
        last_position_update_ms=tank["last_position_update_ms"],
        wire_age_ms=sample_timestamp_ms - tank["last_wire_seen_ms"],
        position_age_ms=sample_timestamp_ms - tank["last_position_update_ms"],
        is_in_threats=in_threats,
        would_locked_target_return=would_return,
        locked_target_source=locked_source,
    )


def build_tracking_observation(
    *,
    sample_index: int,
    sample_timestamp_ms: int,
    tank_id: int,
    tracked_label: str,
    tracked_js_key: str,
    tracked_js_value: str,
    world: WorldStateDict,
    threats: list[EnemyThreatDict],
    world_collections: dict[str, list[dict[str, int | float | bool | str | None]]],
    bot_combat_target_id: int,
    bot_mode_state: str,
) -> TrackingObservationDict:
    """Build one per-tank, per-sample observation row.

    Combines the our-belief and JS-belief sub-rows for one tank
    into the single typed dict the probe persists. Pure -- the
    only inputs are typed dicts and integers.

    Args:
        sample_index: Zero-based sample number.
        sample_timestamp_ms: Wall-clock time of this sample.
        tank_id: Tank id under track.
        tracked_label: Human-readable name captured at acquisition.
        tracked_js_key: JS-side identity key, or empty when not paired.
        tracked_js_value: JS-side identity value as string.
        world: World state for this sample.
        threats: ``analyze_threats`` output for this sample.
        world_collections: ``world_collections`` from the page-client
            snapshot.
        bot_combat_target_id: ``ai_state.combat_target_id`` at sample
            time.
        bot_mode_state: ``ai_state.mode_state`` at sample time.

    Returns:
        One observation row.
    """
    return TrackingObservationDict(
        sample_index=sample_index,
        sample_timestamp_ms=sample_timestamp_ms,
        tank_id=tank_id,
        tracked_label=tracked_label,
        our_belief=build_our_belief(
            tank_id=tank_id,
            world=world,
            threats=threats,
            sample_timestamp_ms=sample_timestamp_ms,
        ),
        js_belief=build_js_belief(world_collections, tracked_js_key, tracked_js_value),
        bot_combat_target_id=bot_combat_target_id,
        bot_mode_state=bot_mode_state,
    )


def select_js_identity_key(
    js_entry: dict[str, int | float | bool | str | None],
    our_tank: TankStateDict,
) -> tuple[str, str]:
    """Pick the minified JS field that identifies the tank across ticks.

    The JS registry hands us minified field names whose semantics we
    don't statically know. We pair entries to our tank at
    acquisition time by *position* -- a tank we just confirmed has
    a wire-fresh ``(x, y)`` -- then ask: which numeric field of
    that entry equals our ``tank_id``? That's the JS-side tank-id
    field, and we use it for cross-tick joins.

    Args:
        js_entry: A registry item that matched our tank by position.
        our_tank: Our world-state entry for the same tank.

    Returns:
        ``(key, str(value))`` pair to record as the tracking key.
        ``("", "")`` when no field equals our tank id (the registry
        either keys by something else or the minification is
        unrecognisable from a single sample -- record that fact so
        the analysis script can see why JS-side rows were unpaired).
    """
    for key, value in js_entry.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value == our_tank["tank_id"]:
            return (key, str(value))
    return ("", "")


def find_js_entry_by_position(
    world_collections: dict[str, list[dict[str, int | float | bool | str | None]]],
    target_x: int,
    target_y: int,
) -> dict[str, int | float | bool | str | None] | None:
    """Return the JS registry entry whose primitives match ``(x, y)``.

    Used once per tank at acquisition time to pair the registry
    entry to a known tank by position. The JS-side x/y fields are
    minified but their values match ours, so we scan all primitive
    integer fields of each entry for one whose value equals the
    target x AND a sibling whose value equals the target y. When
    multiple entries match we return the first -- acquisition
    time is when the bot has freshest data, so ambiguity is rare.

    Args:
        world_collections: ``world_collections`` from the page-client
            snapshot.
        target_x: Our tank's ``x``.
        target_y: Our tank's ``y``.

    Returns:
        The matched registry entry, or ``None`` when none qualifies.
    """
    registry = world_collections.get("P.j")
    if registry is None:
        return None
    for entry in registry:
        has_x = False
        has_y = False
        for value in entry.values():
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                if value == target_x:
                    has_x = True
                if value == target_y:
                    has_y = True
        if has_x and has_y:
            return entry
    return None


__all__ = [
    "EMPTY_JS_FIELDS",
    "build_js_belief",
    "build_our_belief",
    "build_tracking_observation",
    "find_js_entry_by_position",
    "find_js_tank_entry",
    "select_js_identity_key",
]
