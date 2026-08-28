"""Scorecard DIAGNOSTIC routing: fold diagnostic kinds into the accumulator.

Split from :mod:`tankpit_bot.diagnostics.session_scorecard_accumulator`
(2026-08-20, at the file-size bar) when kill attribution landed: the
channel router and the WORLD fuel receipts stay there; every
DIAGNOSTIC-kind arm lives here. First-match-wins disjointness across
the arms is pinned by ``tests/sniffer/test_dispatch_exclusivity.py``.
"""

from __future__ import annotations

from tankpit_bot.diagnostics.issue_report_types import (
    InventoryCountsDict,
)
from tankpit_bot.diagnostics.session_scorecard_types import (
    FuelSampleRecordDict,
    ScorecardAccumulatorDict,
    optional_int_field,
    optional_str_field,
)
from tankpit_bot.runtime_records import (
    RuntimeEventRecordDict,
    require_bool_field,
    require_int_field,
    require_str_field,
)


def _classify_inventory_counts(record: RuntimeEventRecordDict) -> InventoryCountsDict:
    """Build typed inventory counts from a DIAGNOSTIC event.

    Args:
        record: Decoded event record whose ``diagnostic_kind`` is
            ``inventory_sample`` or ``equipment_gain``.

    Returns:
        Strict-typed inventory counts.
    """
    fields = record["fields"]
    return InventoryCountsDict(
        armor=require_int_field(fields, "armor"),
        dual=require_int_field(fields, "dual"),
        missile=require_int_field(fields, "missile"),
        homing=require_int_field(fields, "homing"),
        radar=require_int_field(fields, "radar"),
    )


def _add_inventory_counts(
    totals: InventoryCountsDict,
    gained: InventoryCountsDict,
) -> InventoryCountsDict:
    """Return per-type totals with one gain event added.

    Args:
        totals: Running totals.
        gained: Gain amounts from one ``equipment_gain`` event.

    Returns:
        New totals.
    """
    return InventoryCountsDict(
        armor=totals["armor"] + gained["armor"],
        dual=totals["dual"] + gained["dual"],
        missile=totals["missile"] + gained["missile"],
        homing=totals["homing"] + gained["homing"],
        radar=totals["radar"] + gained["radar"],
    )


def route_scorecard_diagnostic(
    record: RuntimeEventRecordDict,
    accumulator: ScorecardAccumulatorDict,
) -> None:
    """Route one DIAGNOSTIC record into the scorecard accumulator.

    Args:
        record: Decoded DIAGNOSTIC-channel event record.
        accumulator: Scorecard accumulator to update in place.
    """
    kind = record["fields"].get("diagnostic_kind")
    if _route_combat_diagnostic(kind, record, accumulator):
        return
    if _route_fuel_diagnostic(kind, record, accumulator):
        return
    if kind == "inventory_sample":
        accumulator["inventory_samples"].append(_classify_inventory_counts(record))
    elif kind == "equipment_gain":
        accumulator["equipment_gain_events"] += 1
        accumulator["equipment_gained"] = _add_inventory_counts(
            accumulator["equipment_gained"],
            _classify_inventory_counts(record),
        )
    elif kind == "physics_divergence":
        accumulator["physics_divergences"] += 1
    elif kind == "radar_dispatch":
        if require_bool_field(record["fields"], "uses_extra"):
            accumulator["scans_extra"] += 1
        else:
            accumulator["scans_builtin"] += 1
    elif kind == "action_outcome":
        _route_action_outcome(record, accumulator)
    else:
        _route_metrics_diagnostic(kind, record, accumulator)


def _route_action_outcome(
    record: RuntimeEventRecordDict,
    accumulator: ScorecardAccumulatorDict,
) -> None:
    """Tally one action outcome, and remember completed map opens.

    Split out of :func:`route_scorecard_diagnostic` for the same reason
    as :func:`_route_metrics_diagnostic`: keeping the primary router
    under the C901 complexity ceiling.

    The map-open timestamp is kept because a map open is dispatched FROM
    the IDLE state and has no state of its own, so the state budget needs
    it to tell an idle stretch from one that was waiting on MAP_DATA.

    Args:
        record: Decoded ``action_outcome`` DIAGNOSTIC record.
        accumulator: Scorecard accumulator to update in place.
    """
    action_kind = require_str_field(record["fields"], "action_kind")
    counter_key = f"{action_kind}:{require_str_field(record['fields'], 'outcome')}"
    counts = accumulator["action_outcome_counts"]
    counts[counter_key] = counts.get(counter_key, 0) + 1
    if action_kind == "map_open":
        accumulator["map_open_completions_at"].append(record["timestamp"])


def _route_fuel_diagnostic(
    kind: str | int | float | bool | None,
    record: RuntimeEventRecordDict,
    accumulator: ScorecardAccumulatorDict,
) -> bool:
    """Route the fuel-trajectory diagnostics into the accumulator.

    Covers the three kinds the low-water / teleport-spend analysis
    reads: context-stamped fuel samples, the engagement-break escape
    floors (the session's own danger line), and the end-of-run damage
    ledger's authoritative billing totals.

    Args:
        kind: ``diagnostic_kind`` field value.
        record: Decoded event record carrying the structured payload.
        accumulator: Scorecard accumulator to update in place.

    Returns:
        True when ``kind`` matched and was applied, False otherwise.
    """
    fields = record["fields"]
    if kind == "self_alignment_sample":
        accumulator["fuel_samples"].append(
            FuelSampleRecordDict(
                timestamp=record["timestamp"],
                fuel=require_int_field(fields, "belief_fuel"),
                bot_state=optional_str_field(fields, "bot_state", ""),
                in_flight=optional_str_field(fields, "in_flight_action_kind", "none"),
            )
        )
        return True
    if kind == "engagement_break":
        accumulator["max_escape_floor"] = max(
            accumulator["max_escape_floor"],
            optional_int_field(fields, "escape_floor", 0),
        )
        return True
    if kind == "damage_ledger":
        # The fuel book's ``[lo, hi]`` sums are feasibility bounds on
        # spend, both negative; negate into a positive min..max spend
        # interval. Absent fields (pre-ledger artifacts) keep the -1
        # sentinels from the accumulator factory.
        lo = optional_int_field(fields, "teleport_fuel_lo", 1)
        hi = optional_int_field(fields, "teleport_fuel_hi", 1)
        accumulator["ledger_teleport_spend_max"] = -lo if lo <= 0 else -1
        accumulator["ledger_teleport_spend_min"] = -hi if hi <= 0 else -1
        accumulator["ledger_shot_singles"] = optional_int_field(fields, "shot_single_count", -1)
        accumulator["ledger_shot_duals"] = optional_int_field(fields, "shot_dual_count", -1)
        accumulator["ledger_shot_homings"] = optional_int_field(fields, "shot_homing_count", -1)
        return True
    return False


def _route_metrics_diagnostic(
    kind: str | int | float | bool | None,
    record: RuntimeEventRecordDict,
    accumulator: ScorecardAccumulatorDict,
) -> None:
    """Route career-stats and pickup-tally diagnostics into the scorecard accumulator.

    Split out of :func:`route_scorecard_diagnostic` to keep the
    primary router under the C901 complexity ceiling.

    Args:
        kind: ``diagnostic_kind`` field value.
        record: Decoded event record carrying the structured payload.
        accumulator: Scorecard accumulator to update in place.
    """
    if kind == "scope_shift_sent":
        accumulator["scope_shift_sends_at"].append(record["timestamp"])
        return
    if kind == "self_statistics":
        accumulator["career_destroyed_last"] = require_int_field(record["fields"], "destroyed")
        accumulator["career_deactivated_last"] = require_int_field(record["fields"], "deactivated")
        accumulator["career_score_last"] = require_int_field(record["fields"], "score")
        accumulator["career_playtime_seconds_last"] = require_int_field(
            record["fields"], "playtime_seconds_total"
        )
        return
    if kind == "container_pickup_dispatched":
        if record["fields"].get("is_partial") is True:
            accumulator["container_pickups_partial"] += 1
        else:
            accumulator["container_pickups_full"] += 1


def _route_combat_diagnostic(
    kind: str | int | float | bool | None,
    record: RuntimeEventRecordDict,
    accumulator: ScorecardAccumulatorDict,
) -> bool:
    """Increment the combat counters that the freshness gates emit.

    Args:
        kind: ``diagnostic_kind`` field value pulled from the record's
            ``fields`` dict. Non-string values can never match the
            string literals tested below, so they fall through and the
            caller routes the record onward.
        record: Decoded event record carrying the structured payload.
        accumulator: Scorecard accumulator to update in place.

    Returns:
        True when ``kind`` matched a combat counter and was applied,
        False otherwise. The caller routes non-matching kinds onward.
    """
    if kind == "tank_identity":
        # First identity wins, mirroring the run digest: the wire names
        # this session's own tank exactly once per entry.
        if accumulator["self_tank_id"] == -1:
            accumulator["self_tank_id"] = require_int_field(record["fields"], "tank_id")
        return True
    if kind == "tank_deactivated":
        # OUR kill only when the 0x41 names this session's tank as the
        # killer. The pre-fleet counter took the raw event count ("the
        # raw count is the kill count") -- true only while every 0x41
        # in view was our own kill. The first fleet firefight falsified
        # it for the live counter (2026-08-14, arterial banked artax's
        # two kills on zero shots fired) and the first gatherer run
        # falsified it here (2026-08-20, the analyzer's scorecard read
        # kills=2 shots=0). Unattributable events -- no killer_id field
        # on pre-fleet artifacts, or no identity seen yet -- count
        # never, mirroring the live registry's split.
        killer_id = optional_int_field(record["fields"], "killer_id", -1)
        if killer_id != -1 and killer_id == accumulator["self_tank_id"]:
            accumulator["kills"] += 1
        return True
    if kind == "combat_miss":
        accumulator["combat_misses"] += 1
        return True
    if kind == "tank_damage_changed":
        accumulator["tank_damage_changes"] += 1
        return True
    return False


__all__ = [
    "route_scorecard_diagnostic",
]
