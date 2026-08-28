"""Codecs for the session scorecard and every block inside it.

The scorecard is the one deeply nested payload in the issue report --
state budget, fuel low-water episodes, teleport spend, targeted
teleports, and inventory counts each have their own pair, and
:func:`encode_session_scorecard` / :func:`decode_session_scorecard`
compose them. Reads
:mod:`tankpit_bot.diagnostics.issue_report_codecs_records` for the
JSON guards.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    require_int,
    require_str,
)

from tankpit_bot.diagnostics.issue_report_codecs_records import (
    _require_object,
    _require_object_list,
    _require_str_int_map,
)
from tankpit_bot.diagnostics.issue_report_types import (
    FuelLowWaterEpisodeDict,
    InventoryCountsDict,
    SessionScorecardDict,
    StateBudgetRecordDict,
    TeleportSpendRecordDict,
)


def encode_state_budget_record(record: StateBudgetRecordDict) -> JSONObject:
    """Encode a state budget record to JSON.

    Args:
        record: Record to encode.

    Returns:
        JSON-compatible representation.
    """
    return {
        "state": record["state"],
        "seconds": record["seconds"],
        "stretches": record["stretches"],
        "max_seconds": record["max_seconds"],
    }


def decode_state_budget_record(data: JSONObject) -> StateBudgetRecordDict:
    """Decode a state budget record from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated record. Stretch statistics absent on artifacts
        persisted before they existed decode as 0.
    """
    return StateBudgetRecordDict(
        state=require_str(data, "state"),
        seconds=require_int(data, "seconds"),
        stretches=require_int(data, "stretches") if "stretches" in data else 0,
        max_seconds=require_int(data, "max_seconds") if "max_seconds" in data else 0,
    )


def encode_fuel_low_water_episode(record: FuelLowWaterEpisodeDict) -> JSONObject:
    """Encode a fuel low-water episode to JSON.

    Args:
        record: Episode to encode.

    Returns:
        JSON-compatible representation.
    """
    return {
        "start_timestamp": record["start_timestamp"],
        "end_timestamp": record["end_timestamp"],
        "duration_seconds": record["duration_seconds"],
        "entry_fuel": record["entry_fuel"],
        "min_fuel": record["min_fuel"],
        "cause_kind": record["cause_kind"],
        "cause_drop": record["cause_drop"],
        "cause_state": record["cause_state"],
        "recovery_fuel": record["recovery_fuel"],
        "recovery_kind": record["recovery_kind"],
    }


def decode_fuel_low_water_episode(data: JSONObject) -> FuelLowWaterEpisodeDict:
    """Decode a fuel low-water episode from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated episode.
    """
    return FuelLowWaterEpisodeDict(
        start_timestamp=require_str(data, "start_timestamp"),
        end_timestamp=require_str(data, "end_timestamp"),
        duration_seconds=require_int(data, "duration_seconds"),
        entry_fuel=require_int(data, "entry_fuel"),
        min_fuel=require_int(data, "min_fuel"),
        cause_kind=require_str(data, "cause_kind"),
        cause_drop=require_int(data, "cause_drop"),
        cause_state=require_str(data, "cause_state"),
        recovery_fuel=require_int(data, "recovery_fuel"),
        recovery_kind=require_str(data, "recovery_kind"),
    )


def encode_teleport_spend_record(record: TeleportSpendRecordDict) -> JSONObject:
    """Encode a teleport spend record to JSON.

    Args:
        record: Record to encode.

    Returns:
        JSON-compatible representation.
    """
    return {
        "bot_state": record["bot_state"],
        "drops": record["drops"],
        "fuel_spent": record["fuel_spent"],
    }


def decode_teleport_spend_record(data: JSONObject) -> TeleportSpendRecordDict:
    """Decode a teleport spend record from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated record.
    """
    return TeleportSpendRecordDict(
        bot_state=require_str(data, "bot_state"),
        drops=require_int(data, "drops"),
        fuel_spent=require_int(data, "fuel_spent"),
    )


def encode_inventory_counts(counts: InventoryCountsDict) -> JSONObject:
    """Encode inventory counts to JSON.

    Args:
        counts: Counts to encode.

    Returns:
        JSON-compatible representation.
    """
    return {
        "armor": counts["armor"],
        "dual": counts["dual"],
        "missile": counts["missile"],
        "homing": counts["homing"],
        "radar": counts["radar"],
    }


def decode_inventory_counts(data: JSONObject) -> InventoryCountsDict:
    """Decode inventory counts from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated counts.
    """
    return InventoryCountsDict(
        armor=require_int(data, "armor"),
        dual=require_int(data, "dual"),
        missile=require_int(data, "missile"),
        homing=require_int(data, "homing"),
        radar=require_int(data, "radar"),
    )


def encode_session_scorecard(scorecard: SessionScorecardDict) -> JSONObject:
    """Encode a session scorecard to JSON.

    Args:
        scorecard: Scorecard to encode.

    Returns:
        JSON-compatible representation.
    """
    return {
        "duration_seconds": scorecard["duration_seconds"],
        "state_budget": [encode_state_budget_record(r) for r in scorecard["state_budget"]],
        "kills": scorecard["kills"],
        "shots": scorecard["shots"],
        "combat_misses": scorecard["combat_misses"],
        "tank_damage_changes": scorecard["tank_damage_changes"],
        "fuel_min": scorecard["fuel_min"],
        "fuel_last": scorecard["fuel_last"],
        "fuel_sample_count": scorecard["fuel_sample_count"],
        "inventory_first": encode_inventory_counts(scorecard["inventory_first"]),
        "inventory_last": encode_inventory_counts(scorecard["inventory_last"]),
        "inventory_sample_count": scorecard["inventory_sample_count"],
        "equipment_gain_events": scorecard["equipment_gain_events"],
        "equipment_gained": encode_inventory_counts(scorecard["equipment_gained"]),
        "scans_extra": scorecard["scans_extra"],
        "scans_builtin": scorecard["scans_builtin"],
        "physics_divergences": scorecard["physics_divergences"],
        "action_outcome_counts": dict(scorecard["action_outcome_counts"]),
        "fuel_low_water_threshold": scorecard["fuel_low_water_threshold"],
        "fuel_low_water_episodes": [
            encode_fuel_low_water_episode(r) for r in scorecard["fuel_low_water_episodes"]
        ],
        "teleport_spend": [encode_teleport_spend_record(r) for r in scorecard["teleport_spend"]],
        "teleport_spend_total": scorecard["teleport_spend_total"],
        "ledger_teleport_spend_min": scorecard["ledger_teleport_spend_min"],
        "ledger_teleport_spend_max": scorecard["ledger_teleport_spend_max"],
        "ledger_shot_singles": scorecard["ledger_shot_singles"],
        "ledger_shot_duals": scorecard["ledger_shot_duals"],
        "ledger_shot_homings": scorecard["ledger_shot_homings"],
        "career_destroyed_last": scorecard["career_destroyed_last"],
        "career_deactivated_last": scorecard["career_deactivated_last"],
        "career_score_last": scorecard["career_score_last"],
        "career_playtime_seconds_last": scorecard["career_playtime_seconds_last"],
        "container_pickups_full": scorecard["container_pickups_full"],
        "container_pickups_partial": scorecard["container_pickups_partial"],
    }


def decode_session_scorecard(data: JSONObject) -> SessionScorecardDict:
    """Decode a session scorecard from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated scorecard.
    """
    return SessionScorecardDict(
        duration_seconds=require_int(data, "duration_seconds"),
        state_budget=[
            decode_state_budget_record(item) for item in _require_object_list(data, "state_budget")
        ],
        kills=require_int(data, "kills"),
        shots=require_int(data, "shots"),
        combat_misses=require_int(data, "combat_misses"),
        tank_damage_changes=require_int(data, "tank_damage_changes"),
        fuel_min=require_int(data, "fuel_min"),
        fuel_last=require_int(data, "fuel_last"),
        fuel_sample_count=require_int(data, "fuel_sample_count"),
        inventory_first=decode_inventory_counts(_require_object(data, "inventory_first")),
        inventory_last=decode_inventory_counts(_require_object(data, "inventory_last")),
        inventory_sample_count=require_int(data, "inventory_sample_count"),
        equipment_gain_events=require_int(data, "equipment_gain_events"),
        equipment_gained=decode_inventory_counts(_require_object(data, "equipment_gained")),
        scans_extra=require_int(data, "scans_extra"),
        scans_builtin=require_int(data, "scans_builtin"),
        physics_divergences=require_int(data, "physics_divergences"),
        action_outcome_counts=_require_str_int_map(data, "action_outcome_counts"),
        fuel_low_water_threshold=(
            require_int(data, "fuel_low_water_threshold")
            if "fuel_low_water_threshold" in data
            else 0
        ),
        fuel_low_water_episodes=(
            [
                decode_fuel_low_water_episode(item)
                for item in _require_object_list(data, "fuel_low_water_episodes")
            ]
            if "fuel_low_water_episodes" in data
            else []
        ),
        teleport_spend=(
            [
                decode_teleport_spend_record(item)
                for item in _require_object_list(data, "teleport_spend")
            ]
            if "teleport_spend" in data
            else []
        ),
        teleport_spend_total=(
            require_int(data, "teleport_spend_total") if "teleport_spend_total" in data else 0
        ),
        ledger_teleport_spend_min=(
            require_int(data, "ledger_teleport_spend_min")
            if "ledger_teleport_spend_min" in data
            else -1
        ),
        ledger_teleport_spend_max=(
            require_int(data, "ledger_teleport_spend_max")
            if "ledger_teleport_spend_max" in data
            else -1
        ),
        ledger_shot_singles=(
            require_int(data, "ledger_shot_singles") if "ledger_shot_singles" in data else -1
        ),
        ledger_shot_duals=(
            require_int(data, "ledger_shot_duals") if "ledger_shot_duals" in data else -1
        ),
        ledger_shot_homings=(
            require_int(data, "ledger_shot_homings") if "ledger_shot_homings" in data else -1
        ),
        career_destroyed_last=(
            require_int(data, "career_destroyed_last") if "career_destroyed_last" in data else -1
        ),
        career_deactivated_last=(
            require_int(data, "career_deactivated_last")
            if "career_deactivated_last" in data
            else -1
        ),
        career_score_last=(
            require_int(data, "career_score_last") if "career_score_last" in data else -1
        ),
        career_playtime_seconds_last=(
            require_int(data, "career_playtime_seconds_last")
            if "career_playtime_seconds_last" in data
            else -1
        ),
        container_pickups_full=(
            require_int(data, "container_pickups_full") if "container_pickups_full" in data else 0
        ),
        container_pickups_partial=(
            require_int(data, "container_pickups_partial")
            if "container_pickups_partial" in data
            else 0
        ),
    )


__all__ = [
    "decode_fuel_low_water_episode",
    "decode_inventory_counts",
    "decode_session_scorecard",
    "decode_state_budget_record",
    "decode_teleport_spend_record",
    "encode_fuel_low_water_episode",
    "encode_inventory_counts",
    "encode_session_scorecard",
    "encode_state_budget_record",
    "encode_teleport_spend_record",
]
