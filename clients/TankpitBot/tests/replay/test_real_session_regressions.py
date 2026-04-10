"""Replay regressions for real captured bad sessions."""

from __future__ import annotations

from collections import Counter
from itertools import pairwise
from pathlib import Path

import pytest

from tankpit_bot.replay.engine import replay_session
from tankpit_bot.sniffer.viewport import reset_viewport_tracking
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.sniffer.xor import reset_xor_state
from tests.replay.fixture_loader import get_replay_fixture_path, load_capture_fixture


def _reset_replay_globals() -> None:
    """Reset global replay state shared by the decoder pipeline."""
    reset_world_state()
    reset_xor_state()
    reset_viewport_tracking()


def test_fuel_radar_loop_fixture_exists() -> None:
    """The checked-in fuel/radar loop capture fixture is present."""
    path = get_replay_fixture_path("fuel_radar_loop.capture_session.json")
    assert path.name == "fuel_radar_loop.capture_session.json"
    assert path.is_file()


def test_equipment_then_fuel_loop_fixture_exists() -> None:
    """The checked-in equipment-then-fuel loop capture fixture is present."""
    path = get_replay_fixture_path("equipment_then_fuel_loop.capture_session.json")
    assert path.name == "equipment_then_fuel_loop.capture_session.json"
    assert path.is_file()


def test_viewport_enemy_shoot_rejection_loop_fixture_exists() -> None:
    """The checked-in viewport enemy shoot-rejection fixture is present."""
    path = get_replay_fixture_path("viewport_enemy_shoot_rejection_loop.capture_session.json")
    assert path.name == "viewport_enemy_shoot_rejection_loop.capture_session.json"
    assert path.is_file()


def test_combat_to_fuel_stale_lock_loop_fixture_exists() -> None:
    """The checked-in combat-to-fuel stale-lock fixture is present."""
    path = get_replay_fixture_path("combat_to_fuel_stale_lock_loop.capture_session.json")
    assert path.name == "combat_to_fuel_stale_lock_loop.capture_session.json"
    assert path.is_file()


def test_hunt_search_confirm_kill_loop_fixture_exists() -> None:
    """The checked-in HUNT search confirm-kill fixture is present."""
    path = get_replay_fixture_path("hunt_search_confirm_kill_loop.capture_session.json")
    assert path.name == "hunt_search_confirm_kill_loop.capture_session.json"
    assert path.is_file()


def test_missing_replay_fixture_path_raises() -> None:
    """Missing replay fixtures fail loudly with FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        get_replay_fixture_path("missing.capture_session.json")
    assert Path(str(exc_info.value)).name == "missing.capture_session.json"


def test_fuel_radar_loop_replays_known_bad_behavior() -> None:
    """Replay now converts the captured fuel/radar churn into concrete pickups.

    This capture originally produced repeated radar churn after visible
    containers appeared. The repaired behavior should still enter
    ``RECOVER_FUEL`` from ``HUNT``, but it must materially reduce radar spam
    and issue concrete fuel pickups once the radar/cache path surfaces targets.
    """
    _reset_replay_globals()
    session = load_capture_fixture("fuel_radar_loop.capture_session.json")

    result = replay_session(session)
    traces = result["traces"]
    behavior_counts = Counter(trace["behavior_mode"] for trace in traces)
    command_counts = Counter(trace["command_type"] for trace in traces)
    first_visible_container_tick = next(
        trace["tick_index"] for trace in traces if trace["container_count"] > 0
    )
    post_container_traces = [
        trace for trace in traces if trace["tick_index"] >= first_visible_container_tick
    ]
    radar_while_containers_visible = [
        trace
        for trace in post_container_traces
        if trace["behavior_reason"] == "radar_for_fuel" and trace["container_count"] > 0
    ]
    pickup_commands = [
        trace for trace in traces if trace["command_type"] in ("pickup_fuel", "pickup_equipment")
    ]
    fuel_gain_ticks = [
        current["tick_index"]
        for previous, current in pairwise(traces)
        if current["fuel"] > previous["fuel"]
    ]

    assert result["session_id"] == "c6923736-de9a-4d7d-8898-60df9a64485d"
    assert result["total_ticks"] == 42
    assert result["total_messages"] == 244
    assert traces[0]["ai_mode"] == "HUNT"
    assert traces[5]["ai_mode"] == "RECOVER_FUEL"
    assert traces[6]["ai_mode"] == "RECOVER_FUEL"
    assert traces[6]["ai_mode_state"] == "APPROACH"
    assert traces[5]["behavior_mode"] == "COLLECT_FUEL"
    assert behavior_counts["HUNT"] == 5
    assert behavior_counts["COLLECT_FUEL"] == 37
    assert command_counts["radar"] == 1
    assert command_counts["move"] == 25
    assert command_counts["teleport"] == 14
    assert command_counts["pickup_fuel"] == 1
    assert first_visible_container_tick == 6
    assert len(radar_while_containers_visible) == 0
    assert len(pickup_commands) == 1
    assert fuel_gain_ticks == [24]
    assert traces[6]["behavior_reason"] == "known_fuel=251"
    assert traces[6]["command_type"] == "teleport"
    assert traces[17]["behavior_reason"] == "fuel=99"
    assert traces[17]["command_type"] == "pickup_fuel"

    _reset_replay_globals()


def test_equipment_then_fuel_loop_replays_known_bad_behavior() -> None:
    """Replay now converts the captured equipment search into a real pickup."""
    _reset_replay_globals()
    session = load_capture_fixture("equipment_then_fuel_loop.capture_session.json")

    result = replay_session(session)
    traces = result["traces"]
    behavior_counts = Counter(trace["behavior_mode"] for trace in traces)
    command_counts = Counter(trace["command_type"] for trace in traces)
    first_fuel_tick = next(
        trace["tick_index"] for trace in traces if trace["behavior_mode"] == "COLLECT_FUEL"
    )
    equipment_prefix = traces[:first_fuel_tick]
    fuel_suffix = traces[first_fuel_tick:]

    assert result["session_id"] == "90ba8a00-6001-42b1-9e27-b4b4a56882e6"
    assert result["total_ticks"] == 30
    assert result["total_messages"] == 187
    assert behavior_counts["COLLECT_EQUIPMENT"] == 8
    assert behavior_counts["COLLECT_FUEL"] == 22
    assert command_counts["radar"] == 3
    assert command_counts["move"] == 20
    assert command_counts["teleport"] == 6
    assert command_counts["pickup_equipment"] == 1
    assert first_fuel_tick == 8
    assert all(trace["behavior_mode"] == "COLLECT_EQUIPMENT" for trace in equipment_prefix)
    assert all(trace["behavior_mode"] == "COLLECT_FUEL" for trace in fuel_suffix)
    assert [trace["behavior_reason"] for trace in equipment_prefix[:3]] == [
        "radar_for_equipment",
        "search_equipment_local",
        "search_equipment_local",
    ]
    assert traces[7]["behavior_reason"] == "equipment_critical"
    assert traces[7]["command_type"] == "pickup_equipment"
    assert traces[8]["behavior_reason"] == "known_fuel=1019"
    assert sum(1 for trace in traces if trace["container_count"] > 0) == 26
    assert traces[-1]["fuel"] == 0
    assert traces[-1]["behavior_reason"] == "known_fuel=1018"

    _reset_replay_globals()


def test_viewport_enemy_shoot_rejection_loop_replays_known_bad_behavior() -> None:
    """Replay reproduces the visible-enemy shoot/reject loop from live play."""
    _reset_replay_globals()
    session = load_capture_fixture("viewport_enemy_shoot_rejection_loop.capture_session.json")

    result = replay_session(session)
    traces = result["traces"]
    behavior_counts = Counter(trace["behavior_mode"] for trace in traces)
    command_counts = Counter(trace["command_type"] for trace in traces)
    shoot_traces = [trace for trace in traces if trace["command_type"] == "shoot"]

    assert result["session_id"] == "96f3427c-12c2-4c65-a8d6-ec9dc3dc7972"
    assert result["total_ticks"] == 9
    assert result["total_messages"] == 59
    assert behavior_counts == Counter({"HUNT": 9})
    assert command_counts["map_open"] == 1
    assert command_counts["shoot"] == 8
    assert traces[0]["behavior_reason"] == "find orange-1"
    assert traces[0]["command_type"] == "map_open"
    assert traces[0]["ai_mode"] == "HUNT"
    assert traces[0]["ai_mode_state"] == "REFRESH"
    assert all(trace["behavior_reason"] == "shoot orange-1" for trace in shoot_traces)
    assert all(trace["combat_target_id"] == 527 for trace in shoot_traces)
    assert all(trace["ai_mode"] == "HUNT" for trace in traces)
    assert all(trace["ai_mode_state"] == "ENGAGE" for trace in shoot_traces)

    _reset_replay_globals()


def test_combat_to_fuel_stale_lock_loop_replays_recovery_then_reengage() -> None:
    """Replay reproduces the combat-to-fuel handoff without getting stuck."""
    _reset_replay_globals()
    session = load_capture_fixture("combat_to_fuel_stale_lock_loop.capture_session.json")

    result = replay_session(session)
    traces = result["traces"]
    behavior_counts = Counter(trace["behavior_mode"] for trace in traces)
    command_counts = Counter(trace["command_type"] for trace in traces)

    assert result["session_id"] == "43c10dc5-a93b-4d0d-b702-12f0a718cae1"
    assert result["total_ticks"] == 19
    assert result["total_messages"] == 145
    assert behavior_counts["HUNT"] == 15
    assert behavior_counts["COLLECT_FUEL"] == 4
    assert command_counts["shoot"] == 9
    assert command_counts["radar"] == 1
    assert command_counts["pickup_fuel"] == 1
    assert traces[12]["ai_mode"] == "RECOVER_FUEL"
    assert traces[12]["ai_mode_state"] == "SENSE"
    assert traces[15]["ai_mode"] == "RECOVER_FUEL"
    assert traces[15]["ai_mode_state"] == "PICKUP"
    assert traces[15]["behavior_reason"] == "fuel=1065"
    assert traces[16]["ai_mode"] == "HUNT"
    assert traces[16]["ai_mode_state"] == "REFRESH"
    assert traces[16]["behavior_reason"] == "find purple-9"
    assert traces[17]["command_type"] == "teleport"
    assert traces[17]["behavior_reason"] == "teleport purple-9"
    assert traces[17]["combat_target_id"] == 517
    assert traces[18]["command_type"] == "teleport"
    assert traces[18]["behavior_reason"] == "teleport purple-9"
    assert traces[18]["combat_target_id"] == 517

    _reset_replay_globals()


def test_hunt_search_confirm_kill_loop_no_longer_enters_confirm_kill() -> None:
    """Replay locks out the bogus confirm-kill transition from search teleports."""
    _reset_replay_globals()
    session = load_capture_fixture("hunt_search_confirm_kill_loop.capture_session.json")

    result = replay_session(session)
    traces = result["traces"]
    command_counts = Counter(trace["command_type"] for trace in traces)

    assert result["session_id"] == "f04f00df-721f-430d-81a9-fb196b70f124"
    assert result["total_ticks"] == 16
    assert result["total_messages"] == 103
    assert command_counts["map_open"] == 1
    assert command_counts["teleport"] == 15
    assert traces[0]["ai_mode"] == "HUNT"
    assert traces[0]["ai_mode_state"] == "ACQUIRE"
    assert all(trace["behavior_reason"] != "confirm_kill" for trace in traces)
    assert all(trace["ai_mode"] == "HUNT" for trace in traces)
    assert all(trace["combat_target_id"] == 517 for trace in traces[1:])
    assert all(trace["command_type"] == "teleport" for trace in traces[1:])
    assert all(trace["behavior_reason"] == "teleport purple-9" for trace in traces[1:])

    _reset_replay_globals()
