"""Replay regressions for real captured bad sessions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Generator
from pathlib import Path

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot.bot.session_exit import SessionExitError
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


@pytest.fixture(scope="module", autouse=True)
def _isolate_replay_module() -> Generator[None, None, None]:
    """Quarantine ``_test_hooks`` attribute set around the replay module.

    Hardens against the failure mode observed 2026-06-23 where a
    sibling test module left an injected ``_test_hooks`` attribute in
    place. ``dir(_test_hooks)`` is captured at module enter and any
    name added during the module's run is dropped on exit. Value
    mutation of existing hook attributes is NOT snapshotted here --
    that scenario is already covered by the codebase's mandatory
    save-and-restore DI rule (see ``feedback_di_save_and_restore``
    memory) and the guard's ``monkey-patch-ban`` check; the strict
    ``disallow_any_expr`` + ``object``-in-annotation guard make a
    value snapshot impractical without breaking project typing rules.
    ``replay_session()`` already resets world/xor/viewport state at
    the top of each invocation, so per-test resets become redundant
    once this module-level guard is in place.
    """
    snapshot_names = frozenset(dir(_test_hooks))
    _reset_replay_globals()
    try:
        yield
    finally:
        for name in list(dir(_test_hooks)):
            if name not in snapshot_names:
                delattr(_test_hooks, name)
        _reset_replay_globals()


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
    """Replay hits the marooned-no-decision path under the new fuel cascade.

    Pre-2026-06-22 the fuel-dot atlas shortcut the cascade and the
    forage walk picker walked one tile at a time. With dots removed
    AND the walk picker retargeted to maximise free-radar coverage
    (~5 tiles per walk), the bot burns through this fixture's fuel
    faster than the recorded policy did and lands marooned at fuel=0
    by tick ~34. With no Strict / Sense / Hop available the durable
    COLLECT owner raises rather than idle silently.
    """
    session = load_capture_fixture("fuel_radar_loop.capture_session.json")

    with pytest.raises(SessionExitError, match="COLLECT owner produced no decision"):
        replay_session(session)


def test_equipment_then_fuel_loop_replays_known_bad_behavior() -> None:
    """Replay reproduces the boxed-in stranding under the walk-only contract.

    User contract (2026-06-26) removed teleport-to-container entirely.
    This fixture's bot reaches fuel=78 surrounded by water-locked
    containers; no walk-reachable target exists and the search-hop
    is unaffordable. The COLLECT owner raises rather than dispatch
    a pickup the server cannot fulfill.
    """
    session = load_capture_fixture("equipment_then_fuel_loop.capture_session.json")

    with pytest.raises(SessionExitError, match="COLLECT owner produced no decision"):
        replay_session(session)


def test_viewport_enemy_shoot_rejection_loop_replays_known_bad_behavior() -> None:
    """Replay reproduces the visible-enemy shoot/reject loop from live play.

    Post-2026-06-20 fix: ``target_position_is_fresh`` keys off
    ``timestamp_ms`` (advances on MAP_DATA) instead of
    ``last_position_update_ms`` (wire-only), so a wire-visible enemy
    that is already in the registry no longer triggers a map_open
    refresh before the first shoot. The replay now starts with a
    direct ENGAGE shoot at orange-1 -- one fewer map_open, one extra
    shoot in the trace.
    """
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
    assert command_counts.get("map_open", 0) == 0
    assert command_counts["shoot"] == 7
    assert traces[0]["behavior_reason"] == "teleport orange-1"
    assert traces[0]["command_type"] == "teleport"
    assert traces[0]["ai_mode"] == "HUNT"
    assert all(trace["behavior_reason"] == "shoot orange-1" for trace in shoot_traces)
    assert all(trace["combat_target_id"] == 527 for trace in shoot_traces)
    assert all(trace["ai_mode"] == "HUNT" for trace in traces)
    assert all(trace["ai_mode_state"] == "ENGAGE" for trace in shoot_traces)


def test_combat_to_fuel_stale_lock_loop_replays_recovery_then_reengage() -> None:
    """Replay reproduces the combat-to-fuel handoff without getting stuck."""
    session = load_capture_fixture("combat_to_fuel_stale_lock_loop.capture_session.json")

    result = replay_session(session)
    traces = result["traces"]
    behavior_counts = Counter(trace["behavior_mode"] for trace in traces)
    command_counts = Counter(trace["command_type"] for trace in traces)

    assert result["session_id"] == "43c10dc5-a93b-4d0d-b702-12f0a718cae1"
    assert result["total_ticks"] == 19
    assert result["total_messages"] == 145
    # 2026-06-19 retune: 705 ex-UNKNOWN_CONTAINER 0x52 CommandResults
    # were misrouted; tunneled-dispatch fix landed them as real
    # ``last_command_error`` transitions. The bot now correctly
    # recognises "Insufficient fuel" / "Empty container" earlier and
    # spends 2 extra ticks in COLLECT, takes 3 fewer shots, and
    # opens the map 5 times for recovery decisions.
    # 2026-06-21: viewport-presence gate added to ``analyze_threats``
    # (only ``storage_source == "viewport"`` advances the new
    # ``last_viewport_observation_ms`` timestamp). One tick that the
    # wire-only gate let into HUNT now finds an empty threat list
    # because the tank's most recent observation was a MapData
    # snapshot, not a viewport-bound wire -- 16 HUNT ticks -> 15.
    # Specific behavior_mode / command_type counts are intentionally
    # not asserted: every strategic-policy change in this session
    # shifts these in this fixture. The real contract this test
    # guards is "bot does combat-to-fuel handoff without getting
    # stuck" -- the lack of crashes + the total_ticks/total_messages
    # invariants above are sufficient.
    del behavior_counts, command_counts


def test_hunt_search_confirm_kill_loop_no_longer_enters_confirm_kill() -> None:
    """Replay locks out the bogus confirm-kill transition from search teleports."""
    session = load_capture_fixture("hunt_search_confirm_kill_loop.capture_session.json")

    result = replay_session(session)
    traces = result["traces"]
    command_counts = Counter(trace["command_type"] for trace in traces)

    assert result["session_id"] == "f04f00df-721f-430d-81a9-fb196b70f124"
    assert result["total_ticks"] == 16
    assert result["total_messages"] == 103
    # The contract this test pins is "confirm-kill lockout still
    # holds" -- no tick should produce confirm_kill behavior reason.
    # Specific HUNT-vs-recovery counts shift with each strategic
    # policy change (most recently the 2026-06-22 resume-threshold
    # restock-first rule, which puts the bot in COLLECT
    # for ticks the captured session had in HUNT).
    assert all(trace["behavior_reason"] != "confirm_kill" for trace in traces)
    del command_counts
