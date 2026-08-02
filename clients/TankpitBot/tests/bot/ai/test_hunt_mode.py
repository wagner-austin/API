"""Tests for the durable HUNT owner: acquisition and search basics."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.hunt_mode import decide_hunt_mode
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.state.types import TankStateDict
from tests.bot.ai._support import (
    make_enemy_tank,
    make_inventory,
    make_scanned_ai_state,
    make_world,
)


def test_hunt_acquire_searches_for_enemies_when_no_target_exists() -> None:
    """HUNT acquire falls back to enemy search when no threats are visible."""
    world, self_state = make_world(fuel=800)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason_kind"] == "find_enemies"


def test_hunt_search_dispatches_map_open_not_radar_during_acquire() -> None:
    """HUNT enemy search dispatches map_open, never radar.

    Behavior-contract guard: radar reveals hidden entities (fuel /
    equipment / mines), NOT enemies; enemies arrive via the wire
    stream. With no visible threat, HUNT acquire's only fallback is
    a global map snapshot -- the viewport-edge walk was deleted
    2026-06-22 because (a) viewport shifting is OFF in this game
    configuration, so walking to an edge reveals no new ground, and
    (b) the terrain-blocked teleport variant burned fuel without
    aiming at a known enemy.
    """
    world, self_state = make_world(fuel=800, scanned=False)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 90000,
        }
    )
    inventory = make_inventory()
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason_kind"] == "find_enemies"
    assert decision["behavior"]["mode"] == "HUNT"


def test_hunt_acquire_exits_when_fresh_map_has_no_viable_targets() -> None:
    """A fresh map snapshot with no viable enemy ends the session.

    User contract (2026-07-02): when the whole-map view is current and
    no enemy passes the acquisition gates (alive, unblocked, reachable,
    affordable), the bot must not loop on map refreshes -- it exits
    with ``no_viable_targets`` so the run is analyzable.
    """
    world, self_state = make_world(fuel=800)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    with pytest.raises(SessionExitError, match="no_viable_targets"):
        decide_hunt_mode(ctx)


def test_hunt_search_does_not_enter_confirm_kill_without_target() -> None:
    """Consecutive search ticks stay in ACQUIRE, never bogus confirm-kill.

    Each tick with no visible threat dispatches another ``map_open``;
    the mode-state derivation keeps the bot in ACQUIRE until a
    threat actually shows up. CONFIRM_KILL must only fire from a
    locked combat target disappearing -- not from enemy-search churn.
    """
    world, self_state = make_world(fuel=800)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 90000,
            "last_scan_ms": 90000,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    first_decision = decide_hunt_mode(ctx)
    if first_decision["command"]["cmd_type"] != "map_open":
        raise AssertionError("expected map_open from enemy search path")

    next_ai_state = AIStateDict(
        **{
            **first_decision["updated_ai_state"],
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
        }
    )
    next_ctx = DecideCtx(world, self_state, next_ai_state, inventory, 106000, None, "")

    second_decision = decide_hunt_mode(next_ctx)

    assert second_decision["behavior"]["reason_kind"] != "confirm_kill"


def test_hunt_acquire_uses_fresh_target_position_to_close_on_target() -> None:
    """Fresh wire-sourced target position lets HUNT acquire teleport directly."""
    tanks: dict[str, TankStateDict] = {
        "50": make_enemy_tank(),
    }
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason_kind"] == "teleport_target"
    assert decision["updated_ai_state"]["combat_target_id"] == 50


def test_hunt_acquire_targets_enemy_between_break_and_resume_thresholds() -> None:
    """HUNT still acquires targets in the non-emergency reserve band."""
    tanks: dict[str, TankStateDict] = {
        "50": make_enemy_tank(),
    }
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
        }
    )
    inventory = make_inventory(default_count=30, dual_count=18)
    inventory["homing_shots"]["count"] = 23
    inventory["extra_radars"]["count"] = 19
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason_kind"] == "teleport_target"
    assert decision["updated_ai_state"]["combat_target_id"] == 50


def test_hunt_refresh_engages_visible_adjacent_locked_target() -> None:
    """Refresh state immediately engages a visible adjacent locked target."""
    tanks: dict[str, TankStateDict] = {
        "50": make_enemy_tank(x=101, y=100),
    }
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "REFRESH",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 101,
            "combat_target_y": 100,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "shoot"
    assert decision["behavior"]["reason_kind"] == "shoot_target"
    assert decision["updated_ai_state"]["last_shot_target_id"] == 50


def test_hunt_refresh_returns_close_decision_for_visible_nonadjacent_target() -> None:
    """Refresh re-teleports to a visible non-adjacent target to close distance."""
    tanks: dict[str, TankStateDict] = {
        "50": make_enemy_tank(),
    }
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "REFRESH",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 120,
            "combat_target_y": 100,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason_kind"] == "teleport_target"


def test_hunt_acquire_refuels_when_fresh_position_teleport_is_unaffordable() -> None:
    """Acquire delegates to fuel recovery when combat teleport is unaffordable.

    Run 20260611-025636: the old map-open fallback spun 115 map reopens
    without a single shot. Fuel collection is the only action that changes
    the affordability condition, so the bot refuels before re-acquiring.
    With the current viewport already swept by the bot's tile-coverage
    map the forager yields to the search hop, producing the teleport
    the regression guard expects.
    """
    tanks: dict[str, TankStateDict] = {
        "50": make_enemy_tank(x=190, y=100, name="red-50"),
    }
    world, self_state = make_world(fuel=520, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ((116, 100),))

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["updated_ai_state"]["combat_target_id"] == 50


def test_hunt_refresh_reacquires_when_locked_target_is_missing() -> None:
    """Refresh falls back to acquisition when the locked target is gone."""
    world, self_state = make_world(fuel=800)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "REFRESH",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 120,
            "combat_target_y": 100,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason_kind"] == "find_enemies"


def test_hunt_acquire_resumes_visible_locked_target() -> None:
    """ACQUIRE shoots a still-visible locked target instead of re-acquiring fresh.

    Recovery cycles (fuel + equipment) preserve ``combat_target_id``
    so HUNT can resume the same engagement after restocking. When
    ACQUIRE runs and the held lock is still in the threat list, the
    bot engages directly -- no map_open, no fresh
    ``select_new_combat_target``.
    """
    tanks: dict[str, TankStateDict] = {
        "50": make_enemy_tank(x=101, y=100),
    }
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 101,
            "combat_target_y": 100,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "shoot"
    assert decision["behavior"]["reason_kind"] == "shoot_target"
    assert decision["updated_ai_state"]["combat_target_id"] == 50


def test_hunt_acquire_resumes_visible_locked_target_with_close_when_not_adjacent() -> None:
    """ACQUIRE closes distance on a visible-but-distant locked target.

    If the lock is in the threat list but not in a cardinal-fire
    position, ACQUIRE returns a close (teleport) decision -- same as
    the REFRESH state's close branch.
    """
    tanks: dict[str, TankStateDict] = {
        "50": make_enemy_tank(),  # default position (115, 100), distance 15 from (100,100)
    }
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 115,
            "combat_target_y": 100,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason_kind"] == "teleport_target"
    assert decision["updated_ai_state"]["combat_target_id"] == 50
