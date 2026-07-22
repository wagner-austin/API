"""Tests for sniffer world state dispatch handling of movement messages."""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.sniffer.world_state import (
    get_world_service,
    reset_world_state,
    update_world_state_from_position,
)
from tankpit_bot.sniffer.world_state_combat import drain_killed_tank_ids
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update


class TestDispatchMovement:
    """Tests for dispatch_world_state_update with movement messages."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state and hooks after each test."""
        reset_world_state()
        _test_hooks.path_exists = _test_hooks._real_path_exists
        _test_hooks.load_terrain_map = _test_hooks._real_load_terrain_map

    def test_dispatch_movement_response(self) -> None:
        """Test dispatch handles MovementResponse (0x3D) message."""
        from tankpit_bot.protocol import MovementResponseDict

        msg = MovementResponseDict(
            msg_type=0x3D,
            team=0,
            tank_id=1,
            x=150,
            y=160,
            direction=0,
            damage_state=0,
            rank=1,
            lb_score=5,
            carrying=0,
        )

        dispatch_world_state_update(get_world_service(), msg)

        self_state = get_world_service().world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None after dispatch")
        assert self_state["x"] == 150
        assert self_state["y"] == 160

    def test_dispatch_movement_response_updates_existing_self(self) -> None:
        """Test 0x3D updates position when self_state already has this tank_id."""
        from tankpit_bot.protocol import MovementResponseDict

        # First create self_state with tank_id=1
        update_world_state_from_position(100, 100)
        # Set tank_id via a first 0x3D dispatch (creates self_state)
        first = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=5,
            x=100,
            y=100,
            direction=0,
            damage_state=0,
            rank=2,
            lb_score=3,
            carrying=0,
        )
        dispatch_world_state_update(get_world_service(), first)
        self_state = get_world_service().world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should exist after first 0x3D")
        assert self_state["tank_id"] == 5

        # Second dispatch with same tank_id hits the elif branch
        second = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=5,
            x=200,
            y=210,
            direction=0,
            damage_state=0,
            rank=2,
            lb_score=3,
            carrying=0,
        )
        dispatch_world_state_update(get_world_service(), second)
        self_state = get_world_service().world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["x"] == 200
        assert self_state["y"] == 210

    # Container-path Movement dispatch tests were deleted 2026-06-19
    # along with the container MovementDict. Protocol-path 0x47
    # Movement is covered by TestDispatchProtocolMovement below.


# Container position_update dispatch tests deleted 2026-06-20 after the
# container PositionUpdate decoder was removed. 13-byte 0x2E bodies now
# all decode to 0x3D MovementResponse via the protocol tunnel, and self
# position updates flow through the MovementResponse dispatch path.


class TestDispatchProtocolMovement:
    """Tests for dispatch with protocol Movement (0x47) and Deactivation (0x41)."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_dispatch_movement_with_waypoints_updates_enemy(self) -> None:
        """Dispatch 0x47 movement with waypoints updates non-self tank position."""
        from tankpit_bot.protocol import MovementDict, TankEntryDict

        # Register enemy tank at start position
        entry = TankEntryDict(
            msg_type=0x28, team=0, tank_id=600, rank=0, damage_state=0, score=0, x=80, y=90
        )
        dispatch_world_state_update(get_world_service(), entry)

        # Movement from (80, 90) with waypoints leading to (82, 88)
        msg = MovementDict(
            msg_type=0x47,
            tank_id=600,
            start_x=80,
            start_y=90,
            direction=1,
            flag=0,
            lb_score=10,
            rank=0,
            damage_state=0,
            is_carrying=False,
            waypoints=[(82, 88)],
            path_tiles=2,
            path="ee",
        )
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        assert state["tanks"]["600"]["x"] == 82
        assert state["tanks"]["600"]["y"] == 88

    def test_dispatch_movement_no_waypoints_keeps_start(self) -> None:
        """Dispatch 0x47 movement without waypoints uses start position."""
        from tankpit_bot.protocol import MovementDict, TankEntryDict

        entry = TankEntryDict(
            msg_type=0x28, team=0, tank_id=601, rank=0, damage_state=0, score=0, x=50, y=60
        )
        dispatch_world_state_update(get_world_service(), entry)

        msg = MovementDict(
            msg_type=0x47,
            tank_id=601,
            start_x=50,
            start_y=60,
            direction=0,
            flag=0,
            lb_score=0,
            rank=0,
            damage_state=0,
            is_carrying=False,
            waypoints=[],
            path_tiles=0,
            path="",
        )
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        # Position stays at start coords since no waypoints
        assert state["tanks"]["601"]["x"] == 50
        assert state["tanks"]["601"]["y"] == 60

    def test_dispatch_movement_updates_self(self) -> None:
        """Dispatch 0x47 movement updates self to the final waypoint destination."""
        from tankpit_bot.protocol import MovementDict, MovementResponseDict

        # Create self state at (100, 100)
        first = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=10,
            x=100,
            y=100,
            direction=0,
            damage_state=0,
            rank=2,
            lb_score=5,
            carrying=0,
        )
        dispatch_world_state_update(get_world_service(), first)

        # Movement from self's position — should skip _handle_waypoint_movement
        msg = MovementDict(
            msg_type=0x47,
            tank_id=10,
            start_x=100,
            start_y=100,
            direction=1,
            flag=0,
            lb_score=5,
            rank=0,
            damage_state=0,
            is_carrying=False,
            waypoints=[(110, 110)],
            path_tiles=1,
            path="e",
        )
        dispatch_world_state_update(get_world_service(), msg)

        # Self position updated to final waypoint destination
        self_state = get_world_service().world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["x"] == 110
        assert self_state["y"] == 110

    def test_dispatch_movement_updates_self_by_tank_id_when_position_is_stale(self) -> None:
        """Self 0x47 movement uses tank_id, not stale current coordinates, to update self."""
        from tankpit_bot.protocol import MovementDict, MovementResponseDict

        first = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=10,
            x=100,
            y=100,
            direction=0,
            damage_state=0,
            rank=2,
            lb_score=5,
            carrying=0,
        )
        dispatch_world_state_update(get_world_service(), first)
        update_world_state_from_position(90, 90)

        msg = MovementDict(
            msg_type=0x47,
            tank_id=10,
            start_x=100,
            start_y=100,
            direction=1,
            flag=0,
            lb_score=5,
            rank=0,
            damage_state=0,
            is_carrying=False,
            waypoints=[(110, 110)],
            path_tiles=1,
            path="e",
        )
        dispatch_world_state_update(get_world_service(), msg)

        self_state = get_world_service().world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["x"] == 110
        assert self_state["y"] == 110

    def test_dispatch_movement_no_matching_tank(self) -> None:
        """Dispatch 0x47 movement with start position matching no tank."""
        from tankpit_bot.protocol import MovementDict, TankEntryDict

        # Tank at (50, 60) but movement starts at (200, 200)
        entry = TankEntryDict(
            msg_type=0x28, team=0, tank_id=610, rank=0, damage_state=0, score=0, x=50, y=60
        )
        dispatch_world_state_update(get_world_service(), entry)

        msg = MovementDict(
            msg_type=0x47,
            tank_id=610,
            start_x=200,
            start_y=200,
            direction=0,
            flag=0,
            lb_score=0,
            rank=0,
            damage_state=0,
            is_carrying=False,
            waypoints=[(205, 205)],
            path_tiles=1,
            path="e",
        )
        dispatch_world_state_update(get_world_service(), msg)

        # Tank position unchanged — start coords didn't match
        assert get_world_service().world_state["tanks"]["610"]["x"] == 50
        assert get_world_service().world_state["tanks"]["610"]["y"] == 60

    def test_dispatch_deactivate_marks_liveness_and_keeps_position(self) -> None:
        """Dispatch 0x41 transitions the victim to ``deactivated`` and
        preserves the death tile.

        Pre-2026-06-20 the handler invalidated position to (0, 0) as a
        sentinel. That hack is replaced by an explicit
        ``liveness == "deactivated"`` filter in ``analyze_threats``;
        the death tile is preserved so the bot can still reason about
        it (e.g. depositing fuel or planting mines on the corpse tile,
        as seen in 2026-06-20 ghost_visual capture).
        """
        from tankpit_bot.protocol import DeactivationDict, TankEntryDict

        entry = TankEntryDict(
            msg_type=0x28, team=0, tank_id=700, rank=0, damage_state=0, score=0, x=100, y=100
        )
        dispatch_world_state_update(get_world_service(), entry)
        assert get_world_service().world_state["tanks"]["700"]["x"] == 100
        assert get_world_service().world_state["tanks"]["700"]["liveness"] == "alive"

        msg = DeactivationDict(
            msg_type=0x41,
            status=0,
            victim_id=700,
            promo_eligible=False,
            killer_id=1,
            is_mine_kill=False,
        )
        dispatch_world_state_update(get_world_service(), msg)

        tank = get_world_service().world_state["tanks"]["700"]
        assert tank["liveness"] == "deactivated"
        # Death tile preserved on the tank state.
        assert tank["x"] == 100
        assert tank["y"] == 100
        assert drain_killed_tank_ids(get_world_service()) == {700}


class TestDispatchMoveResponseUpdatesSelf:
    """Tests for 0x3D move response updating existing self position."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_move_response_for_different_tank_id(self) -> None:
        """0x3D for a different tank than self skips self position update."""
        from tankpit_bot.protocol import MovementResponseDict

        # First establish self with tank_id=5
        first = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=5,
            x=100,
            y=100,
            direction=0,
            damage_state=0,
            rank=2,
            lb_score=3,
            carrying=0,
        )
        dispatch_world_state_update(get_world_service(), first)

        self_state = get_world_service().world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["tank_id"] == 5

        # Second 0x3D with different tank_id=99 — should NOT update self position
        second = MovementResponseDict(
            msg_type=0x3D,
            team=2,
            tank_id=99,
            x=200,
            y=200,
            direction=0,
            damage_state=0,
            rank=3,
            lb_score=10,
            carrying=0,
        )
        dispatch_world_state_update(get_world_service(), second)

        # Self position unchanged
        self_state = get_world_service().world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["x"] == 100
        assert self_state["y"] == 100
        # But tank 99 should be registered
        assert "99" in get_world_service().world_state["tanks"]

    def test_stationary_self_echo_records_no_walk_entry(self) -> None:
        """A 0x47 for self with no path leaves the fuel book untouched."""
        from tankpit_bot.protocol import MovementDict, MovementResponseDict

        first = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=11,
            x=100,
            y=100,
            direction=0,
            damage_state=0,
            rank=2,
            lb_score=5,
            carrying=0,
        )
        dispatch_world_state_update(get_world_service(), first)
        msg = MovementDict(
            msg_type=0x47,
            tank_id=11,
            start_x=100,
            start_y=100,
            direction=0,
            flag=0,
            lb_score=5,
            rank=0,
            damage_state=0,
            is_carrying=False,
            waypoints=[],
            path_tiles=0,
            path="",
        )
        dispatch_world_state_update(get_world_service(), msg)
        assert get_world_service().fuel_book["entries"] == []
