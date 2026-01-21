"""Tests for sniffer world state dispatch function."""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.sniffer import (
    dispatch_world_state_update,
    reset_world_state,
    update_world_state_from_position,
    update_world_state_from_radar,
    world_state,
)
from tests.fakes import FakeTerrainMap


class TestDispatchRadar:
    """Tests for dispatch_world_state_update with radar messages."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state and hooks after each test."""
        reset_world_state()
        _test_hooks.path_exists = _test_hooks._real_path_exists
        _test_hooks.load_terrain_map = _test_hooks._real_load_terrain_map

    def test_dispatch_radar_response(self) -> None:
        """Test dispatch handles radar_response message."""
        from tankpit_bot.container import (
            RadarContainerDict,
            RadarMineDict,
            RadarResponseDict,
        )

        _test_hooks.path_exists = lambda path: False

        msg = RadarResponseDict(
            msg_type="radar_response",
            container_count=1,
            containers=[RadarContainerDict(x=100, y=100, volume=50)],
            mines=[RadarMineDict(x=110, y=110, team=0)],
        )

        dispatch_world_state_update(msg)

        assert "100,100" in world_state._world_state["containers"]
        assert "110,110" in world_state._world_state["mines"]

    def test_dispatch_radar_response_renders_ascii(self) -> None:
        """Test dispatch renders ASCII after radar update when terrain available."""
        from tankpit_bot.container import RadarContainerDict, RadarResponseDict

        fake_terrain = FakeTerrainMap()
        _test_hooks.path_exists = lambda path: True
        _test_hooks.load_terrain_map = lambda path: fake_terrain

        update_world_state_from_position(128, 128)

        msg = RadarResponseDict(
            msg_type="radar_response",
            container_count=1,
            containers=[RadarContainerDict(x=128, y=128, volume=50)],
            mines=[],
        )

        dispatch_world_state_update(msg)

    def test_dispatch_handles_empty_radar_response(self) -> None:
        """Test dispatch handles radar_response with empty data."""
        from tankpit_bot.container import RadarResponseDict

        _test_hooks.path_exists = lambda path: False

        msg: RadarResponseDict = {
            "msg_type": "radar_response",
            "container_count": 0,
            "containers": [],
            "mines": [],
        }

        dispatch_world_state_update(msg)


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
            rank=1,
            leaderboard_position=5,
        )

        dispatch_world_state_update(msg)

        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None after dispatch")
        assert self_state["x"] == 150
        assert self_state["y"] == 160

    def test_dispatch_movement_updates_self_position(self) -> None:
        """Test dispatch handles movement message for self.

        Movement messages have absolute start_x, start_y coordinates.
        Final position is calculated by applying waypoints to start position.
        Path: eeeessssssseeeeeeeeennnnnnn = 4e + 7s + 9e + 7n
        Final: (162 + 4 + 9, 111 + 7 - 7) = (175, 111)
        """
        from tankpit_bot.container import MovementDict

        msg = MovementDict(
            msg_type="movement",
            flags=0x7E,
            start_x=162,
            start_y=111,
            player_id=230446,
            tank_id=None,
            waypoints="eeeessssssseeeeeeeeennnnnnn",
            is_self=True,
        )

        dispatch_world_state_update(msg)

        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None after dispatch")
        # Final position after applying waypoints
        assert self_state["x"] == 175
        assert self_state["y"] == 111

    def test_dispatch_movement_ignores_enemy(self) -> None:
        """Test dispatch ignores movement message for enemies.

        Movement messages with is_self=False should not update self position.
        """
        from tankpit_bot.container import MovementDict

        # First set a known position
        update_world_state_from_position(100, 100)

        msg = MovementDict(
            msg_type="movement",
            flags=0x1E,
            start_x=200,
            start_y=200,
            player_id=12345,
            tank_id=None,
            waypoints="nnnn",
            is_self=False,
        )

        dispatch_world_state_update(msg)

        # Position should remain unchanged
        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["x"] == 100
        assert self_state["y"] == 100


class TestDispatchPositionUpdate:
    """Tests for dispatch_world_state_update with position update messages."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state and hooks after each test."""
        reset_world_state()
        _test_hooks.path_exists = _test_hooks._real_path_exists
        _test_hooks.load_terrain_map = _test_hooks._real_load_terrain_map

    def test_dispatch_position_update_absolute_coords_self(self) -> None:
        """Test dispatch handles position_update with absolute coords for self.

        When x or y >= 18 and flags indicate self (0x02 bit set),
        should update the self position.
        """
        from tankpit_bot.container import PositionUpdateDict

        msg = PositionUpdateDict(
            msg_type="position_update",
            flags=0x02,  # Self flag
            tank_id=638,
            x=202,
            y=149,
            extra_data=b"\x08\x03\x03\x00\x2e\x84\x00",
        )

        dispatch_world_state_update(msg)

        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None after dispatch")
        assert self_state["x"] == 202
        assert self_state["y"] == 149

    def test_dispatch_position_update_other_tank_ignored(self) -> None:
        """Test dispatch ignores position_update for other tanks.

        Position updates with flags=0x00 are for other tanks (bots, enemies)
        and should NOT update self position.
        """
        from tankpit_bot.container import PositionUpdateDict

        # First set a known position
        update_world_state_from_position(100, 100)

        msg = PositionUpdateDict(
            msg_type="position_update",
            flags=0x00,  # Other tank flag
            tank_id=539,
            x=193,
            y=150,
            extra_data=b"\x08\x03\x01\x00\x48\xe2\x00",
        )

        dispatch_world_state_update(msg)

        # Position should remain unchanged
        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["x"] == 100
        assert self_state["y"] == 100

    def test_dispatch_position_update_viewport_relative_ignored(self) -> None:
        """Test dispatch ignores position_update with viewport-relative coords.

        When both x < 18 and y < 18, they are viewport-relative coordinates
        and should NOT update the self position even with self flag.
        """
        from tankpit_bot.container import PositionUpdateDict

        # First set a known position
        update_world_state_from_position(100, 100)

        msg = PositionUpdateDict(
            msg_type="position_update",
            flags=0x02,  # Self flag
            tank_id=638,
            x=3,
            y=3,
            extra_data=b"\x00\x2e\x85\x0a\x00\x0c\x05",
        )

        dispatch_world_state_update(msg)

        # Position should remain unchanged
        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["x"] == 100
        assert self_state["y"] == 100

    def test_dispatch_position_update_viewport_relative_other_coords_ignored(self) -> None:
        """Test dispatch ignores position_update with any small viewport coords.

        Any coords where both x < 18 and y < 18 are viewport-relative,
        not just (3,3). For example, (2,3) at the left edge of viewport.
        """
        from tankpit_bot.container import PositionUpdateDict

        # First set a known position
        update_world_state_from_position(100, 100)

        msg = PositionUpdateDict(
            msg_type="position_update",
            flags=0x02,  # Self flag
            tank_id=638,
            x=2,
            y=3,
            extra_data=b"\x00" * 7,
        )

        dispatch_world_state_update(msg)

        # Position should remain unchanged
        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["x"] == 100
        assert self_state["y"] == 100


class TestDispatchOther:
    """Tests for dispatch_world_state_update with other message types."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state and hooks after each test."""
        reset_world_state()
        _test_hooks.path_exists = _test_hooks._real_path_exists
        _test_hooks.load_terrain_map = _test_hooks._real_load_terrain_map

    def test_dispatch_ignores_other_messages(self) -> None:
        """Test dispatch ignores non-radar/movement messages."""
        from tankpit_bot.protocol import SyncDict

        initial_self = world_state._world_state["self_state"]

        msg = SyncDict(msg_type=0x3F)

        dispatch_world_state_update(msg)

        assert world_state._world_state["self_state"] == initial_self

    def test_dispatch_tank_registry_container(self) -> None:
        """Test dispatch handles tank_registry container message."""
        from tankpit_bot.container import TankRegistryDict
        from tankpit_bot.sniffer import viewport

        # Set viewport_left for container calculation
        viewport._viewport_left = 200

        msg = TankRegistryDict(
            msg_type="tank_registry",
            flags=0,
            tank_id=1000,
            info_bytes=b"",
            team="red",
            tank_name="",
            military_rank=0,
            badge_count=0,
            is_bot=False,
            is_container=True,
            container_x=None,
            container_y=75,
            container_viewport_x=3,
        )
        dispatch_world_state_update(msg)

        state = world_state._world_state
        # Container x = 200 + 3 = 203
        assert "203,75" in state["containers"]

        # Reset viewport state
        viewport._viewport_left = None

    def test_dispatch_fuel_gain_message(self) -> None:
        """Test dispatch handles FuelGain (0x44) message."""
        from tankpit_bot.protocol import FuelGainDict

        # First set up a position to create self_state
        update_world_state_from_position(100, 100)

        state = world_state._world_state
        if state["self_state"] is None:
            raise AssertionError("self_state should not be None")
        initial_fuel = state["self_state"]["fuel"]

        msg = FuelGainDict(msg_type=0x44, amount=25, is_free=False)
        dispatch_world_state_update(msg)

        state = world_state._world_state
        if state["self_state"] is None:
            raise AssertionError("self_state should not be None")
        assert state["self_state"]["fuel"] == initial_fuel + 25

    def test_dispatch_fuel_deposit_message(self) -> None:
        """Test dispatch handles FuelDeposit (0x64) message."""
        from tankpit_bot.protocol import FuelDepositDict

        # First set up a position to create self_state
        update_world_state_from_position(100, 100)

        state = world_state._world_state
        if state["self_state"] is None:
            raise AssertionError("self_state should not be None")
        initial_fuel = state["self_state"]["fuel"]

        msg = FuelDepositDict(msg_type=0x64, amount=30)
        dispatch_world_state_update(msg)

        state = world_state._world_state
        if state["self_state"] is None:
            raise AssertionError("self_state should not be None")
        assert state["self_state"]["fuel"] == initial_fuel + 30

    def test_dispatch_container_pickup_message(self) -> None:
        """Test dispatch handles container_pickup message."""
        from tankpit_bot.container import (
            ContainerPickupDict,
            RadarContainerDict,
            RadarMineDict,
        )

        # First set up a position to create self_state
        update_world_state_from_position(100, 100)

        # Add a container via radar
        containers: list[RadarContainerDict] = [RadarContainerDict(x=80, y=90, volume=50)]
        mines: list[RadarMineDict] = []
        update_world_state_from_radar(containers, mines)

        state = world_state._world_state
        assert "80,90" in state["containers"]

        # Dispatch container pickup
        msg = ContainerPickupDict(msg_type="container_pickup", x=80, y=90, volume=50, is_fuel=True)
        dispatch_world_state_update(msg)

        state = world_state._world_state
        # Container should be removed
        assert "80,90" not in state["containers"]
