"""Tests for sniffer world state dispatch function."""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.container import WorldStateDict as WorldStateBlobDict
from tankpit_bot.sniffer import (
    dispatch_world_state_update,
    reset_world_state,
    update_world_state_from_fuel_total,
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
            rank=2,
            leaderboard_position=3,
        )
        dispatch_world_state_update(first)
        if world_state._world_state["self_state"] is None:
            raise AssertionError("self_state should exist after first 0x3D")
        assert world_state._world_state["self_state"]["tank_id"] == 5

        # Second dispatch with same tank_id hits the elif branch
        second = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=5,
            x=200,
            y=210,
            direction=0,
            rank=2,
            leaderboard_position=3,
        )
        dispatch_world_state_update(second)
        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["x"] == 200
        assert self_state["y"] == 210

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
            tank_y=None,
            tank_viewport_x=None,
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

        msg = FuelGainDict(msg_type=0x44, fuel_total=25, is_free=False)
        dispatch_world_state_update(msg)

        state = world_state._world_state
        if state["self_state"] is None:
            raise AssertionError("self_state should not be None")
        # fuel_total is an absolute value, not a delta
        assert state["self_state"]["fuel"] == 25

    def test_dispatch_fuel_deposit_message(self) -> None:
        """Test dispatch handles FuelDeposit (0x64) message."""
        from tankpit_bot.protocol import FuelDepositDict

        # First set up a position to create self_state
        update_world_state_from_position(100, 100)

        msg = FuelDepositDict(msg_type=0x64, fuel_total=30)
        dispatch_world_state_update(msg)

        state = world_state._world_state
        if state["self_state"] is None:
            raise AssertionError("self_state should not be None")
        # fuel_total is an absolute value, not a delta
        assert state["self_state"]["fuel"] == 30

    def test_dispatch_tank_status_sync_with_fuel(self) -> None:
        """Test dispatch handles TankStatusSync (0x2E) with fuel field."""
        from tankpit_bot.protocol import TankStatusSyncDict

        # First set up a position to create self_state
        update_world_state_from_position(100, 100)

        msg = TankStatusSyncDict(
            msg_type=0x2E,
            subtype=3,
            tank_id=1227,
            damage_state=2,
            rank=4,
            flags=b"\x00\x22\x84",
            leaderboard_position=8,
            fuel=1400,
        )
        dispatch_world_state_update(msg)

        state = world_state._world_state
        if state["self_state"] is None:
            raise AssertionError("self_state should not be None")
        assert state["self_state"]["fuel"] == 1400

    def test_dispatch_tank_status_sync_without_fuel(self) -> None:
        """Test dispatch ignores TankStatusSync (0x2E) without fuel (short format)."""
        from tankpit_bot.protocol import TankStatusSyncDict

        update_world_state_from_position(100, 100)
        # Set fuel to a known value first
        update_world_state_from_fuel_total(500)

        msg = TankStatusSyncDict(
            msg_type=0x2E,
            subtype=1,
            tank_id=1227,
            damage_state=2,
            rank=4,
            flags=b"\x00\x22\x84",
            leaderboard_position=0,
            fuel=None,
        )
        dispatch_world_state_update(msg)

        state = world_state._world_state
        if state["self_state"] is None:
            raise AssertionError("self_state should not be None")
        # Fuel unchanged — short-format TankStatusSync has no fuel
        assert state["self_state"]["fuel"] == 500

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


class TestDispatchTankMessages:
    """Tests for dispatch_world_state_update with tank messages."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_dispatch_tank_entry(self) -> None:
        """Test dispatch handles TankEntry (0x28) message."""
        from tankpit_bot.protocol import TankEntryDict

        msg = TankEntryDict(msg_type=0x28, tank_id=42, x=100, y=150, name="EnemyBot")
        dispatch_world_state_update(msg)

        state = world_state._world_state
        assert "42" in state["tanks"]
        assert state["tanks"]["42"]["name"] == "EnemyBot"
        assert state["tanks"]["42"]["x"] == 100
        assert state["tanks"]["42"]["y"] == 150

    def test_dispatch_tank_status(self) -> None:
        """Test dispatch handles TankStatus (0x3E) message."""
        from tankpit_bot.protocol import TankStatusDict

        msg = TankStatusDict(
            msg_type=0x3E,
            team=2,
            rank=5,
            tank_id=99,
            decoration_state=b"\x00",
            leaderboard_score=1000,
            leaderboard_position=3,
            name="TopPlayer",
        )
        dispatch_world_state_update(msg)

        state = world_state._world_state
        assert "99" in state["tanks"]
        assert state["tanks"]["99"]["name"] == "TopPlayer"
        assert state["tanks"]["99"]["team"] == 2
        assert state["tanks"]["99"]["rank"] == 5

    def test_dispatch_tank_exit(self) -> None:
        """Test dispatch handles TankExit (0x58) message."""
        from tankpit_bot.protocol import TankEntryDict, TankExitDict

        # First add a tank
        entry_msg = TankEntryDict(msg_type=0x28, tank_id=42, x=100, y=150, name="LeavingBot")
        dispatch_world_state_update(entry_msg)
        assert "42" in world_state._world_state["tanks"]

        # Then remove it
        exit_msg = TankExitDict(msg_type=0x58, tank_id=42)
        dispatch_world_state_update(exit_msg)
        assert "42" not in world_state._world_state["tanks"]

    def test_dispatch_tank_registry_non_container(self) -> None:
        """Test dispatch handles tank_registry for actual tanks (not containers)."""
        from tankpit_bot.container import TankRegistryDict
        from tankpit_bot.sniffer import viewport

        # Set viewport for absolute x calculation
        viewport._viewport_left = 50

        msg = TankRegistryDict(
            msg_type="tank_registry",
            flags=0x01,
            tank_id=7,
            info_bytes=b"\x00\x00\x00\x00",
            team="blue",
            tank_name="ScoutBot",
            military_rank=3,
            badge_count=1,
            is_bot=True,
            is_container=False,
            container_x=None,
            container_y=None,
            container_viewport_x=None,
            tank_y=120,
            tank_viewport_x=5,
        )
        dispatch_world_state_update(msg)

        state = world_state._world_state
        assert "7" in state["tanks"]
        assert state["tanks"]["7"]["name"] == "ScoutBot"
        # x = viewport_left(50) + tank_viewport_x(5) = 55
        assert state["tanks"]["7"]["x"] == 55
        assert state["tanks"]["7"]["y"] == 120

        viewport._viewport_left = None

    def test_dispatch_tank_registry_non_container_no_position(self) -> None:
        """Test dispatch handles tank_registry with None position (short info_bytes)."""
        from tankpit_bot.container import TankRegistryDict

        msg = TankRegistryDict(
            msg_type="tank_registry",
            flags=0x01,
            tank_id=8,
            info_bytes=b"\x00\x00\x00\x00",
            team="red",
            tank_name="ShortBot",
            military_rank=2,
            badge_count=0,
            is_bot=False,
            is_container=False,
            container_x=None,
            container_y=None,
            container_viewport_x=None,
            tank_y=None,
            tank_viewport_x=None,
        )
        dispatch_world_state_update(msg)

        # Tank should NOT be added since position is None (match falls through)
        state = world_state._world_state
        assert "8" not in state["tanks"]

    def test_dispatch_tank_update_compact_sets_position(self) -> None:
        """Test dispatch handles tank_update_compact and extracts x,y from status_data."""
        from tankpit_bot.container import TankUpdateCompactDict

        msg = TankUpdateCompactDict(
            msg_type="tank_update_compact",
            flags=0x44,
            tank_id=200,
            status_data=bytes([82, 26, 0x2B, 0x9B, 0xF7, 0x8B]),
        )
        dispatch_world_state_update(msg)

        state = world_state._world_state
        assert "200" in state["tanks"]
        assert state["tanks"]["200"]["x"] == 82
        assert state["tanks"]["200"]["y"] == 26

    def test_dispatch_tank_update_extended_sets_position(self) -> None:
        """Test dispatch handles tank_update_extended and extracts x,y from status_data."""
        from tankpit_bot.container import TankUpdateExtendedDict

        msg = TankUpdateExtendedDict(
            msg_type="tank_update_extended",
            flags=0x44,
            tank_id=201,
            status_data=bytes([110, 55, 0, 0x1B, 0x11, 0x87, 0x9A, 0x3C, 0x24, 0x79]),
        )
        dispatch_world_state_update(msg)

        state = world_state._world_state
        assert "201" in state["tanks"]
        assert state["tanks"]["201"]["x"] == 110
        assert state["tanks"]["201"]["y"] == 55

    def test_dispatch_tank_update_full_sets_position(self) -> None:
        """Test dispatch handles tank_update_full and extracts x,y from status_data."""
        from tankpit_bot.container import TankUpdateFullDict

        msg = TankUpdateFullDict(
            msg_type="tank_update_full",
            flags=0x46,
            tank_id=202,
            status_data=bytes([84, 26, 0, 0x1B, 0x11, 0x87, 0x1C, 0x59, 0x64, 0x25, 0x25]),
        )
        dispatch_world_state_update(msg)

        state = world_state._world_state
        assert "202" in state["tanks"]
        assert state["tanks"]["202"]["x"] == 84
        assert state["tanks"]["202"]["y"] == 26

    def test_dispatch_tank_update_compact_short_status_data(self) -> None:
        """Test dispatch handles tank_update_compact with too-short status_data (< 2 bytes)."""
        from tankpit_bot.container import TankUpdateCompactDict

        msg = TankUpdateCompactDict(
            msg_type="tank_update_compact",
            flags=0x44,
            tank_id=203,
            status_data=bytes([0x01]),
        )
        dispatch_world_state_update(msg)

        # Tank should NOT be created since status_data too short for position
        state = world_state._world_state
        assert "203" not in state["tanks"]

    def test_dispatch_tank_status_short_updates_damage(self) -> None:
        """Test dispatch handles tank_status_short by updating damage."""
        from tankpit_bot.container import TankStatusShortDict
        from tankpit_bot.protocol import TankEntryDict

        # First create a tank
        entry = TankEntryDict(msg_type=0x28, tank_id=300, x=50, y=60, name="Target")
        dispatch_world_state_update(entry)

        msg = TankStatusShortDict(
            msg_type="tank_status_short",
            flags=0x82,
            tank_id=300,
            damage_state=3,
            rank=4,
            leaderboard_position=21,
        )
        dispatch_world_state_update(msg)

        state = world_state._world_state
        assert "300" in state["tanks"]
        assert state["tanks"]["300"]["damage_state"] == 3

    def test_dispatch_tank_leave_removes_tank(self) -> None:
        """Test dispatch handles tank_leave by removing the tank."""
        from tankpit_bot.container import TankLeaveDict
        from tankpit_bot.protocol import TankEntryDict

        # First create a tank
        entry = TankEntryDict(msg_type=0x28, tank_id=400, x=50, y=60, name="Leaving")
        dispatch_world_state_update(entry)
        assert "400" in world_state._world_state["tanks"]

        msg = TankLeaveDict(
            msg_type="tank_leave",
            tank_id=400,
            flags=0x13,
            extra_data=b"\x42\x13",
        )
        dispatch_world_state_update(msg)

        assert "400" not in world_state._world_state["tanks"]

    def test_dispatch_position_update_other_tank_updates_position(self) -> None:
        """Test dispatch updates enemy tank position from non-self position_update."""
        from tankpit_bot.container import PositionUpdateDict

        msg = PositionUpdateDict(
            msg_type="position_update",
            flags=0x00,  # Other tank flag
            tank_id=539,
            x=193,
            y=150,
            extra_data=b"\x08\x03\x01\x00\x48\xe2\x00",
        )
        dispatch_world_state_update(msg)

        state = world_state._world_state
        assert "539" in state["tanks"]
        assert state["tanks"]["539"]["x"] == 193
        assert state["tanks"]["539"]["y"] == 150

    def test_dispatch_enemy_movement_with_resolved_player_id(self) -> None:
        """Test dispatch updates enemy position from movement with resolved player_id."""
        from tankpit_bot.container import MovementDict
        from tankpit_bot.sniffer.player_tracking import _player_id_mapper

        # Register a player_id -> tank_id mapping
        _player_id_mapper._player_to_tank[99999] = 550

        msg = MovementDict(
            msg_type="movement",
            flags=0x1E,
            start_x=100,
            start_y=80,
            player_id=99999,
            tank_id=None,
            waypoints="eeeesss",
            is_self=False,
        )
        dispatch_world_state_update(msg)

        state = world_state._world_state
        assert "550" in state["tanks"]
        # Final position: (100+4, 80+3) = (104, 83)
        assert state["tanks"]["550"]["x"] == 104
        assert state["tanks"]["550"]["y"] == 83

        # Clean up
        _player_id_mapper._player_to_tank.clear()
        _player_id_mapper._position_to_tank.clear()


class TestWorldStateBlobParsing:
    """Tests for world_state blob parsing (map response tank positions)."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    @staticmethod
    def _make_msg(blob: bytes) -> WorldStateBlobDict:
        """Wrap blob in a properly typed world_state dispatch message."""
        return WorldStateBlobDict(
            msg_type="world_state",
            subtype=0,
            length=len(blob),
            world_data=blob,
        )

    def _build_world_state_blob(
        self,
        terrain_count: int,
        tank_entries: list[tuple[int, int, int, int, int]],
    ) -> bytes:
        """Build a world_state blob with terrain and tank entries.

        Args:
            terrain_count: Number of terrain delta bytes.
            tank_entries: List of (x, y, tank_id, team, rank) tuples.

        Returns:
            Raw bytes matching the verified format.
        """
        # 2-byte LE terrain count
        data = bytearray(terrain_count.to_bytes(2, "little"))
        # Terrain delta bytes (zeros as placeholder)
        data.extend(b"\x00" * terrain_count)
        # 5-byte tank entries: [x][y][id_lo][id_hi][packed]
        for x, y, tank_id, team, rank in tank_entries:
            id_lo = tank_id & 0xFF
            id_hi = (tank_id >> 8) & 0xFF
            packed = (team & 0x03) | ((rank & 0x0F) << 4)
            data.extend(bytes([x, y, id_lo, id_hi, packed]))
        return bytes(data)

    def test_parses_tank_positions_from_blob(self) -> None:
        """Blob with 3 tanks populates world state with correct positions."""
        blob = self._build_world_state_blob(
            terrain_count=10,
            tank_entries=[
                (100, 120, 500, 1, 2),  # red, corporal
                (200, 50, 501, 2, 0),  # blue, recruit
                (134, 121, 1229, 3, 0),  # purple, recruit (our bot)
            ],
        )
        dispatch_world_state_update(self._make_msg(blob))

        state = world_state._world_state
        assert "500" in state["tanks"]
        assert state["tanks"]["500"]["x"] == 100
        assert state["tanks"]["500"]["y"] == 120
        assert state["tanks"]["500"]["team"] == 1
        assert state["tanks"]["500"]["rank"] == 2

        assert "501" in state["tanks"]
        assert state["tanks"]["501"]["x"] == 200
        assert state["tanks"]["501"]["y"] == 50
        assert state["tanks"]["501"]["team"] == 2

        assert "1229" in state["tanks"]
        assert state["tanks"]["1229"]["x"] == 134
        assert state["tanks"]["1229"]["y"] == 121
        assert state["tanks"]["1229"]["team"] == 3

    def test_preserves_existing_tank_names(self) -> None:
        """Blob update preserves existing name and is_bot fields."""
        from tankpit_bot.sniffer.world_state import update_world_state_from_tank_info

        update_world_state_from_tank_info(500, team=1, name="EnemyBot")

        blob = self._build_world_state_blob(
            terrain_count=5,
            tank_entries=[(150, 80, 500, 1, 3)],
        )
        dispatch_world_state_update(self._make_msg(blob))

        state = world_state._world_state
        assert state["tanks"]["500"]["name"] == "EnemyBot"
        assert state["tanks"]["500"]["x"] == 150
        assert state["tanks"]["500"]["y"] == 80

    def test_empty_blob_no_crash(self) -> None:
        """Too-short blob is handled gracefully."""
        dispatch_world_state_update(self._make_msg(b"\x00"))

        state = world_state._world_state
        assert len(state["tanks"]) == 0

    def test_zero_tanks_after_terrain(self) -> None:
        """Blob with terrain but no tank entries is handled gracefully."""
        blob = self._build_world_state_blob(terrain_count=50, tank_entries=[])
        dispatch_world_state_update(self._make_msg(blob))

        state = world_state._world_state
        assert len(state["tanks"]) == 0

    def test_large_terrain_count(self) -> None:
        """Blob with large terrain (694 bytes) + tanks parses correctly."""
        blob = self._build_world_state_blob(
            terrain_count=694,
            tank_entries=[
                (9, 33, 504, 0, 1),  # red, private
                (134, 121, 1229, 3, 0),  # purple, recruit
            ],
        )
        dispatch_world_state_update(self._make_msg(blob))

        state = world_state._world_state
        assert "504" in state["tanks"]
        assert state["tanks"]["504"]["x"] == 9
        assert state["tanks"]["504"]["y"] == 33
        assert state["tanks"]["504"]["team"] == 0
        assert state["tanks"]["504"]["rank"] == 1

    def test_terrain_count_exceeds_blob_length(self) -> None:
        """Blob with terrain_count larger than data is handled gracefully."""
        # terrain_count=1000 but blob only has 10 bytes total
        data = (1000).to_bytes(2, "little") + b"\x00" * 8
        dispatch_world_state_update(self._make_msg(bytes(data)))

        state = world_state._world_state
        assert len(state["tanks"]) == 0
