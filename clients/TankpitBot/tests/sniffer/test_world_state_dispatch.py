"""Tests for sniffer world state dispatch function."""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.container import RadarContainerDict
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

    def test_dispatch_radar_ack_marks_scan_complete(self) -> None:
        """Test dispatch handles radar ack messages as scan completion."""
        from tankpit_bot.protocol import RadarResultDict
        from tankpit_bot.sniffer.world_state import check_and_clear_radar_scan_complete

        msg = RadarResultDict(
            msg_type=0x46,
            detection_type=0,
            found=True,
        )

        dispatch_world_state_update(msg)

        assert check_and_clear_radar_scan_complete() is True


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
        """Tank-registry container messages do not populate resource truth."""
        from tankpit_bot.container import TankRegistryDict
        from tankpit_bot.sniffer import viewport

        viewport.update_viewport_origin(200, 0)

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
        assert "203,75" not in state["containers"]

        viewport.reset_viewport_tracking()

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

        viewport.update_viewport_origin(50, 0)

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

        viewport.reset_viewport_tracking()

    def test_dispatch_tank_registry_non_container_without_viewport_origin(self) -> None:
        """Test tank_registry tank is ignored until viewport origin is known."""
        from tankpit_bot.container import TankRegistryDict
        from tankpit_bot.sniffer import viewport

        viewport.reset_viewport_tracking()

        msg = TankRegistryDict(
            msg_type="tank_registry",
            flags=0x01,
            tank_id=9,
            info_bytes=b"\x00\x00\x00\x00",
            team="blue",
            tank_name="NoViewportBot",
            military_rank=1,
            badge_count=0,
            is_bot=False,
            is_container=False,
            container_x=None,
            container_y=None,
            container_viewport_x=None,
            tank_y=130,
            tank_viewport_x=6,
        )
        dispatch_world_state_update(msg)

        state = world_state._world_state
        assert "9" not in state["tanks"]

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

    def test_dispatch_tank_update_compact_flag_cd_does_not_set_tank_position(self) -> None:
        """Test obstacle-correlated 0xCD compact updates do not create tank positions."""
        from tankpit_bot.container import TankUpdateCompactDict

        msg = TankUpdateCompactDict(
            msg_type="tank_update_compact",
            flags=0xCD,
            tank_id=2308,
            status_data=bytes.fromhex("a50aa5000200"),
        )
        dispatch_world_state_update(msg)

        state = world_state._world_state
        assert "2308" not in state["tanks"]

    def test_dispatch_tunneled_terrain_update_sets_terrain_tile(self) -> None:
        """Test 0x4A terrain updates modify world terrain state."""
        from tankpit_bot.protocol import TerrainUpdateDict

        msg = TerrainUpdateDict(msg_type=0x4A, updates=[(8, 166, 2)])
        dispatch_world_state_update(msg)

        state = world_state._world_state
        tile = state["terrain"]["8,166"]
        assert tile["x"] == 8
        assert tile["y"] == 166
        assert tile["terrain_type"] == 2
        assert tile["cache_value"] == 0
        assert tile["overlay_value"] == 255

    def test_dispatch_tunneled_mine_placement_adds_mines(self) -> None:
        """Test tunneled 0x4B mine placement updates world mine state."""
        from tankpit_bot.protocol import MovementResponseDict

        dispatch_world_state_update(
            MovementResponseDict(
                msg_type=0x3D,
                team=2,
                tank_id=1301,
                x=131,
                y=126,
                direction=8,
                rank=1,
                leaderboard_position=1313,
            )
        )

        dispatch_world_state_update(
            {
                "msg_type": 0x4B,
                "mine_type": 2,
                "tank_id": 1301,
                "positions": [
                    (131, 126),
                    (131, 125),
                    (132, 125),
                    (132, 126),
                    (132, 127),
                ],
            }
        )

        state = world_state._world_state
        assert state["mines"]["131,126"]["team"] == 2
        assert state["mines"]["131,126"]["tank_id"] == 1301
        assert state["mines"]["131,126"]["mine_type"] == 2
        assert state["mines"]["132,127"]["x"] == 132
        assert state["mines"]["132,127"]["y"] == 127

    def test_dispatch_tunneled_mine_placement_uses_known_tank_team(self) -> None:
        """Test tunneled 0x4B uses tracked tank team when placer is not self."""
        from tankpit_bot.protocol import TankEntryDict, TankInfoDict

        dispatch_world_state_update(
            TankInfoDict(
                msg_type=0x21,
                tank_id=777,
                name="placer",
                team=3,
                decoration_state=b"",
                score=0,
            )
        )

        dispatch_world_state_update(
            TankEntryDict(
                msg_type=0x28,
                tank_id=777,
                x=40,
                y=41,
                name="placer",
            )
        )

        dispatch_world_state_update(
            {
                "msg_type": 0x4B,
                "mine_type": 1,
                "tank_id": 777,
                "positions": [(40, 41), (40, 42)],
            }
        )

        state = world_state._world_state
        assert state["mines"]["40,41"]["team"] == 3
        assert state["mines"]["40,42"]["team"] == 3
        assert state["mines"]["40,41"]["tank_id"] == 777

    def test_dispatch_tunneled_mine_placement_skips_unknown_team(self) -> None:
        """Test tunneled 0x4B does nothing when placer team is unknown."""
        dispatch_world_state_update(
            {
                "msg_type": 0x4B,
                "mine_type": 2,
                "tank_id": 9999,
                "positions": [(10, 11), (11, 11)],
            }
        )

        state = world_state._world_state
        assert state["mines"] == {}

    def test_dispatch_tunneled_mine_detonation_removes_mines(self) -> None:
        """Test tunneled 0x45 removes mines at decoded coordinates."""
        from tankpit_bot.protocol import MovementResponseDict

        dispatch_world_state_update(
            MovementResponseDict(
                msg_type=0x3D,
                team=2,
                tank_id=1301,
                x=38,
                y=53,
                direction=8,
                rank=1,
                leaderboard_position=1313,
            )
        )

        dispatch_world_state_update(
            {
                "msg_type": 0x4B,
                "mine_type": 2,
                "tank_id": 1301,
                "positions": [(38, 52), (39, 53), (38, 54)],
            }
        )

        dispatch_world_state_update({"msg_type": 0x45, "positions": [(39, 53), (38, 54)]})

        state = world_state._world_state
        assert "38,52" in state["mines"]
        assert "39,53" not in state["mines"]
        assert "38,54" not in state["mines"]

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


class TestDispatchEnemyDetection:
    """Tests for dispatch_world_state_update with EnemyDetection (0x48) messages."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_dispatch_enemy_detection_creates_tank(self) -> None:
        """Dispatch 0x48 creates enemy tank entry via _update_enemy_from_detection."""
        from tankpit_bot.protocol import EnemyDetectionDict

        msg = EnemyDetectionDict(
            msg_type=0x48,
            tank_id=555,
            x=120,
            y=130,
            rank=3,
            team=2,
        )
        dispatch_world_state_update(msg)

        state = world_state._world_state
        assert "555" in state["tanks"]
        assert state["tanks"]["555"]["x"] == 120
        assert state["tanks"]["555"]["y"] == 130
        assert state["tanks"]["555"]["team"] == 2
        assert state["tanks"]["555"]["rank"] == 3

    def test_dispatch_enemy_detection_updates_existing_tank(self) -> None:
        """Dispatch 0x48 updates position of already-registered enemy tank."""
        from tankpit_bot.protocol import EnemyDetectionDict, TankEntryDict

        # First create a tank with an old position
        entry = TankEntryDict(msg_type=0x28, tank_id=556, x=50, y=60, name="OldPos")
        dispatch_world_state_update(entry)

        # Detection updates to new position
        msg = EnemyDetectionDict(
            msg_type=0x48,
            tank_id=556,
            x=200,
            y=210,
            rank=5,
            team=1,
        )
        dispatch_world_state_update(msg)

        state = world_state._world_state
        assert state["tanks"]["556"]["x"] == 200
        assert state["tanks"]["556"]["y"] == 210


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
        entry = TankEntryDict(msg_type=0x28, tank_id=600, x=80, y=90, name="Moving")
        dispatch_world_state_update(entry)

        # Movement from (80, 90) with waypoints leading to (82, 88)
        msg = MovementDict(
            msg_type=0x47,
            tank_id=600,
            start_x=80,
            start_y=90,
            direction=1,
            flag=0,
            leaderboard_position=10,
            waypoints=[(82, 88)],
        )
        dispatch_world_state_update(msg)

        state = world_state._world_state
        assert state["tanks"]["600"]["x"] == 82
        assert state["tanks"]["600"]["y"] == 88

    def test_dispatch_movement_no_waypoints_keeps_start(self) -> None:
        """Dispatch 0x47 movement without waypoints uses start position."""
        from tankpit_bot.protocol import MovementDict, TankEntryDict

        entry = TankEntryDict(msg_type=0x28, tank_id=601, x=50, y=60, name="Stationary")
        dispatch_world_state_update(entry)

        msg = MovementDict(
            msg_type=0x47,
            tank_id=601,
            start_x=50,
            start_y=60,
            direction=0,
            flag=0,
            leaderboard_position=0,
            waypoints=[],
        )
        dispatch_world_state_update(msg)

        state = world_state._world_state
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
            rank=2,
            leaderboard_position=5,
        )
        dispatch_world_state_update(first)

        # Movement from self's position — should skip _handle_waypoint_movement
        msg = MovementDict(
            msg_type=0x47,
            tank_id=10,
            start_x=100,
            start_y=100,
            direction=1,
            flag=0,
            leaderboard_position=5,
            waypoints=[(110, 110)],
        )
        dispatch_world_state_update(msg)

        # Self position updated to final waypoint destination
        self_state = world_state._world_state["self_state"]
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
            rank=2,
            leaderboard_position=5,
        )
        dispatch_world_state_update(first)
        world_state.update_world_state_from_position(90, 90)

        msg = MovementDict(
            msg_type=0x47,
            tank_id=10,
            start_x=100,
            start_y=100,
            direction=1,
            flag=0,
            leaderboard_position=5,
            waypoints=[(110, 110)],
        )
        dispatch_world_state_update(msg)

        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["x"] == 110
        assert self_state["y"] == 110

    def test_dispatch_movement_no_matching_tank(self) -> None:
        """Dispatch 0x47 movement with start position matching no tank."""
        from tankpit_bot.protocol import MovementDict, TankEntryDict

        # Tank at (50, 60) but movement starts at (200, 200)
        entry = TankEntryDict(msg_type=0x28, tank_id=610, x=50, y=60, name="Far")
        dispatch_world_state_update(entry)

        msg = MovementDict(
            msg_type=0x47,
            tank_id=610,
            start_x=200,
            start_y=200,
            direction=0,
            flag=0,
            leaderboard_position=0,
            waypoints=[(205, 205)],
        )
        dispatch_world_state_update(msg)

        # Tank position unchanged — start coords didn't match
        assert world_state._world_state["tanks"]["610"]["x"] == 50
        assert world_state._world_state["tanks"]["610"]["y"] == 60

    def test_dispatch_deactivate_invalidates_position(self) -> None:
        """Dispatch 0x41 invalidates position and records the kill."""
        from tankpit_bot.protocol import DeactivationDict, TankEntryDict
        from tankpit_bot.sniffer.world_state import drain_killed_tank_ids

        entry = TankEntryDict(msg_type=0x28, tank_id=700, x=100, y=100, name="Victim")
        dispatch_world_state_update(entry)
        assert world_state._world_state["tanks"]["700"]["x"] == 100

        msg = DeactivationDict(
            msg_type=0x41,
            victim_id=700,
            killer_id=1,
            rank=2,
            points=100,
        )
        dispatch_world_state_update(msg)

        # Position invalidated to (0, 0)
        assert world_state._world_state["tanks"]["700"]["x"] == 0
        assert world_state._world_state["tanks"]["700"]["y"] == 0
        assert drain_killed_tank_ids() == {700}


class TestDispatchViewportUpdate:
    """Tests for dispatch with ViewportUpdate (0x5A)."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_dispatch_viewport_update_invalidates_absent_tanks(self) -> None:
        """Viewport update does not invalidate tanks absent from ``0x5A``."""
        from tankpit_bot.protocol import (
            MovementResponseDict,
            TankEntryDict,
            ViewportUpdateDict,
        )

        # Create self at (55, 55)
        self_msg = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=900,
            x=55,
            y=55,
            direction=0,
            rank=2,
            leaderboard_position=5,
        )
        dispatch_world_state_update(self_msg)

        # Place enemy tank at (58, 58) — inside viewport
        entry = TankEntryDict(msg_type=0x28, tank_id=800, x=58, y=58, name="Ghost")
        dispatch_world_state_update(entry)
        assert world_state._world_state["tanks"]["800"]["x"] == 58

        # Viewport update with only self → buffers entities
        msg = ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=46,
            viewport_top=46,
            entities=[],
        )
        dispatch_world_state_update(msg)

        # Tank 800 invalidated — in viewport but not in entity list
        assert world_state._world_state["tanks"]["800"]["x"] == 58
        assert world_state._world_state["tanks"]["800"]["y"] == 58

    def test_dispatch_viewport_update_skips_empty_cache_rows(self) -> None:
        """Viewport rows with cache_value=0 do not affect tracked tank positions."""
        from tankpit_bot.protocol import (
            MovementResponseDict,
            TankEntryDict,
            ViewportUpdateDict,
        )
        from tankpit_bot.protocol.types import ViewportEntityDict

        # Create self at (55, 55)
        self_msg = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=900,
            x=55,
            y=55,
            direction=0,
            rank=2,
            leaderboard_position=5,
        )
        dispatch_world_state_update(self_msg)

        # Place enemy at (58, 58) — inside viewport
        entry = TankEntryDict(msg_type=0x28, tank_id=815, x=58, y=58, name="Target")
        dispatch_world_state_update(entry)

        # Viewport update with cache_value=0 (no container cache)
        entity_zero = ViewportEntityDict(
            col=5,
            row=5,
            cache_value=0,
            overlay_value=0,
            terrain_type=0,
        )
        msg = ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=46,
            viewport_top=46,
            entities=[entity_zero],
        )
        dispatch_world_state_update(msg)

        # Tank position is unchanged because viewport cache rows do not track tanks
        assert world_state._world_state["tanks"]["815"]["x"] == 58
        assert world_state._world_state["tanks"]["815"]["y"] == 58

    def test_dispatch_viewport_update_keeps_visible_tank(self) -> None:
        """Positive ``0x5A`` rows do not overwrite tracked tank positions."""
        from tankpit_bot.protocol import TankEntryDict, ViewportUpdateDict
        from tankpit_bot.protocol.types import ViewportEntityDict
        from tankpit_bot.state.types import ViewportStateDict

        world_state._world_state["viewport"] = ViewportStateDict(
            left=50,
            top=50,
            width=18,
            height=18,
        )

        # Tank 810 in viewport at (55, 55)
        entry = TankEntryDict(msg_type=0x28, tank_id=810, x=55, y=55, name="Visible")
        dispatch_world_state_update(entry)

        # Viewport update with unrelated cache row leaves tracked tank untouched
        entity = ViewportEntityDict(
            col=5,
            row=5,
            cache_value=810,
            overlay_value=0,
            terrain_type=0,
        )
        msg = ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=50,
            viewport_top=50,
            entities=[entity],
        )
        dispatch_world_state_update(msg)

        # Tank position preserved
        assert world_state._world_state["tanks"]["810"]["x"] == 55
        assert world_state._world_state["tanks"]["810"]["y"] == 55

    def test_dispatch_viewport_update_skips_self_and_zeroed_tanks(self) -> None:
        """Dispatch 0x5A skips self tank and already-invalidated (0,0) tanks."""
        from tankpit_bot.protocol import (
            MovementResponseDict,
            TankEntryDict,
            ViewportUpdateDict,
        )
        from tankpit_bot.state.types import ViewportStateDict

        world_state._world_state["viewport"] = ViewportStateDict(
            left=50,
            top=50,
            width=18,
            height=18,
        )

        # Create self at (55, 55) — inside viewport
        self_msg = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=820,
            x=55,
            y=55,
            direction=0,
            rank=2,
            leaderboard_position=5,
        )
        dispatch_world_state_update(self_msg)

        # Create enemy already at (0, 0) — already invalidated
        entry = TankEntryDict(msg_type=0x28, tank_id=821, x=0, y=0, name="Zeroed")
        dispatch_world_state_update(entry)

        # Viewport update with no entities — should skip self and zeroed tank
        msg = ViewportUpdateDict(msg_type=0x5A, viewport_left=50, viewport_top=50, entities=[])
        dispatch_world_state_update(msg)

        # Self position preserved (skipped because is_self)
        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["x"] == 55

    def test_dispatch_viewport_update_initializes_from_packet_origin(self) -> None:
        """Dispatch 0x5A uses packet origin even when prior viewport is default."""
        from tankpit_bot.protocol import TankEntryDict, ViewportUpdateDict
        from tankpit_bot.sniffer.viewport import get_viewport_left, get_viewport_top

        # Default viewport: left=0, top=0, width=18 is still a valid origin.
        entry = TankEntryDict(msg_type=0x28, tank_id=801, x=5, y=5, name="Safe")
        dispatch_world_state_update(entry)

        msg = ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=0,
            viewport_top=0,
            entities=[],
        )
        dispatch_world_state_update(msg)

        assert world_state._world_state["viewport"]["left"] == 0
        assert world_state._world_state["viewport"]["top"] == 0
        assert get_viewport_left() == 0
        assert get_viewport_top() == 0
        assert world_state._world_state["tanks"]["801"]["x"] == 5
        assert world_state._world_state["tanks"]["801"]["y"] == 5

    def test_dispatch_viewport_update_does_not_mark_viewport_confirmed(self) -> None:
        """Dispatch 0x5A alone should not count as a radar-confirmed scan."""
        from tankpit_bot.protocol import ViewportUpdateDict

        msg = ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=51,
            viewport_top=29,
            entities=[],
        )

        dispatch_world_state_update(msg)

        assert "51,29" not in world_state._world_state["scanned_viewports"]

    def test_dispatch_viewport_update_clears_failed_scan_mark(self) -> None:
        """Fresh visible viewport data clears a recent failed-scan mark."""
        from tankpit_bot.protocol import ViewportUpdateDict
        from tankpit_bot.sniffer.world_state import (
            is_scan_viewport_failed,
            mark_scan_viewport_failed,
        )

        mark_scan_viewport_failed(51, 29, 1000)
        assert is_scan_viewport_failed(51, 29, 1001) is True

        msg = ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=51,
            viewport_top=29,
            entities=[],
        )
        dispatch_world_state_update(msg)

        assert is_scan_viewport_failed(51, 29, 1001) is False


class TestViewportContainerExtraction:
    """Tests for fuel/equipment container extraction from viewport entities."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_fuel_cache_from_viewport_entity_updates_terrain_only(self) -> None:
        """Viewport entity fuel cache updates terrain only until radar confirms it."""
        from tankpit_bot.protocol import MovementResponseDict, ViewportUpdateDict
        from tankpit_bot.protocol.types import ViewportEntityDict

        self_msg = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=1229,
            x=60,
            y=38,
            direction=0,
            rank=1,
            leaderboard_position=5,
        )
        dispatch_world_state_update(self_msg)

        fuel_ent = ViewportEntityDict(
            col=15,
            row=3,
            cache_value=994,
            overlay_value=255,
            terrain_type=0,
        )
        msg = ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=51,
            viewport_top=29,
            entities=[fuel_ent],
        )
        dispatch_world_state_update(msg)

        # Fuel container at abs (51+15-1, 29+3-1) = (65, 31) with volume ≈ 994
        tile = world_state._world_state["terrain"]["65,31"]
        assert tile["cache_value"] == 994
        assert "65,31" not in world_state._world_state["containers"]

    def test_equipment_cache_from_viewport_entity_updates_terrain_only(self) -> None:
        """Viewport entity equipment cache updates terrain only until radar confirms it."""
        from tankpit_bot.protocol import MovementResponseDict, ViewportUpdateDict
        from tankpit_bot.protocol.types import ViewportEntityDict

        self_msg = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=1229,
            x=60,
            y=38,
            direction=0,
            rank=1,
            leaderboard_position=5,
        )
        dispatch_world_state_update(self_msg)

        equip_ent = ViewportEntityDict(
            col=4,
            row=11,
            cache_value=-1,
            overlay_value=255,
            terrain_type=0,
        )
        msg = ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=51,
            viewport_top=29,
            entities=[equip_ent],
        )
        dispatch_world_state_update(msg)

        # Equipment container at abs (51+4-1, 29+11-1) = (54, 39)
        tile = world_state._world_state["terrain"]["54,39"]
        assert tile["cache_value"] == -1
        assert "54,39" not in world_state._world_state["containers"]

    def test_positive_cache_value_does_not_create_fuel_container(self) -> None:
        """Positive cache values remain visual hints until radar confirms them."""
        from tankpit_bot.protocol import MovementResponseDict, ViewportUpdateDict
        from tankpit_bot.protocol.types import ViewportEntityDict

        self_msg = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=1229,
            x=60,
            y=38,
            direction=0,
            rank=1,
            leaderboard_position=5,
        )
        dispatch_world_state_update(self_msg)

        # Positive cache_value — treated as fuel container
        unknown_ent = ViewportEntityDict(
            col=5,
            row=5,
            cache_value=999,
            overlay_value=255,
            terrain_type=0,
        )
        msg = ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=51,
            viewport_top=29,
            entities=[unknown_ent],
        )
        dispatch_world_state_update(msg)

        # Viewport offset = (60-9, 38-9) = (51, 29). Abs pos = (51+5-1, 29+5-1) = (55, 33).
        tile = world_state._world_state["terrain"]["55,33"]
        assert tile["cache_value"] == 999
        assert "55,33" not in world_state._world_state["containers"]

    def test_empty_viewport_cache_does_not_override_radar_container(self) -> None:
        """A 0x5A cache clear does not override radar-confirmed container truth."""
        from tankpit_bot.container import RadarContainerDict
        from tankpit_bot.protocol import MovementResponseDict, ViewportUpdateDict
        from tankpit_bot.protocol.types import ViewportEntityDict
        from tankpit_bot.sniffer.world_state import update_world_state_from_radar

        dispatch_world_state_update(
            MovementResponseDict(
                msg_type=0x3D,
                team=1,
                tank_id=1229,
                x=60,
                y=38,
                direction=0,
                rank=1,
                leaderboard_position=5,
            )
        )
        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        self_state["fuel"] = 400

        update_world_state_from_radar([RadarContainerDict(x=55, y=33, volume=900)], [])

        assert "55,33" in world_state._world_state["containers"]

        dispatch_world_state_update(
            ViewportUpdateDict(
                msg_type=0x5A,
                viewport_left=51,
                viewport_top=29,
                entities=[
                    ViewportEntityDict(
                        col=5,
                        row=5,
                        cache_value=0,
                        overlay_value=255,
                        terrain_type=0,
                    )
                ],
            )
        )

        assert world_state._world_state["terrain"]["55,33"]["cache_value"] == 0
        assert "55,33" in world_state._world_state["containers"]
        assert world_state._world_state["containers"]["55,33"]["volume"] == 900
        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["fuel"] == 400


class TestViewportInvalidationEdgeCases:
    """Tests for viewport entity processing edge cases."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_viewport_update_applies_without_self_state(self) -> None:
        """Viewport origin updates even before self movement is known."""
        from tankpit_bot.protocol import ViewportUpdateDict

        dispatch_world_state_update(
            ViewportUpdateDict(
                msg_type=0x5A,
                viewport_left=50,
                viewport_top=50,
                entities=[],
            )
        )

        assert world_state._world_state["viewport"]["left"] == 50
        assert world_state._world_state["viewport"]["top"] == 50

    def test_invalidation_skips_already_zeroed_tanks(self) -> None:
        """Tanks already at (0,0) are not re-invalidated."""
        from tankpit_bot.protocol import MovementResponseDict, TankEntryDict, ViewportUpdateDict
        from tankpit_bot.protocol.types import ViewportEntityDict

        # Create self at (55, 55)
        self_msg = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=950,
            x=55,
            y=55,
            direction=0,
            rank=2,
            leaderboard_position=5,
        )
        dispatch_world_state_update(self_msg)

        # Place enemy already at (0,0) — already invalidated
        entry = TankEntryDict(msg_type=0x28, tank_id=951, x=0, y=0, name="Zeroed")
        dispatch_world_state_update(entry)

        # Viewport update with only self
        self_ent = ViewportEntityDict(
            col=9,
            row=9,
            cache_value=950,
            overlay_value=255,
            terrain_type=0,
        )
        msg = ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=46,
            viewport_top=46,
            entities=[self_ent],
        )
        dispatch_world_state_update(msg)

        # Zeroed tank stays at (0,0) — was already invalidated, just skipped
        assert world_state._world_state["tanks"]["951"]["x"] == 0
        assert world_state._world_state["tanks"]["951"]["y"] == 0

    def test_invalidation_skips_tanks_outside_viewport(self) -> None:
        """Tanks outside the viewport bounds are not invalidated."""
        from tankpit_bot.protocol import MovementResponseDict, TankEntryDict, ViewportUpdateDict
        from tankpit_bot.protocol.types import ViewportEntityDict

        # Create self at (55, 55) → viewport is (46, 46) to (64, 64)
        self_msg = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=960,
            x=55,
            y=55,
            direction=0,
            rank=2,
            leaderboard_position=5,
        )
        dispatch_world_state_update(self_msg)

        # Place enemy far away at (200, 200) — outside viewport
        entry = TankEntryDict(msg_type=0x28, tank_id=961, x=200, y=200, name="FarAway")
        dispatch_world_state_update(entry)

        # Viewport update with only self
        self_ent = ViewportEntityDict(
            col=9,
            row=9,
            cache_value=960,
            overlay_value=255,
            terrain_type=0,
        )
        msg = ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=46,
            viewport_top=46,
            entities=[self_ent],
        )
        dispatch_world_state_update(msg)

        # Far-away tank position preserved — not in viewport bounds
        assert world_state._world_state["tanks"]["961"]["x"] == 200
        assert world_state._world_state["tanks"]["961"]["y"] == 200


class TestDispatchTilePatchUpdates:
    """Tests for absolute tile patch dispatch in world state."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_dispatch_overlay_update_preserves_existing_cache_and_terrain(self) -> None:
        """Top-level 0x40 updates only the overlay layer for existing tiles."""
        from tankpit_bot.protocol import OverlayUpdateDict, TerrainUpdateDict

        dispatch_world_state_update(TerrainUpdateDict(msg_type=0x4A, updates=[(70, 80, 6)]))
        dispatch_world_state_update(OverlayUpdateDict(msg_type=0x40, updates=[(70, 80, 9)]))

        tile = world_state._world_state["terrain"]["70,80"]
        assert tile["terrain_type"] == 6
        assert tile["cache_value"] == 0
        assert tile["overlay_value"] == 9

    def test_dispatch_cache_update_updates_terrain_only(self) -> None:
        """Top-level 0x43 updates tile cache without creating targets."""
        from tankpit_bot.protocol import CacheUpdateDict

        dispatch_world_state_update(CacheUpdateDict(msg_type=0x43, updates=[(33, 44, 600)]))

        tile = world_state._world_state["terrain"]["33,44"]
        assert tile["terrain_type"] == 0
        assert tile["cache_value"] == 600
        assert tile["overlay_value"] == 255
        assert "33,44" not in world_state._world_state["containers"]

        dispatch_world_state_update(CacheUpdateDict(msg_type=0x43, updates=[(33, 44, 0)]))

        cleared_tile = world_state._world_state["terrain"]["33,44"]
        assert cleared_tile["cache_value"] == 0
        assert "33,44" not in world_state._world_state["containers"]

    def test_dispatch_cache_clear_does_not_override_radar_container(self) -> None:
        """A 0x43 cache clear does not erase radar-confirmed container truth."""
        from tankpit_bot.container import RadarContainerDict
        from tankpit_bot.protocol import CacheUpdateDict, MovementResponseDict
        from tankpit_bot.sniffer.world_state import update_world_state_from_radar

        dispatch_world_state_update(
            MovementResponseDict(
                msg_type=0x3D,
                team=1,
                tank_id=1300,
                x=33,
                y=44,
                direction=0,
                rank=1,
                leaderboard_position=5,
            )
        )
        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        self_state["fuel"] = 250

        update_world_state_from_radar([RadarContainerDict(x=33, y=44, volume=600)], [])
        dispatch_world_state_update(CacheUpdateDict(msg_type=0x43, updates=[(33, 44, 0)]))

        assert world_state._world_state["terrain"]["33,44"]["cache_value"] == 0
        assert "33,44" in world_state._world_state["containers"]
        assert world_state._world_state["containers"]["33,44"]["volume"] == 600
        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["fuel"] == 250

    def test_dispatch_combined_tile_update_applies_cache_and_overlay_sections(self) -> None:
        """Top-level 0x4F applies both cache and overlay sections visually only."""
        from tankpit_bot.protocol import CombinedTileUpdateDict, TerrainUpdateDict

        dispatch_world_state_update(TerrainUpdateDict(msg_type=0x4A, updates=[(90, 91, 4)]))
        dispatch_world_state_update(
            CombinedTileUpdateDict(
                msg_type=0x4F,
                cache_updates=[(90, 91, -1)],
                overlay_updates=[(90, 91, 12)],
            )
        )

        tile = world_state._world_state["terrain"]["90,91"]
        assert tile["terrain_type"] == 4
        assert tile["cache_value"] == -1
        assert tile["overlay_value"] == 12
        assert "90,91" not in world_state._world_state["containers"]


class TestDispatchContainerCombatEvents:
    """Tests for container combat events: combat_hit, deactivation_kill, deactivation_death."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_dispatch_combat_hit_marks_hit_for_self(self) -> None:
        """Dispatch combat_hit calls mark_combat_hit when attacker is self."""
        from tankpit_bot.container import CombatHitDict
        from tankpit_bot.protocol import MovementResponseDict

        # Set up self with tank_id=10
        first = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=10,
            x=100,
            y=100,
            direction=0,
            rank=2,
            leaderboard_position=5,
        )
        dispatch_world_state_update(first)

        # combat_hit where attacker_id matches self
        msg = CombatHitDict(
            msg_type="combat_hit",
            direction=0x09,
            attacker_id=10,
            combat_data=b"\x00\x01\x02\x03\x04\x05",
            is_outgoing=True,
        )
        dispatch_world_state_update(msg)

        # Verify mark_combat_hit was called
        assert world_state.check_and_clear_combat_hit() is True

    def test_dispatch_combat_hit_ignores_other_attacker(self) -> None:
        """Dispatch combat_hit does not mark hit when attacker is not self."""
        from tankpit_bot.container import CombatHitDict
        from tankpit_bot.protocol import MovementResponseDict

        # Set up self with tank_id=10
        first = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=10,
            x=100,
            y=100,
            direction=0,
            rank=2,
            leaderboard_position=5,
        )
        dispatch_world_state_update(first)

        # combat_hit from a different tank
        msg = CombatHitDict(
            msg_type="combat_hit",
            direction=0x03,
            attacker_id=99,
            combat_data=b"\x00\x01\x02\x03\x04\x05",
            is_outgoing=False,
        )
        dispatch_world_state_update(msg)

        # No hit recorded for self
        assert world_state.check_and_clear_combat_hit() is False

    def test_dispatch_deactivation_kill_invalidates_victim(self) -> None:
        """Dispatch deactivation_kill invalidates victim tank position."""
        from tankpit_bot.container import DeactivationKillDict
        from tankpit_bot.protocol import TankEntryDict

        entry = TankEntryDict(msg_type=0x28, tank_id=900, x=100, y=100, name="Killed")
        dispatch_world_state_update(entry)

        msg = DeactivationKillDict(
            msg_type="deactivation_kill",
            victim_id=900,
            killer_id=1,
        )
        dispatch_world_state_update(msg)

        assert world_state._world_state["tanks"]["900"]["x"] == 0
        assert world_state._world_state["tanks"]["900"]["y"] == 0

    def test_dispatch_deactivation_death_is_handled(self) -> None:
        """Dispatch deactivation_death is handled without error."""
        from tankpit_bot.container import DeactivationDeathDict

        msg = DeactivationDeathDict(
            msg_type="deactivation_death",
            flags=0,
            killer_id=42,
            extra_data=b"\x00\x01\x02",
        )
        dispatch_world_state_update(msg)  # should not raise


class TestCombatHitTracking:
    """Tests for mark_combat_hit and check_and_clear_combat_hit."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_check_and_clear_returns_false_by_default(self) -> None:
        """check_and_clear_combat_hit returns False when no hit recorded."""
        assert world_state.check_and_clear_combat_hit() is False

    def test_mark_and_check_returns_true(self) -> None:
        """mark_combat_hit sets flag, check_and_clear returns True then False."""
        world_state.mark_combat_hit(weapon_byte=1)
        assert world_state.check_and_clear_combat_hit() is True
        # Second call returns False (cleared)
        assert world_state.check_and_clear_combat_hit() is False

    def test_mark_with_zero_weapon_byte_is_miss(self) -> None:
        """mark_combat_hit with weapon_byte=0 does not set hit flag."""
        world_state.mark_combat_hit(weapon_byte=0)
        assert world_state.check_and_clear_combat_hit() is False

    def test_zero_weapon_byte_sets_our_shot_response(self) -> None:
        """mark_combat_hit with weapon_byte=0 still sets the shot response flag."""
        world_state.mark_combat_hit(weapon_byte=0)
        assert world_state.peek_our_shot_response() is True
        assert world_state.check_and_clear_our_shot_response() is True
        assert world_state.check_and_clear_our_shot_response() is False

    def test_nonzero_weapon_byte_sets_our_shot_response(self) -> None:
        """mark_combat_hit with weapon_byte>0 sets both hit and response flags."""
        world_state.mark_combat_hit(weapon_byte=1)
        assert world_state.peek_our_shot_response() is True
        assert world_state.peek_combat_hit() is True

    def test_our_shot_response_default_false(self) -> None:
        """check_and_clear_our_shot_response returns False by default."""
        assert world_state.check_and_clear_our_shot_response() is False
        assert world_state.peek_our_shot_response() is False

    def test_dual_hit_decrements_dual_count(self) -> None:
        """mark_combat_hit with weapon_byte=1 decrements dual_shots count."""
        from tankpit_bot.sniffer.world_state import (
            get_inventory_state,
            update_inventory_from_protocol,
        )

        update_inventory_from_protocol(
            [0, 7, 0, 0, 10],
            [False, True, False, False, True],
        )
        world_state.mark_combat_hit(weapon_byte=1)
        assert get_inventory_state()["dual_shots"]["count"] == 6
        assert get_inventory_state()["dual_shots"]["enabled"] is True

    def test_missile_hit_decrements_missile_count(self) -> None:
        """mark_combat_hit with weapon_byte=2 decrements missile_shots count."""
        from tankpit_bot.sniffer.world_state import (
            get_inventory_state,
            update_inventory_from_protocol,
        )

        update_inventory_from_protocol(
            [0, 0, 5, 0, 10],
            [False, False, True, False, True],
        )
        world_state.mark_combat_hit(weapon_byte=2)
        assert get_inventory_state()["missile_shots"]["count"] == 4

    def test_homing_hit_decrements_homing_count(self) -> None:
        """mark_combat_hit with weapon_byte=3 decrements homing_shots count."""
        from tankpit_bot.sniffer.world_state import (
            get_inventory_state,
            update_inventory_from_protocol,
        )

        update_inventory_from_protocol(
            [0, 0, 0, 3, 10],
            [False, False, False, True, True],
        )
        world_state.mark_combat_hit(weapon_byte=3)
        assert get_inventory_state()["homing_shots"]["count"] == 2

    def test_hit_decrement_does_not_go_below_zero(self) -> None:
        """mark_combat_hit does not decrement below zero."""
        from tankpit_bot.sniffer.world_state import (
            get_inventory_state,
            update_inventory_from_protocol,
        )

        update_inventory_from_protocol(
            [0, 0, 0, 0, 10],
            [False, True, False, False, True],
        )
        world_state.mark_combat_hit(weapon_byte=1)
        assert get_inventory_state()["dual_shots"]["count"] == 0

    def test_consecutive_hits_deplete_dual(self) -> None:
        """Multiple dual hits decrement count to zero progressively."""
        from tankpit_bot.sniffer.world_state import (
            get_inventory_state,
            update_inventory_from_protocol,
        )

        update_inventory_from_protocol(
            [0, 3, 0, 0, 10],
            [False, True, False, False, True],
        )
        world_state.mark_combat_hit(weapon_byte=1)
        assert get_inventory_state()["dual_shots"]["count"] == 2
        world_state.mark_combat_hit(weapon_byte=1)
        assert get_inventory_state()["dual_shots"]["count"] == 1
        world_state.mark_combat_hit(weapon_byte=1)
        assert get_inventory_state()["dual_shots"]["count"] == 0
        # Fourth hit: already at zero, stays at zero
        world_state.mark_combat_hit(weapon_byte=1)
        assert get_inventory_state()["dual_shots"]["count"] == 0

    def test_reset_clears_our_shot_response(self) -> None:
        """reset_world_state clears the our_shot_response flag."""
        world_state.mark_combat_hit(weapon_byte=0)
        reset_world_state()
        assert world_state.peek_our_shot_response() is False


class TestIncrementContainerFailedPickups:
    """Tests for increment_container_failed_pickups."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_increments_failed_pickups(self) -> None:
        """Incrementing raises the failed_pickups counter by 1."""
        from tankpit_bot.sniffer.world_state import (
            increment_container_failed_pickups,
        )

        update_world_state_from_radar(
            [RadarContainerDict(x=50, y=60, volume=100)],
            [],
        )
        assert world_state._world_state["containers"]["50,60"]["failed_pickups"] == 0
        increment_container_failed_pickups(50, 60)
        assert world_state._world_state["containers"]["50,60"]["failed_pickups"] == 1
        increment_container_failed_pickups(50, 60)
        assert world_state._world_state["containers"]["50,60"]["failed_pickups"] == 2

    def test_noop_for_missing_container(self) -> None:
        """Incrementing a missing container is a no-op."""
        from tankpit_bot.sniffer.world_state import (
            increment_container_failed_pickups,
        )

        increment_container_failed_pickups(99, 99)
        assert len(world_state._world_state["containers"]) == 0


class TestTeleportLandedTracking:
    """Tests for mark_teleport_landed and check_and_clear_teleport_landed."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_check_returns_false_by_default(self) -> None:
        """check_and_clear_teleport_landed returns False with no teleport."""
        assert world_state.check_and_clear_teleport_landed() is False

    def test_mark_and_check_returns_true(self) -> None:
        """mark_teleport_landed sets flag, check returns True then False."""
        world_state.mark_teleport_landed()
        assert world_state.check_and_clear_teleport_landed() is True
        assert world_state.check_and_clear_teleport_landed() is False

    def test_dispatch_teleport_landed_sets_flag(self) -> None:
        """Container message with msg_type=teleport_landed marks landing."""
        from tankpit_bot.container.types import TeleportLandedDict

        msg = TeleportLandedDict(msg_type="teleport_landed", subtype=0)
        dispatch_world_state_update(msg)
        assert world_state.check_and_clear_teleport_landed() is True

    def test_reset_clears_teleport_flag(self) -> None:
        """reset_world_state clears the teleport landed flag."""
        world_state.mark_teleport_landed()
        reset_world_state()
        assert world_state.check_and_clear_teleport_landed() is False


class TestRemoveContainerAt:
    """Tests for remove_container_at world state mutation."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_removes_existing_container(self) -> None:
        """remove_container_at removes a container at the given coordinates."""
        from tankpit_bot.sniffer.world_state import remove_container_at

        update_world_state_from_radar(
            [RadarContainerDict(x=50, y=60, volume=100)],
            [],
        )
        assert "50,60" in world_state._world_state["containers"]
        remove_container_at(50, 60)
        assert "50,60" not in world_state._world_state["containers"]

    def test_noop_for_missing_container(self) -> None:
        """remove_container_at is a no-op when container doesn't exist."""
        from tankpit_bot.sniffer.world_state import remove_container_at

        remove_container_at(99, 99)
        assert len(world_state._world_state["containers"]) == 0


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
            rank=2,
            leaderboard_position=3,
        )
        dispatch_world_state_update(first)

        self_state = world_state._world_state["self_state"]
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
            rank=3,
            leaderboard_position=10,
        )
        dispatch_world_state_update(second)

        # Self position unchanged
        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["x"] == 100
        assert self_state["y"] == 100
        # But tank 99 should be registered
        assert "99" in world_state._world_state["tanks"]
