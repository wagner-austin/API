"""Tests for sniffer world state dispatch handling of tank-related messages."""

from __future__ import annotations

from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
from tests.conftest import FakeFileSystem


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
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
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
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        assert "99" in state["tanks"]
        assert state["tanks"]["99"]["name"] == "TopPlayer"
        assert state["tanks"]["99"]["team"] == 2
        assert state["tanks"]["99"]["rank"] == 5

    def test_dispatch_tank_exit(self) -> None:
        """Test dispatch handles TankExit (0x58) message."""
        from tankpit_bot.protocol import TankEntryDict, TankExitDict

        # First add a tank
        entry_msg = TankEntryDict(msg_type=0x28, tank_id=42, x=100, y=150, name="LeavingBot")
        dispatch_world_state_update(get_world_service(), entry_msg)
        assert "42" in get_world_service().world_state["tanks"]

        # Then remove it
        exit_msg = TankExitDict(msg_type=0x58, tank_id=42)
        dispatch_world_state_update(get_world_service(), exit_msg)
        assert "42" not in get_world_service().world_state["tanks"]

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
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
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
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
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
        dispatch_world_state_update(get_world_service(), msg)

        # Tank should NOT be added since position is None (match falls through)
        state = get_world_service().world_state
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
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
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
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
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
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
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
        dispatch_world_state_update(get_world_service(), msg)

        # Tank should NOT be created since status_data too short for position
        state = get_world_service().world_state
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
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        assert "2308" not in state["tanks"]

    def test_dispatch_tunneled_terrain_update_sets_terrain_tile(self) -> None:
        """Test 0x4A terrain updates modify world terrain state."""
        from tankpit_bot.protocol import TerrainUpdateDict

        msg = TerrainUpdateDict(msg_type=0x4A, updates=[(8, 166, 2)])
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
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
            get_world_service(),
            MovementResponseDict(
                msg_type=0x3D,
                team=2,
                tank_id=1301,
                x=131,
                y=126,
                direction=8,
                rank=1,
                leaderboard_position=1313,
            ),
        )

        dispatch_world_state_update(
            get_world_service(),
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
            },
        )

        state = get_world_service().world_state
        assert state["mines"]["131,126"]["team"] == 2
        assert state["mines"]["131,126"]["tank_id"] == 1301
        assert state["mines"]["131,126"]["mine_type"] == 2
        assert state["mines"]["132,127"]["x"] == 132
        assert state["mines"]["132,127"]["y"] == 127

    def test_dispatch_tunneled_mine_placement_uses_known_tank_team(self) -> None:
        """Test tunneled 0x4B uses tracked tank team when placer is not self."""
        from tankpit_bot.protocol import TankEntryDict, TankInfoDict

        dispatch_world_state_update(
            get_world_service(),
            TankInfoDict(
                msg_type=0x21,
                tank_id=777,
                name="placer",
                team=3,
                decoration_state=b"",
                score=0,
            ),
        )

        dispatch_world_state_update(
            get_world_service(),
            TankEntryDict(
                msg_type=0x28,
                tank_id=777,
                x=40,
                y=41,
                name="placer",
            ),
        )

        dispatch_world_state_update(
            get_world_service(),
            {
                "msg_type": 0x4B,
                "mine_type": 1,
                "tank_id": 777,
                "positions": [(40, 41), (40, 42)],
            },
        )

        state = get_world_service().world_state
        assert state["mines"]["40,41"]["team"] == 3
        assert state["mines"]["40,42"]["team"] == 3
        assert state["mines"]["40,41"]["tank_id"] == 777

    def test_dispatch_tunneled_mine_placement_skips_unknown_team(self) -> None:
        """Test tunneled 0x4B does nothing when placer team is unknown."""
        dispatch_world_state_update(
            get_world_service(),
            {
                "msg_type": 0x4B,
                "mine_type": 2,
                "tank_id": 9999,
                "positions": [(10, 11), (11, 11)],
            },
        )

        state = get_world_service().world_state
        assert state["mines"] == {}

    def test_dispatch_tunneled_mine_detonation_removes_mines(self) -> None:
        """Test tunneled 0x45 removes mines at decoded coordinates."""
        from tankpit_bot.protocol import MovementResponseDict

        dispatch_world_state_update(
            get_world_service(),
            MovementResponseDict(
                msg_type=0x3D,
                team=2,
                tank_id=1301,
                x=38,
                y=53,
                direction=8,
                rank=1,
                leaderboard_position=1313,
            ),
        )

        dispatch_world_state_update(
            get_world_service(),
            {
                "msg_type": 0x4B,
                "mine_type": 2,
                "tank_id": 1301,
                "positions": [(38, 52), (39, 53), (38, 54)],
            },
        )

        dispatch_world_state_update(
            get_world_service(), {"msg_type": 0x45, "positions": [(39, 53), (38, 54)]}
        )

        state = get_world_service().world_state
        assert "38,52" in state["mines"]
        assert "39,53" not in state["mines"]
        assert "38,54" not in state["mines"]

    def test_dispatch_tank_status_short_updates_damage(self) -> None:
        """Test dispatch handles tank_status_short by updating damage."""
        from tankpit_bot.container import TankStatusShortDict
        from tankpit_bot.protocol import TankEntryDict

        # First create a tank
        entry = TankEntryDict(msg_type=0x28, tank_id=300, x=50, y=60, name="Target")
        dispatch_world_state_update(get_world_service(), entry)

        msg = TankStatusShortDict(
            msg_type="tank_status_short",
            flags=0x82,
            tank_id=300,
            damage_state=3,
            rank=4,
            leaderboard_position=21,
        )
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        assert "300" in state["tanks"]
        assert state["tanks"]["300"]["damage_state"] == 3

    def test_tank_damage_transition_emits_diagnostic(self, fake_fs: FakeFileSystem) -> None:
        """A damage-tier change surfaces as a diagnostic instead of silence.

        Run 20260610-223x fired 27 homing shots with zero artifact
        evidence of their effect: the tier sync was consumed silently.
        Repeats of the same tier stay quiet; only transitions emit.
        """
        from pathlib import Path

        from tankpit_bot.container import TankStatusShortDict
        from tankpit_bot.diagnostics.event_stream import load_event_records
        from tankpit_bot.protocol import TankEntryDict
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging

        artifacts = configure_bot_runtime_logging("20260610-230000")
        entry = TankEntryDict(msg_type=0x28, tank_id=300, x=50, y=60, name="Target")
        dispatch_world_state_update(get_world_service(), entry)

        for repeated_damage_state in (2, 2):
            msg = TankStatusShortDict(
                msg_type="tank_status_short",
                flags=0x82,
                tank_id=300,
                damage_state=repeated_damage_state,
                rank=4,
                leaderboard_position=21,
            )
            dispatch_world_state_update(get_world_service(), msg)

        records = [
            record
            for record in load_event_records(Path(artifacts["latest_events_path"]))
            if record["fields"].get("diagnostic_kind") == "tank_damage_changed"
        ]
        assert len(records) == 1
        assert records[0]["fields"] == {
            "diagnostic_kind": "tank_damage_changed",
            "tank_id": 300,
            "tank_name": "Target",
            "previous_damage_state": 0,
            "damage_state": 2,
        }

    def test_tank_damage_for_unknown_tank_emits_nothing(self, fake_fs: FakeFileSystem) -> None:
        """A tier sync for an untracked tank has no transition to report."""
        from pathlib import Path

        from tankpit_bot.container import TankStatusShortDict
        from tankpit_bot.diagnostics.event_stream import load_event_records
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging

        artifacts = configure_bot_runtime_logging("20260610-230000")
        msg = TankStatusShortDict(
            msg_type="tank_status_short",
            flags=0x82,
            tank_id=999,
            damage_state=3,
            rank=4,
            leaderboard_position=21,
        )
        dispatch_world_state_update(get_world_service(), msg)

        records = [
            record
            for record in load_event_records(Path(artifacts["latest_events_path"]))
            if record["fields"].get("diagnostic_kind") == "tank_damage_changed"
        ]
        assert records == []

    def test_dispatch_tank_leave_removes_tank(self) -> None:
        """Test dispatch handles tank_leave by removing the tank."""
        from tankpit_bot.container import TankLeaveDict
        from tankpit_bot.protocol import TankEntryDict

        # First create a tank
        entry = TankEntryDict(msg_type=0x28, tank_id=400, x=50, y=60, name="Leaving")
        dispatch_world_state_update(get_world_service(), entry)
        assert "400" in get_world_service().world_state["tanks"]

        msg = TankLeaveDict(
            msg_type="tank_leave",
            tank_id=400,
            flags=0x13,
            extra_data=b"\x42\x13",
        )
        dispatch_world_state_update(get_world_service(), msg)

        assert "400" not in get_world_service().world_state["tanks"]

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
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
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
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        assert "550" in state["tanks"]
        # Final position: (100+4, 80+3) = (104, 83)
        assert state["tanks"]["550"]["x"] == 104
        assert state["tanks"]["550"]["y"] == 83

        # Clean up
        _player_id_mapper._player_to_tank.clear()
        _player_id_mapper._position_to_tank.clear()


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
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
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
        dispatch_world_state_update(get_world_service(), entry)

        # Detection updates to new position
        msg = EnemyDetectionDict(
            msg_type=0x48,
            tank_id=556,
            x=200,
            y=210,
            rank=5,
            team=1,
        )
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        assert state["tanks"]["556"]["x"] == 200
        assert state["tanks"]["556"]["y"] == 210
