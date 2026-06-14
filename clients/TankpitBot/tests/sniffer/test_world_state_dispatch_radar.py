"""Tests for sniffer world state dispatch handling of radar messages."""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.container import RadarContainerDict
from tankpit_bot.sniffer import (
    dispatch_world_state_update,
    reset_world_state,
    update_world_state_from_position,
    world_state,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


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

        fake_terrain = InMemoryTerrainMap()
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


class TestDispatchRadarEmptyDelta:
    """Tests for tunneled 0x4F empty radar delta dispatch."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_empty_tunneled_radar_marks_pending_delta(self) -> None:
        """Dispatching an empty tunneled 0x4F marks a pending radar empty delta."""
        from tankpit_bot.protocol.types import RadarScanResultDict

        update_world_state_from_position(100, 100)
        msg = RadarScanResultDict(msg_type=0x4F, containers=[], mines=[])

        dispatch_world_state_update(msg)

        # The empty delta is pending; a RadarAck(found=True) should preserve
        from tankpit_bot.sniffer.world_state import _consume_pending_radar_empty_delta

        assert _consume_pending_radar_empty_delta() is True

    def test_nonempty_tunneled_radar_processes_immediately(self) -> None:
        """Dispatching a non-empty tunneled 0x4F processes containers immediately."""
        from tankpit_bot.protocol.types import RadarScanResultDict

        update_world_state_from_position(100, 100)
        containers = [RadarContainerDict(x=98, y=98, volume=500)]
        msg = RadarScanResultDict(msg_type=0x4F, containers=containers, mines=[])

        dispatch_world_state_update(msg)

        from tankpit_bot.sniffer.world_state import _consume_pending_radar_empty_delta

        assert _consume_pending_radar_empty_delta() is False
        result = world_state.get_world_state()
        assert "98,98" in result["containers"]
