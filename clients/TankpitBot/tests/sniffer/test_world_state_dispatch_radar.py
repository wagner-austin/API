"""Tests for sniffer world state dispatch handling of radar messages."""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.protocol import RadarContainerDict
from tankpit_bot.sniffer.world_state import (
    get_world_service,
    update_world_state_from_position,
)
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
from tests.in_memory_terrain_map import InMemoryTerrainMap


class TestDispatchRadar:
    """Tests for dispatch_world_state_update with radar messages."""

    def teardown_method(self) -> None:
        """Reset world state and hooks after each test."""
        _test_hooks.path_exists = _test_hooks._real_path_exists
        _test_hooks.load_terrain_map = _test_hooks._real_load_terrain_map

    def test_dispatch_radar_response(self) -> None:
        """Test dispatch handles 0x4F RadarScanResult message."""
        from tankpit_bot.protocol import RadarContainerDict, RadarMineDict, RadarScanResultDict

        _test_hooks.path_exists = lambda path: False

        msg = RadarScanResultDict(
            msg_type=0x4F,
            containers=[RadarContainerDict(x=100, y=100, volume=50)],
            mines=[RadarMineDict(x=110, y=110, team=0)],
            mine_clears=[],
        )

        dispatch_world_state_update(get_world_service(), msg)

        assert "100,100" in get_world_service().world_state["containers"]
        assert "110,110" in get_world_service().world_state["mines"]

    def test_dispatch_radar_response_renders_ascii(self) -> None:
        """Test dispatch renders ASCII after radar update when terrain available."""
        from tankpit_bot.protocol import RadarContainerDict, RadarScanResultDict

        fake_terrain = InMemoryTerrainMap()
        _test_hooks.path_exists = lambda path: True
        _test_hooks.load_terrain_map = lambda path: fake_terrain

        update_world_state_from_position(128, 128)

        msg = RadarScanResultDict(
            msg_type=0x4F,
            containers=[RadarContainerDict(x=128, y=128, volume=50)],
            mines=[],
            mine_clears=[],
        )

        dispatch_world_state_update(get_world_service(), msg)

    def test_dispatch_radar_ack_marks_scan_complete(self) -> None:
        """Test dispatch handles radar ack messages as scan completion."""
        from tankpit_bot.protocol import RadarResultDict
        from tankpit_bot.sniffer.world_state import check_and_clear_radar_scan_complete

        msg = RadarResultDict(
            msg_type=0x46,
            detection_type=0,
            found=True,
        )

        dispatch_world_state_update(get_world_service(), msg)

        assert check_and_clear_radar_scan_complete() is True


class TestDispatchRadarEmptyDelta:
    """Tests for tunneled 0x4F empty radar delta dispatch."""

    def test_empty_tunneled_radar_marks_pending_delta(self) -> None:
        """Dispatching an empty tunneled 0x4F marks a pending radar empty delta."""
        from tankpit_bot.protocol.types import RadarScanResultDict

        update_world_state_from_position(100, 100)
        msg = RadarScanResultDict(msg_type=0x4F, containers=[], mines=[], mine_clears=[])

        dispatch_world_state_update(get_world_service(), msg)

        # The empty delta is pending; a RadarAck(found=True) should preserve
        assert get_world_service().consume_pending_radar_empty_delta() is True

    def test_nonempty_tunneled_radar_processes_immediately(self) -> None:
        """Dispatching a non-empty tunneled 0x4F processes containers immediately."""
        from tankpit_bot.protocol.types import RadarScanResultDict

        update_world_state_from_position(100, 100)
        containers = [RadarContainerDict(x=98, y=98, volume=500)]
        msg = RadarScanResultDict(msg_type=0x4F, containers=containers, mines=[], mine_clears=[])

        dispatch_world_state_update(get_world_service(), msg)

        assert get_world_service().consume_pending_radar_empty_delta() is False
        result = get_world_service().get_world_state()
        assert "98,98" in result["containers"]
