"""Tests for sniffer world state dispatch handling of radar messages."""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.protocol import RadarContainerDict
from tankpit_bot.sniffer.world_service import WorldService
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

        ws = WorldService()
        _test_hooks.path_exists = lambda path: False

        msg = RadarScanResultDict(
            msg_type=0x4F,
            containers=[RadarContainerDict(x=100, y=100, volume=50)],
            mines=[RadarMineDict(x=110, y=110, team=0)],
            mine_clears=[],
        )

        dispatch_world_state_update(ws, msg)

        assert "100,100" in ws.world_state["containers"]
        assert "110,110" in ws.world_state["mines"]

    def test_dispatch_radar_response_renders_ascii(self) -> None:
        """Test dispatch renders ASCII after radar update when terrain available."""
        from tankpit_bot.protocol import RadarContainerDict, RadarScanResultDict

        ws = WorldService()
        fake_terrain = InMemoryTerrainMap()
        _test_hooks.path_exists = lambda path: True
        _test_hooks.load_terrain_map = lambda path: fake_terrain

        ws.update_world_state_from_position(128, 128)

        msg = RadarScanResultDict(
            msg_type=0x4F,
            containers=[RadarContainerDict(x=128, y=128, volume=50)],
            mines=[],
            mine_clears=[],
        )

        dispatch_world_state_update(ws, msg)

    def test_dispatch_radar_ack_marks_scan_complete(self) -> None:
        """Test dispatch handles radar ack messages as scan completion."""
        from tankpit_bot.protocol import RadarResultDict

        ws = WorldService()
        msg = RadarResultDict(
            msg_type=0x46,
            detection_type=0,
            found=True,
        )

        dispatch_world_state_update(ws, msg)

        assert ws.check_and_clear_radar_scan_complete() is True


class TestDispatchRadarEmptyDelta:
    """Tests for tunneled 0x4F empty radar delta dispatch."""

    def test_empty_tunneled_radar_marks_pending_delta(self) -> None:
        """Dispatching an empty tunneled 0x4F marks a pending radar empty delta."""
        from tankpit_bot.protocol.types import RadarScanResultDict

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        msg = RadarScanResultDict(msg_type=0x4F, containers=[], mines=[], mine_clears=[])

        dispatch_world_state_update(ws, msg)

        # The empty delta is pending; a RadarAck(found=True) should preserve
        assert ws.consume_pending_radar_empty_delta() is True

    def test_no_pending_delta_is_not_recent_on_a_fresh_clock(self) -> None:
        """Nothing pending is never "recent", however young the clock is.

        The sibling test below already consumes with nothing pending,
        but under the wall clock (~1.79e12 ms) the recency arithmetic
        answers False on its own: ``now - 0 <= 2000`` cannot hold. It
        holds whenever the clock reads under the 2s window, which is
        exactly what a replay or sim clock does -- ``_replay_page``
        starts at 0 and ``ReplayClock(1000)`` starts at 1000.

        There, an unset sentinel of 0 would be read as a timestamp two
        seconds ago and the caller would preserve a radar cache on the
        strength of an empty delta that was never observed.
        """
        from tankpit_bot import _test_hooks as core_hooks

        def _fresh_clock() -> int:
            return 500

        ws = WorldService()
        saved = core_hooks.get_current_time_ms
        core_hooks.get_current_time_ms = _fresh_clock
        try:
            assert ws.consume_pending_radar_empty_delta() is False
        finally:
            core_hooks.get_current_time_ms = saved

    def test_a_real_pending_delta_is_recent_on_the_same_fresh_clock(self) -> None:
        """Control: a delta actually marked on that clock does read as recent."""
        from tankpit_bot import _test_hooks as core_hooks

        def _fresh_clock() -> int:
            return 500

        ws = WorldService()
        saved = core_hooks.get_current_time_ms
        core_hooks.get_current_time_ms = _fresh_clock
        try:
            ws.mark_pending_radar_empty_delta()
            assert ws.consume_pending_radar_empty_delta() is True
        finally:
            core_hooks.get_current_time_ms = saved

    def test_nonempty_tunneled_radar_processes_immediately(self) -> None:
        """Dispatching a non-empty tunneled 0x4F processes containers immediately."""
        from tankpit_bot.protocol.types import RadarScanResultDict

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        containers = [RadarContainerDict(x=98, y=98, volume=500)]
        msg = RadarScanResultDict(msg_type=0x4F, containers=containers, mines=[], mine_clears=[])

        dispatch_world_state_update(ws, msg)

        assert ws.consume_pending_radar_empty_delta() is False
        result = ws.get_world_state()
        assert "98,98" in result["containers"]
