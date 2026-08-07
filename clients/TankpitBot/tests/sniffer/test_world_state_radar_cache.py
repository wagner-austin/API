"""Tests for radar cache promotion and differential radar handling."""

from __future__ import annotations

from tankpit_bot.protocol.types import RadarContainerDict
from tankpit_bot.sniffer.world_state import (
    check_and_clear_radar_scan_complete,
    get_world_service,
    get_world_state,
    mark_scan_viewport_failed,
    update_world_state_from_position,
)
from tankpit_bot.sniffer.world_state_radar import (
    handle_radar_ack as _handle_radar_ack,
)
from tankpit_bot.sniffer.world_state_radar import (
    reconcile_radar_viewport_resources as _reconcile_radar_viewport_resources,
)
from tankpit_bot.sniffer.world_state_radar import (
    update_world_state_from_radar,
    update_world_state_from_radar_cache,
    update_world_state_from_radar_known_resources,
)
from tankpit_bot.state.types import WorldStateDict, coord_key, make_container_state


class TestUpdateWorldStateFromRadarCache:
    """Tests for update_world_state_from_radar_cache."""

    def setup_method(self) -> None:
        """Reset state before each test."""
        update_world_state_from_position(100, 100)

    def test_refreshes_existing_envelope_containers(self) -> None:
        """Radar cache refresh bumps refresh_kind on in-envelope containers."""
        from tankpit_bot.state.viewport_geometry import make_visible_viewport_state

        svc = get_world_service()
        key = coord_key(96, 96)
        svc.world_state = WorldStateDict(
            **{
                **svc.world_state,
                "viewport": make_visible_viewport_state(92, 92),
                "containers": {
                    key: make_container_state(
                        x=96,
                        y=96,
                        is_fuel=True,
                        volume=400,
                        source="viewport",
                        refresh_kind="viewport_patch",
                        timestamp_ms=50000,
                    ),
                },
            },
        )

        update_world_state_from_radar_cache(svc)

        result = get_world_state()
        assert check_and_clear_radar_scan_complete() is True
        assert key in result["containers"]
        assert result["containers"][key]["volume"] == 400
        assert result["containers"][key]["refresh_kind"] == "radar_cache_refresh"

    def test_regular_radar_marks_only_5x5_around_tank(self) -> None:
        """Built-in radar marks just its 5x5 footprint, not the whole viewport."""
        from tankpit_bot.state.viewport_geometry import make_visible_viewport_state

        svc = get_world_service()
        svc.world_state = WorldStateDict(
            **{
                **svc.world_state,
                "viewport": make_visible_viewport_state(92, 92),
                "scanned_tiles": {},
            },
        )
        svc.record_radar_command(use_extra_radar=False)

        update_world_state_from_radar_cache(svc)

        result = get_world_state()
        # Tank at (100, 100) inside viewport (92..107)^2: free radar covers the
        # interior 5x5 around the tank = 25 tiles, not the whole 256-tile viewport.
        assert len(result["scanned_tiles"]) == 25
        assert "100,100" in result["scanned_tiles"]
        assert "92,92" not in result["scanned_tiles"]

    def test_extra_radar_marks_every_viewport_tile_on_explicit_radar_response(self) -> None:
        """Extra radar marks every tile in the viewport."""
        from tankpit_bot.state.viewport_geometry import make_visible_viewport_state

        svc = get_world_service()
        svc.world_state = WorldStateDict(
            **{**svc.world_state, "viewport": make_visible_viewport_state(92, 92)},
        )
        get_world_service().record_radar_command(use_extra_radar=True)

        update_world_state_from_radar(svc, [RadarContainerDict(x=98, y=98, volume=500)], [], [])

        result = get_world_state()
        assert len(result["scanned_tiles"]) == 16 * 16
        assert "92,92" in result["scanned_tiles"]
        assert "107,107" in result["scanned_tiles"]

    def test_regular_radar_response_marks_only_5x5_around_tank(self) -> None:
        """Built-in radar responses mark just the 5x5 around the tank."""
        from tankpit_bot.state.viewport_geometry import make_visible_viewport_state

        svc = get_world_service()
        svc.world_state = WorldStateDict(
            **{
                **svc.world_state,
                "viewport": make_visible_viewport_state(92, 92),
                "scanned_tiles": {},
            },
        )
        get_world_service().record_radar_command(use_extra_radar=False)

        update_world_state_from_radar(svc, [RadarContainerDict(x=100, y=100, volume=500)], [], [])

        result = get_world_state()
        # Tank at (100, 100) reveals the 5x5 around the tank, all inside the viewport.
        assert len(result["scanned_tiles"]) == 25
        assert "100,100" in result["scanned_tiles"]


class TestUpdateWorldStateFromRadarKnownResources:
    """Tests for update_world_state_from_radar_known_resources."""

    def setup_method(self) -> None:
        """Reset state before each test."""
        update_world_state_from_position(100, 100)

    def test_preserves_existing_containers(self) -> None:
        """Zero-delta radar preserves known containers in viewport."""
        from tankpit_bot.state.viewport_geometry import make_visible_viewport_state

        svc = get_world_service()
        svc.world_state = WorldStateDict(
            **{**svc.world_state, "viewport": make_visible_viewport_state(92, 92)},
        )
        containers = [RadarContainerDict(x=98, y=98, volume=500)]
        update_world_state_from_radar(svc, containers, [], [])

        update_world_state_from_radar_known_resources(svc)

        result = get_world_state()
        assert check_and_clear_radar_scan_complete() is True
        key = coord_key(98, 98)
        assert key in result["containers"]
        assert result["containers"][key]["volume"] == 500

    def test_regular_radar_known_resources_marks_only_5x5_around_tank(self) -> None:
        """Built-in zero-delta radar marks just its 5x5 footprint, not the whole viewport."""
        from tankpit_bot.state.viewport_geometry import make_visible_viewport_state

        svc = get_world_service()
        svc.world_state = WorldStateDict(
            **{
                **svc.world_state,
                "viewport": make_visible_viewport_state(92, 92),
                "containers": {
                    coord_key(100, 100): make_container_state(
                        x=100,
                        y=100,
                        is_fuel=True,
                        volume=500,
                        timestamp_ms=100000,
                    ),
                },
                "scanned_tiles": {},
            },
        )
        get_world_service().record_radar_command(use_extra_radar=False)

        update_world_state_from_radar_known_resources(svc)

        result = get_world_state()
        assert len(result["scanned_tiles"]) == 25
        assert "100,100" in result["scanned_tiles"]


class TestHandleRadarAck:
    """Tests for _handle_radar_ack dispatch."""

    def setup_method(self) -> None:
        """Reset state before each test."""
        update_world_state_from_position(100, 100)

    def test_empty_delta_found_true_preserves(self) -> None:
        """RadarAck(found=True) after empty delta preserves known resources."""
        containers = [RadarContainerDict(x=98, y=98, volume=500)]
        update_world_state_from_radar(get_world_service(), containers, [], [])
        get_world_service().mark_pending_radar_empty_delta()

        _handle_radar_ack(get_world_service(), found=True)

        result = get_world_state()
        assert coord_key(98, 98) in result["containers"]

    def test_empty_delta_found_true_routes_through_cache_refresh(self) -> None:
        """RadarAck(found=True) with non-empty envelope refreshes existing containers."""
        from tankpit_bot.state.viewport_geometry import make_visible_viewport_state

        svc = get_world_service()
        fuel_key = coord_key(96, 96)
        equip_key = coord_key(97, 96)
        svc.world_state = WorldStateDict(
            **{
                **svc.world_state,
                "viewport": make_visible_viewport_state(92, 92),
                "containers": {
                    fuel_key: make_container_state(
                        x=96,
                        y=96,
                        is_fuel=True,
                        volume=400,
                        source="viewport",
                        refresh_kind="viewport_patch",
                        timestamp_ms=50000,
                    ),
                    equip_key: make_container_state(
                        x=97,
                        y=96,
                        is_fuel=False,
                        volume=0,
                        source="viewport",
                        refresh_kind="viewport_patch",
                        timestamp_ms=50000,
                    ),
                },
            },
        )
        svc.mark_pending_radar_empty_delta()

        _handle_radar_ack(svc, found=True)

        result = get_world_state()
        assert check_and_clear_radar_scan_complete() is True
        assert fuel_key in result["containers"]
        assert equip_key in result["containers"]
        assert result["containers"][fuel_key]["volume"] == 400
        assert result["containers"][equip_key]["is_fuel"] is False
        assert result["containers"][fuel_key]["refresh_kind"] == "radar_cache_refresh"

    def test_empty_delta_found_false_clears(self) -> None:
        """RadarAck(found=False) after empty delta clears viewport."""
        containers = [RadarContainerDict(x=98, y=98, volume=500)]
        update_world_state_from_radar(get_world_service(), containers, [], [])
        get_world_service().mark_pending_radar_empty_delta()

        _handle_radar_ack(get_world_service(), found=False)

        assert check_and_clear_radar_scan_complete() is True

    def test_no_pending_marks_scan_complete(self) -> None:
        """RadarAck without pending cache or delta just marks scan done."""
        _handle_radar_ack(get_world_service(), found=True)

        assert check_and_clear_radar_scan_complete() is True


class TestReconcileRadarViewportResources:
    """Tests for _reconcile_radar_viewport_resources."""

    def setup_method(self) -> None:
        """Reset state before each test."""
        update_world_state_from_position(100, 100)

    def test_removes_stale_containers_not_in_radar(self) -> None:
        """Containers in radar bounds but absent from new radar are removed."""
        # Set viewport so radar bounds cover (98,98)
        from tankpit_bot.state.viewport_geometry import make_visible_viewport_state

        svc = get_world_service()
        svc.world_state = WorldStateDict(
            **{**svc.world_state, "viewport": make_visible_viewport_state(92, 92)},
        )
        stale = [RadarContainerDict(x=98, y=98, volume=500)]
        update_world_state_from_radar(svc, stale, [], [])
        assert coord_key(98, 98) in get_world_state()["containers"]

        # New radar shows different container, stale one should be reconciled
        fresh = [RadarContainerDict(x=99, y=99, volume=300)]
        _reconcile_radar_viewport_resources(svc, fresh, [])

        result = get_world_state()
        assert coord_key(98, 98) not in result["containers"]

    def test_reconcile_with_none_mines_skips_mine_removal(self) -> None:
        """Passing None for mines skips mine reconciliation."""
        _reconcile_radar_viewport_resources(get_world_service(), [], None)
        # Should not raise or modify mines

    def test_reconcile_spares_visible_containers(self) -> None:
        """Viewport-sourced containers survive a radar that omits them.

        The radar response lists only newly revealed HIDDEN containers;
        visible containers stay on screen and are not re-sent. Live run
        2026-07-01 20:20:12: the old whole-envelope reconcile deleted
        all 7 visible landing containers because the scan-on-landing
        radar listed only the 2 hidden ones.
        """
        from tankpit_bot.state.viewport_geometry import make_visible_viewport_state

        svc = get_world_service()
        visible = make_container_state(98, 98, True, 1042, source="viewport")
        svc.world_state = WorldStateDict(
            **{
                **svc.world_state,
                "viewport": make_visible_viewport_state(92, 92),
                "containers": {coord_key(98, 98): visible},
            },
        )

        _reconcile_radar_viewport_resources(svc, [RadarContainerDict(x=99, y=99, volume=300)], [])

        assert coord_key(98, 98) in get_world_state()["containers"]

    def test_reconcile_spares_visible_mines_and_removes_radar_mines(self) -> None:
        """Mine reconciliation removes only radar-sourced entries."""
        from tankpit_bot.state.types import make_mine_state
        from tankpit_bot.state.viewport_geometry import make_visible_viewport_state

        svc = get_world_service()
        visible_mine = make_mine_state(97, 97, 0, -1, 3, source="viewport")
        radar_mine = make_mine_state(98, 98, 0, -1, 3, source="radar")
        svc.world_state = WorldStateDict(
            **{
                **svc.world_state,
                "viewport": make_visible_viewport_state(92, 92),
                "mines": {
                    coord_key(97, 97): visible_mine,
                    coord_key(98, 98): radar_mine,
                },
            },
        )

        _reconcile_radar_viewport_resources(svc, [], [])

        result = get_world_state()
        assert coord_key(97, 97) in result["mines"]
        assert coord_key(98, 98) not in result["mines"]


class TestScanViewportFailed:
    """Tests for mark_scan_viewport_failed and is_scan_viewport_failed."""

    def test_failed_viewport_prevents_repeat_scan(self) -> None:
        """A recently failed viewport scan is recognized as failed."""
        from tankpit_bot.sniffer.world_state import is_scan_viewport_failed

        mark_scan_viewport_failed(92, 92, 100000)

        assert is_scan_viewport_failed(92, 92, 100000) is True

    def test_different_viewport_is_not_failed(self) -> None:
        """A different viewport origin is not affected by a prior failure."""
        from tankpit_bot.sniffer.world_state import is_scan_viewport_failed

        mark_scan_viewport_failed(92, 92, 100000)

        assert is_scan_viewport_failed(50, 50, 100000) is False
