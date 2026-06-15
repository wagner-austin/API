"""Tests for radar cache promotion and differential radar handling."""

from __future__ import annotations

from tankpit_bot.protocol.types import RadarContainerDict
from tankpit_bot.sniffer.world_state import (
    check_and_clear_radar_scan_complete,
    get_world_service,
    get_world_state,
    mark_scan_viewport_failed,
    reset_world_state,
    update_world_state_from_position,
)
from tankpit_bot.sniffer.world_state_radar import (
    clear_container_tile_cache as _clear_container_tile_cache,
)
from tankpit_bot.sniffer.world_state_radar import (
    containers_from_current_radar_cache as _containers_from_current_radar_cache,
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
from tankpit_bot.state.mutations import update_terrain_from_viewport
from tankpit_bot.state.types import WorldStateDict, coord_key, make_container_state


class TestContainersFromCurrentRadarCache:
    """Tests for _containers_from_current_radar_cache."""

    def setup_method(self) -> None:
        """Reset state before each test."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset state after each test."""
        reset_world_state()

    def test_extracts_fuel_from_terrain_cache(self) -> None:
        """Fuel containers are extracted from terrain cache_value > 0."""
        world = get_world_state()
        entities: list[tuple[int, int, int, int, int]] = [
            (5, 5, 0, 500, 255),
        ]
        # update_terrain_from_viewport sets viewport to (92,92)
        updated = update_terrain_from_viewport(world, 92, 92, entities, 100000)
        get_world_service().world_state = updated

        containers = _containers_from_current_radar_cache(get_world_service())

        assert len(containers) == 1
        assert containers[0]["x"] == 96
        assert containers[0]["volume"] == 500

    def test_extracts_equipment_from_terrain_cache(self) -> None:
        """Equipment containers are extracted from terrain cache_value == -1."""
        world = get_world_state()
        entities: list[tuple[int, int, int, int, int]] = [
            (3, 3, 0, -1, 255),
        ]
        updated = update_terrain_from_viewport(world, 92, 92, entities, 100000)
        get_world_service().world_state = updated

        containers = _containers_from_current_radar_cache(get_world_service())

        assert len(containers) == 1
        assert containers[0]["volume"] == -1

    def test_extracts_mixed_fuel_and_equipment_ignoring_empty(self) -> None:
        """Both fuel and equipment tiles are extracted; cache_value=0 is skipped."""
        world = get_world_state()
        entities: list[tuple[int, int, int, int, int]] = [
            (3, 3, 0, 400, 255),
            (4, 4, 0, -1, 255),
            (5, 5, 0, 0, 255),
        ]
        updated = update_terrain_from_viewport(world, 92, 92, entities, 100000)
        get_world_service().world_state = updated

        containers = _containers_from_current_radar_cache(get_world_service())

        assert len(containers) == 2
        volumes = {c["volume"] for c in containers}
        assert 400 in volumes
        assert -1 in volumes

    def test_skips_tiles_outside_radar_bounds(self) -> None:
        """Tiles outside the radar envelope are excluded."""
        world = get_world_state()
        # Put a tile far outside the viewport
        entities: list[tuple[int, int, int, int, int]] = [
            (0, 0, 0, 700, 255),
        ]
        updated = update_terrain_from_viewport(world, 92, 92, entities, 100000)
        # Add a tile at (200, 200) which is outside radar bounds
        from tankpit_bot.state.types import make_terrain_tile

        new_terrain = dict(updated["terrain"])
        new_terrain[coord_key(200, 200)] = make_terrain_tile(
            x=200,
            y=200,
            terrain_type=0,
            cache_value=999,
            overlay_value=255,
        )
        get_world_service().world_state = WorldStateDict(**{**updated, "terrain": new_terrain})

        containers = _containers_from_current_radar_cache(get_world_service())

        # The tile at (200,200) is outside radar bounds and should be skipped
        for c in containers:
            assert c["x"] != 200

    def test_regular_radar_uses_centered_5x5_bounds(self) -> None:
        """Built-in radar only covers a 5x5 square centered on the tank."""
        world = get_world_state()
        entities: list[tuple[int, int, int, int, int]] = [
            (5, 5, 0, 400, 255),
            (9, 9, 0, 700, 255),
        ]
        updated = update_terrain_from_viewport(world, 92, 92, entities, 100000)
        get_world_service().world_state = updated
        get_world_service().record_radar_command(use_extra_radar=False)

        containers = _containers_from_current_radar_cache(get_world_service())

        assert get_world_service().current_radar_uses_extra() is False
        assert coord_key(96, 96) not in {coord_key(c["x"], c["y"]) for c in containers}
        assert coord_key(100, 100) in {coord_key(c["x"], c["y"]) for c in containers}


class TestUpdateWorldStateFromRadarCache:
    """Tests for update_world_state_from_radar_cache."""

    def setup_method(self) -> None:
        """Reset state before each test."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset state after each test."""
        reset_world_state()

    def test_promotes_cache_to_containers(self) -> None:
        """Radar cache promotion creates authoritative containers."""
        world = get_world_state()
        entities: list[tuple[int, int, int, int, int]] = [
            (5, 5, 0, 400, 255),
        ]
        updated = update_terrain_from_viewport(world, 92, 92, entities, 100000)
        get_world_service().world_state = updated

        update_world_state_from_radar_cache(get_world_service())

        result = get_world_state()
        assert check_and_clear_radar_scan_complete() is True
        key = coord_key(96, 96)
        assert key in result["containers"]
        assert result["containers"][key]["volume"] == 400

    def test_regular_radar_does_not_mark_entire_viewport_scanned(self) -> None:
        """Built-in radar does not claim authoritative coverage for the full viewport."""
        world = get_world_state()
        entities: list[tuple[int, int, int, int, int]] = [
            (9, 9, 0, 400, 255),
        ]
        updated = update_terrain_from_viewport(world, 92, 92, entities, 100000)
        get_world_service().world_state = WorldStateDict(**{**updated, "scanned_viewports": {}})
        get_world_service().record_radar_command(use_extra_radar=False)

        update_world_state_from_radar_cache(get_world_service())

        result = get_world_state()
        assert result["scanned_viewports"] == {}

    def test_extra_radar_marks_viewport_scanned_on_explicit_radar_response(self) -> None:
        """Extra radar still marks the full viewport as scanned."""
        from tankpit_bot.state.viewport_geometry import make_visible_viewport_state

        svc = get_world_service()
        svc.world_state = WorldStateDict(
            **{**svc.world_state, "viewport": make_visible_viewport_state(92, 92)},
        )
        get_world_service().record_radar_command(use_extra_radar=True)

        update_world_state_from_radar(svc, [RadarContainerDict(x=98, y=98, volume=500)], [])

        result = get_world_state()
        assert "92,92" in result["scanned_viewports"]

    def test_regular_radar_response_does_not_mark_entire_viewport_scanned(self) -> None:
        """Built-in radar responses do not mark the whole viewport as scanned."""
        from tankpit_bot.state.viewport_geometry import make_visible_viewport_state

        svc = get_world_service()
        svc.world_state = WorldStateDict(
            **{
                **svc.world_state,
                "viewport": make_visible_viewport_state(92, 92),
                "scanned_viewports": {},
            },
        )
        get_world_service().record_radar_command(use_extra_radar=False)

        update_world_state_from_radar(svc, [RadarContainerDict(x=100, y=100, volume=500)], [])

        result = get_world_state()
        assert result["scanned_viewports"] == {}


class TestUpdateWorldStateFromRadarKnownResources:
    """Tests for update_world_state_from_radar_known_resources."""

    def setup_method(self) -> None:
        """Reset state before each test."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset state after each test."""
        reset_world_state()

    def test_preserves_existing_containers(self) -> None:
        """Zero-delta radar preserves known containers in viewport."""
        from tankpit_bot.state.viewport_geometry import make_visible_viewport_state

        svc = get_world_service()
        svc.world_state = WorldStateDict(
            **{**svc.world_state, "viewport": make_visible_viewport_state(92, 92)},
        )
        containers = [RadarContainerDict(x=98, y=98, volume=500)]
        update_world_state_from_radar(svc, containers, [])

        update_world_state_from_radar_known_resources(svc)

        result = get_world_state()
        assert check_and_clear_radar_scan_complete() is True
        key = coord_key(98, 98)
        assert key in result["containers"]
        assert result["containers"][key]["volume"] == 500

    def test_regular_radar_known_resources_does_not_mark_entire_viewport_scanned(self) -> None:
        """Built-in zero-delta radar preserves resources without viewport-wide confirmation."""
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
                "scanned_viewports": {},
            },
        )
        get_world_service().record_radar_command(use_extra_radar=False)

        update_world_state_from_radar_known_resources(svc)

        result = get_world_state()
        assert result["scanned_viewports"] == {}


class TestHandleRadarAck:
    """Tests for _handle_radar_ack dispatch."""

    def setup_method(self) -> None:
        """Reset state before each test."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset state after each test."""
        reset_world_state()

    def test_cache_refresh_path(self) -> None:
        """RadarAck after a cache refresh promotes to authoritative containers."""
        world = get_world_state()
        entities: list[tuple[int, int, int, int, int]] = [
            (5, 5, 0, 300, 255),
        ]
        updated = update_terrain_from_viewport(world, 92, 92, entities, 100000)
        get_world_service().world_state = updated
        get_world_service().mark_pending_radar_cache_refresh()

        _handle_radar_ack(get_world_service(), found=True)

        assert check_and_clear_radar_scan_complete() is True

    def test_empty_delta_found_true_preserves(self) -> None:
        """RadarAck(found=True) after empty delta preserves known resources."""
        containers = [RadarContainerDict(x=98, y=98, volume=500)]
        update_world_state_from_radar(get_world_service(), containers, [])
        get_world_service().mark_pending_radar_empty_delta()

        _handle_radar_ack(get_world_service(), found=True)

        result = get_world_state()
        assert coord_key(98, 98) in result["containers"]

    def test_empty_delta_found_true_promotes_cache_when_known_resources_are_empty(self) -> None:
        """RadarAck(found=True) promotes cache-backed resources before preserving empties."""
        world = get_world_state()
        entities: list[tuple[int, int, int, int, int]] = [
            (5, 5, 0, 400, 255),
            (6, 5, 0, -1, 255),
        ]
        updated = update_terrain_from_viewport(world, 92, 92, entities, 100000)
        get_world_service().world_state = updated
        get_world_service().mark_pending_radar_empty_delta()

        _handle_radar_ack(get_world_service(), found=True)

        result = get_world_state()
        assert check_and_clear_radar_scan_complete() is True
        assert coord_key(96, 96) in result["containers"]
        assert coord_key(97, 96) in result["containers"]
        assert result["containers"][coord_key(96, 96)]["volume"] == 400
        assert result["containers"][coord_key(97, 96)]["is_fuel"] is False

    def test_empty_delta_found_false_clears(self) -> None:
        """RadarAck(found=False) after empty delta clears viewport."""
        containers = [RadarContainerDict(x=98, y=98, volume=500)]
        update_world_state_from_radar(get_world_service(), containers, [])
        get_world_service().mark_pending_radar_empty_delta()

        _handle_radar_ack(get_world_service(), found=False)

        assert check_and_clear_radar_scan_complete() is True

    def test_no_pending_marks_scan_complete(self) -> None:
        """RadarAck without pending cache or delta just marks scan done."""
        _handle_radar_ack(get_world_service(), found=True)

        assert check_and_clear_radar_scan_complete() is True


class TestClearContainerTileCache:
    """Tests for _clear_container_tile_cache."""

    def setup_method(self) -> None:
        """Reset state before each test."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset state after each test."""
        reset_world_state()

    def test_clears_cache_value(self) -> None:
        """Clearing a tile's cache sets cache_value to 0."""
        world = get_world_state()
        entities: list[tuple[int, int, int, int, int]] = [
            (5, 5, 0, 700, 255),
        ]
        updated = update_terrain_from_viewport(world, 92, 92, entities, 100000)
        get_world_service().world_state = updated
        key = coord_key(96, 96)
        assert get_world_service().world_state["terrain"][key]["cache_value"] == 700

        _clear_container_tile_cache(get_world_service(), 96, 96)

        assert get_world_service().world_state["terrain"][key]["cache_value"] == 0

    def test_no_op_for_missing_tile(self) -> None:
        """Clearing a tile that doesn't exist in terrain is a no-op."""
        _clear_container_tile_cache(get_world_service(), 200, 200)
        # Should not raise


class TestReconcileRadarViewportResources:
    """Tests for _reconcile_radar_viewport_resources."""

    def setup_method(self) -> None:
        """Reset state before each test."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset state after each test."""
        reset_world_state()

    def test_removes_stale_containers_not_in_radar(self) -> None:
        """Containers in radar bounds but absent from new radar are removed."""
        # Set viewport so radar bounds cover (98,98)
        from tankpit_bot.state.viewport_geometry import make_visible_viewport_state

        svc = get_world_service()
        svc.world_state = WorldStateDict(
            **{**svc.world_state, "viewport": make_visible_viewport_state(92, 92)},
        )
        stale = [RadarContainerDict(x=98, y=98, volume=500)]
        update_world_state_from_radar(svc, stale, [])
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


class TestScanViewportFailed:
    """Tests for mark_scan_viewport_failed and is_scan_viewport_failed."""

    def setup_method(self) -> None:
        """Reset state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset state after each test."""
        reset_world_state()

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
