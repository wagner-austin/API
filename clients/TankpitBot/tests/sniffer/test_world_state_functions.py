"""Tests for sniffer world state module-level functions."""

from __future__ import annotations

from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_radar import update_world_state_from_radar
from tankpit_bot.state.viewport_geometry import (
    make_visible_viewport_state,
    viewport_radar_bounds,
    viewport_visible_bounds,
)


class TestViewportGeometry:
    """Tests for visible viewport and radar envelope geometry."""

    def test_radar_bounds_extend_visible_viewport_by_one_tile(self) -> None:
        """Radar bounds extend one tile beyond the visible viewport."""
        ws = WorldService()
        viewport = make_visible_viewport_state(140, 149)
        ws.world_state["viewport"] = viewport

        assert viewport_visible_bounds(viewport) == (140, 149, 155, 164)
        assert viewport_radar_bounds(viewport) == (139, 148, 156, 165)

    def test_viewport_bounds_delegates_to_geometry(self) -> None:
        """WorldService.viewport_bounds() returns the visible viewport."""
        ws = WorldService()
        viewport = make_visible_viewport_state(140, 149)
        ws.world_state["viewport"] = viewport

        assert ws.viewport_bounds() == (140, 149, 155, 164)

    def test_patch_coords_translate_from_radar_margin(self) -> None:
        """Patch col/row 0 is the radar margin; visible viewport starts at 1."""
        from tankpit_bot.state.viewport_geometry import viewport_patch_world_coords

        assert viewport_patch_world_coords(140, 149, 0, 0) == (139, 148)
        assert viewport_patch_world_coords(140, 149, 1, 1) == (140, 149)
        assert viewport_patch_world_coords(140, 149, 16, 16) == (155, 164)


class TestWorldStateGetter:
    """Tests for get_world_state function."""

    def test_get_world_state_returns_current_state(self) -> None:
        """Test get_world_state returns the current world state."""

        # After reset, state should have None self_state
        ws = WorldService()
        state = ws.get_world_state()
        assert state["self_state"] is None
        assert state["containers"] == {}
        assert state["mines"] == {}

        # Update position and verify state is updated
        ws.update_world_state_from_position(50, 60)
        state = ws.get_world_state()
        if state["self_state"] is None:
            raise AssertionError("self_state should not be None")
        assert state["self_state"]["x"] == 50
        assert state["self_state"]["y"] == 60


# TestContainerUpdate class deleted 2026-06-20 with its sole tests
# (update_world_state_from_tank_registry_container); container
# TankRegistry was removed after corpus sweep proved zero production
# fires.


class TestFuelUpdate:
    """Tests for fuel update functions."""

    def test_update_world_state_from_fuel_total(self) -> None:
        """Test fuel total sets self_state fuel to absolute value."""
        from tankpit_bot.sniffer.world_state_containers import update_world_state_from_fuel_total

        # First set up a position to create self_state
        ws = WorldService()
        ws.update_world_state_from_position(100, 100)

        # Update fuel - sets to absolute value
        update_world_state_from_fuel_total(ws, 50)

        state = ws.world_state
        if state["self_state"] is None:
            raise AssertionError("self_state should not be None")
        assert state["self_state"]["fuel"] == 50

    def test_fuel_underflow_is_the_self_death_receipt(self) -> None:
        """A u16-wrapped reading books a self-death and ingests zero.

        Arterial's two main-map deaths (2026-08-26 18:42:01 and
        18:45:58, readings 65475 and 65530) were invisible: no 0x41
        deactivation arrives for self on a Normal field, and the
        belief ingested 65k fuel as if it were a pickup. The wrap IS
        the deactivation receipt.
        """
        from tankpit_bot.sniffer.world_state_containers import update_world_state_from_fuel_total

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        update_world_state_from_fuel_total(ws, 29)

        update_world_state_from_fuel_total(ws, 65475)

        state = ws.world_state["self_state"]
        if state is None:
            raise AssertionError("self_state should not be None")
        assert state["fuel"] == 0
        # The garbage reading never reaches the fuel book as a pickup.
        assert ws.fuel_book["last_fuel"] != 65475

    def test_update_world_state_from_fuel_total_no_self_state(self) -> None:
        """Test fuel total does nothing when self_state is None."""
        from tankpit_bot.sniffer.world_state_containers import update_world_state_from_fuel_total

        # Reset to ensure no self_state

        # Verify self_state is None
        ws = WorldService()
        state = ws.world_state
        assert state["self_state"] is None

        # Update fuel - should do nothing since no self_state
        update_world_state_from_fuel_total(ws, 50)

        state = ws.world_state
        assert state["self_state"] is None


class TestContainerPickup:
    """Tests for container pickup functions."""

    def test_update_world_state_from_container_pickup(self) -> None:
        """Test container pickup removes container and adds fuel."""
        from tankpit_bot.protocol import RadarContainerDict, RadarMineDict
        from tankpit_bot.sniffer.world_state_containers import (
            update_world_state_from_container_pickup,
        )

        # First set up a position to create self_state
        ws = WorldService()
        ws.update_world_state_from_position(100, 100)

        # Add a container via radar
        containers: list[RadarContainerDict] = [RadarContainerDict(x=50, y=60, volume=100)]
        mines: list[RadarMineDict] = []
        update_world_state_from_radar(ws, containers, mines, [])

        state = ws.world_state
        assert "50,60" in state["containers"]

        # Pick up the container
        update_world_state_from_container_pickup(ws, 50, 60)

        state = ws.world_state
        # Container should be removed
        assert "50,60" not in state["containers"]


class TestRadarViewportReconciliation:
    """Tests for radar-confirmed viewport resource reconciliation."""

    def test_radar_marks_current_viewport_as_scanned(self) -> None:
        """Radar records authoritative coverage for the current viewport origin."""
        from tankpit_bot.protocol import RadarContainerDict

        ws = WorldService()
        ws.world_state["viewport"]["left"] = 100
        ws.world_state["viewport"]["top"] = 200

        update_world_state_from_radar(ws, [RadarContainerDict(x=101, y=201, volume=500)], [], [])

        assert ws.world_state["scanned_tiles"]["100,200"] > 0

    def test_radar_clears_failed_scan_mark_for_current_viewport(self) -> None:
        """Successful radar clears the recent failed-scan quarantine."""
        from tankpit_bot.protocol import RadarContainerDict

        ws = WorldService()
        ws.world_state["viewport"]["left"] = 100
        ws.world_state["viewport"]["top"] = 200
        ws.mark_scan_viewport_failed(100, 200, 1000)

        assert ws.is_scan_viewport_failed(100, 200, 1001) is True

        update_world_state_from_radar(ws, [RadarContainerDict(x=101, y=201, volume=500)], [], [])

        assert ws.is_scan_viewport_failed(100, 200, 1001) is False

    def test_radar_clears_missing_current_viewport_containers(self) -> None:
        """Radar removes stale current-viewport containers not returned by the scan."""
        from tankpit_bot.protocol import RadarContainerDict
        from tankpit_bot.state.types import make_container_state

        ws = WorldService()
        ws.world_state["viewport"]["left"] = 100
        ws.world_state["viewport"]["top"] = 200
        ws.world_state["containers"]["101,201"] = make_container_state(
            x=101,
            y=201,
            is_fuel=True,
            volume=500,
        )
        ws.world_state["containers"]["150,150"] = make_container_state(
            x=150,
            y=150,
            is_fuel=True,
            volume=600,
        )

        update_world_state_from_radar(ws, [RadarContainerDict(x=102, y=202, volume=700)], [], [])

        assert "101,201" not in ws.world_state["containers"]
        assert "102,202" in ws.world_state["containers"]
        assert "150,150" in ws.world_state["containers"]

    def test_radar_clears_multiple_missing_current_viewport_containers(self) -> None:
        """Radar removes each stale container after the first deletion snapshot."""
        from tankpit_bot.state.types import make_container_state

        ws = WorldService()
        ws.world_state["viewport"]["left"] = 100
        ws.world_state["viewport"]["top"] = 200
        ws.world_state["containers"]["101,201"] = make_container_state(
            x=101,
            y=201,
            is_fuel=True,
            volume=500,
        )
        ws.world_state["containers"]["102,202"] = make_container_state(
            x=102,
            y=202,
            is_fuel=True,
            volume=600,
        )

        update_world_state_from_radar(ws, [], [], [])

        assert "101,201" not in ws.world_state["containers"]
        assert "102,202" not in ws.world_state["containers"]

    def test_radar_clears_missing_current_viewport_mines(self) -> None:
        """Radar removes stale radar-sourced viewport mines not returned by the scan.

        Only radar-sourced entries are reconciled: the radar response
        lists just the newly revealed hidden entities, so visible
        (viewport/placement-sourced) mines are never removed by it
        (2026-07-01).
        """
        from tankpit_bot.protocol import RadarMineDict
        from tankpit_bot.state.types import make_mine_state

        ws = WorldService()
        ws.world_state["viewport"]["left"] = 100
        ws.world_state["viewport"]["top"] = 200
        ws.world_state["mines"]["101,201"] = make_mine_state(
            x=101,
            y=201,
            mine_type=1,
            tank_id=77,
            team=2,
            source="radar",
        )
        ws.world_state["mines"]["150,150"] = make_mine_state(
            x=150,
            y=150,
            mine_type=1,
            tank_id=88,
            team=3,
            source="radar",
        )

        update_world_state_from_radar(ws, [], [RadarMineDict(x=102, y=202, team=1)], [])

        assert "101,201" not in ws.world_state["mines"]
        assert "102,202" in ws.world_state["mines"]
        assert "150,150" in ws.world_state["mines"]

    def test_radar_clears_multiple_missing_current_viewport_mines(self) -> None:
        """Radar removes each stale radar-sourced mine after the first deletion snapshot."""
        from tankpit_bot.state.types import make_mine_state

        ws = WorldService()
        ws.world_state["viewport"]["left"] = 100
        ws.world_state["viewport"]["top"] = 200
        ws.world_state["mines"]["101,201"] = make_mine_state(
            x=101,
            y=201,
            mine_type=1,
            tank_id=77,
            team=2,
            source="radar",
        )
        ws.world_state["mines"]["102,202"] = make_mine_state(
            x=102,
            y=202,
            mine_type=1,
            tank_id=88,
            team=3,
            source="radar",
        )

        update_world_state_from_radar(ws, [], [], [])

        assert "101,201" not in ws.world_state["mines"]
        assert "102,202" not in ws.world_state["mines"]

    def test_radar_reconciliation_is_noop_when_viewport_already_matches(self) -> None:
        """Radar reconciliation preserves resources when scan exactly matches state."""
        from tankpit_bot.protocol import RadarContainerDict, RadarMineDict
        from tankpit_bot.state.types import make_container_state, make_mine_state

        ws = WorldService()
        ws.world_state["viewport"]["left"] = 100
        ws.world_state["viewport"]["top"] = 200
        ws.world_state["containers"]["101,201"] = make_container_state(
            x=101,
            y=201,
            is_fuel=True,
            volume=500,
        )
        ws.world_state["mines"]["102,202"] = make_mine_state(
            x=102,
            y=202,
            mine_type=1,
            tank_id=77,
            team=2,
        )

        update_world_state_from_radar(
            ws,
            [RadarContainerDict(x=101, y=201, volume=500)],
            [RadarMineDict(x=102, y=202, team=2)],
            [],
        )

        assert "101,201" in ws.world_state["containers"]
        assert "102,202" in ws.world_state["mines"]
