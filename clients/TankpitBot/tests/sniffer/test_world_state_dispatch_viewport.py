"""Tests for sniffer world state dispatch handling of viewport messages."""

from __future__ import annotations

from tankpit_bot.sniffer import (
    dispatch_world_state_update,
    reset_world_state,
)
from tankpit_bot.sniffer.world_state import get_world_service


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
        dispatch_world_state_update(get_world_service(), self_msg)

        # Place enemy tank at (58, 58) — inside viewport
        entry = TankEntryDict(msg_type=0x28, tank_id=800, x=58, y=58, name="Ghost")
        dispatch_world_state_update(get_world_service(), entry)
        assert get_world_service().world_state["tanks"]["800"]["x"] == 58

        # Viewport update with only self → buffers entities
        msg = ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=46,
            viewport_top=46,
            entities=[],
        )
        dispatch_world_state_update(get_world_service(), msg)

        # Tank 800 invalidated — in viewport but not in entity list
        assert get_world_service().world_state["tanks"]["800"]["x"] == 58
        assert get_world_service().world_state["tanks"]["800"]["y"] == 58

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
        dispatch_world_state_update(get_world_service(), self_msg)

        # Place enemy at (58, 58) — inside viewport
        entry = TankEntryDict(msg_type=0x28, tank_id=815, x=58, y=58, name="Target")
        dispatch_world_state_update(get_world_service(), entry)

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
        dispatch_world_state_update(get_world_service(), msg)

        # Tank position is unchanged because viewport cache rows do not track tanks
        assert get_world_service().world_state["tanks"]["815"]["x"] == 58
        assert get_world_service().world_state["tanks"]["815"]["y"] == 58

    def test_dispatch_viewport_update_keeps_visible_tank(self) -> None:
        """Positive ``0x5A`` rows do not overwrite tracked tank positions."""
        from tankpit_bot.protocol import TankEntryDict, ViewportUpdateDict
        from tankpit_bot.protocol.types import ViewportEntityDict
        from tankpit_bot.state.types import ViewportStateDict

        get_world_service().world_state["viewport"] = ViewportStateDict(
            left=50,
            top=50,
            width=18,
            height=18,
        )

        # Tank 810 in viewport at (55, 55)
        entry = TankEntryDict(msg_type=0x28, tank_id=810, x=55, y=55, name="Visible")
        dispatch_world_state_update(get_world_service(), entry)

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
        dispatch_world_state_update(get_world_service(), msg)

        # Tank position preserved
        assert get_world_service().world_state["tanks"]["810"]["x"] == 55
        assert get_world_service().world_state["tanks"]["810"]["y"] == 55

    def test_dispatch_viewport_update_skips_self_and_zeroed_tanks(self) -> None:
        """Dispatch 0x5A skips self tank and already-invalidated (0,0) tanks."""
        from tankpit_bot.protocol import (
            MovementResponseDict,
            TankEntryDict,
            ViewportUpdateDict,
        )
        from tankpit_bot.state.types import ViewportStateDict

        get_world_service().world_state["viewport"] = ViewportStateDict(
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
        dispatch_world_state_update(get_world_service(), self_msg)

        # Create enemy already at (0, 0) — already invalidated
        entry = TankEntryDict(msg_type=0x28, tank_id=821, x=0, y=0, name="Zeroed")
        dispatch_world_state_update(get_world_service(), entry)

        # Viewport update with no entities — should skip self and zeroed tank
        msg = ViewportUpdateDict(msg_type=0x5A, viewport_left=50, viewport_top=50, entities=[])
        dispatch_world_state_update(get_world_service(), msg)

        # Self position preserved (skipped because is_self)
        self_state = get_world_service().world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["x"] == 55

    def test_dispatch_viewport_update_initializes_from_packet_origin(self) -> None:
        """Dispatch 0x5A uses packet origin even when prior viewport is default."""
        from tankpit_bot.protocol import TankEntryDict, ViewportUpdateDict
        from tankpit_bot.sniffer.viewport import get_viewport_left, get_viewport_top

        # Default viewport: left=0, top=0, width=18 is still a valid origin.
        entry = TankEntryDict(msg_type=0x28, tank_id=801, x=5, y=5, name="Safe")
        dispatch_world_state_update(get_world_service(), entry)

        msg = ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=0,
            viewport_top=0,
            entities=[],
        )
        dispatch_world_state_update(get_world_service(), msg)

        assert get_world_service().world_state["viewport"]["left"] == 0
        assert get_world_service().world_state["viewport"]["top"] == 0
        assert get_viewport_left() == 0
        assert get_viewport_top() == 0
        assert get_world_service().world_state["tanks"]["801"]["x"] == 5
        assert get_world_service().world_state["tanks"]["801"]["y"] == 5

    def test_dispatch_viewport_update_does_not_mark_viewport_confirmed(self) -> None:
        """Dispatch 0x5A alone should not count as a radar-confirmed scan."""
        from tankpit_bot.protocol import ViewportUpdateDict

        msg = ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=51,
            viewport_top=29,
            entities=[],
        )

        dispatch_world_state_update(get_world_service(), msg)

        assert "51,29" not in get_world_service().world_state["scanned_viewports"]

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
        dispatch_world_state_update(get_world_service(), msg)

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
        dispatch_world_state_update(get_world_service(), self_msg)

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
        dispatch_world_state_update(get_world_service(), msg)

        # Fuel container at abs (51+15-1, 29+3-1) = (65, 31) with volume ≈ 994
        tile = get_world_service().world_state["terrain"]["65,31"]
        assert tile["cache_value"] == 994
        assert "65,31" not in get_world_service().world_state["containers"]

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
        dispatch_world_state_update(get_world_service(), self_msg)

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
        dispatch_world_state_update(get_world_service(), msg)

        # Equipment container at abs (51+4-1, 29+11-1) = (54, 39)
        tile = get_world_service().world_state["terrain"]["54,39"]
        assert tile["cache_value"] == -1
        assert "54,39" not in get_world_service().world_state["containers"]

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
        dispatch_world_state_update(get_world_service(), self_msg)

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
        dispatch_world_state_update(get_world_service(), msg)

        # Viewport offset = (60-9, 38-9) = (51, 29). Abs pos = (51+5-1, 29+5-1) = (55, 33).
        tile = get_world_service().world_state["terrain"]["55,33"]
        assert tile["cache_value"] == 999
        assert "55,33" not in get_world_service().world_state["containers"]

    def test_empty_viewport_cache_does_not_override_radar_container(self) -> None:
        """A 0x5A cache clear does not override radar-confirmed container truth."""
        from tankpit_bot.container import RadarContainerDict
        from tankpit_bot.protocol import MovementResponseDict, ViewportUpdateDict
        from tankpit_bot.protocol.types import ViewportEntityDict
        from tankpit_bot.sniffer.world_state_radar import update_world_state_from_radar

        dispatch_world_state_update(
            get_world_service(),
            MovementResponseDict(
                msg_type=0x3D,
                team=1,
                tank_id=1229,
                x=60,
                y=38,
                direction=0,
                rank=1,
                leaderboard_position=5,
            ),
        )
        self_state = get_world_service().world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        self_state["fuel"] = 400

        update_world_state_from_radar(
            get_world_service(), [RadarContainerDict(x=55, y=33, volume=900)], []
        )

        assert "55,33" in get_world_service().world_state["containers"]

        dispatch_world_state_update(
            get_world_service(),
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
            ),
        )

        assert get_world_service().world_state["terrain"]["55,33"]["cache_value"] == 0
        assert "55,33" in get_world_service().world_state["containers"]
        assert get_world_service().world_state["containers"]["55,33"]["volume"] == 900
        self_state = get_world_service().world_state["self_state"]
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
            get_world_service(),
            ViewportUpdateDict(
                msg_type=0x5A,
                viewport_left=50,
                viewport_top=50,
                entities=[],
            ),
        )

        assert get_world_service().world_state["viewport"]["left"] == 50
        assert get_world_service().world_state["viewport"]["top"] == 50

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
        dispatch_world_state_update(get_world_service(), self_msg)

        # Place enemy already at (0,0) — already invalidated
        entry = TankEntryDict(msg_type=0x28, tank_id=951, x=0, y=0, name="Zeroed")
        dispatch_world_state_update(get_world_service(), entry)

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
        dispatch_world_state_update(get_world_service(), msg)

        # Zeroed tank stays at (0,0) — was already invalidated, just skipped
        assert get_world_service().world_state["tanks"]["951"]["x"] == 0
        assert get_world_service().world_state["tanks"]["951"]["y"] == 0

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
        dispatch_world_state_update(get_world_service(), self_msg)

        # Place enemy far away at (200, 200) — outside viewport
        entry = TankEntryDict(msg_type=0x28, tank_id=961, x=200, y=200, name="FarAway")
        dispatch_world_state_update(get_world_service(), entry)

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
        dispatch_world_state_update(get_world_service(), msg)

        # Far-away tank position preserved — not in viewport bounds
        assert get_world_service().world_state["tanks"]["961"]["x"] == 200
        assert get_world_service().world_state["tanks"]["961"]["y"] == 200
