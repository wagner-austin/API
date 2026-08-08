"""Tests for viewport dispatch: containers, invalidation, and sweeps.

``test_world_state_dispatch_viewport.py`` was 665 lines; the update
detail is now a sibling.
"""

from __future__ import annotations

from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
from tankpit_bot.state.types import make_viewport_state


class TestDispatchViewportUpdate:
    """Tests for dispatch with ViewportUpdate (0x5A)."""

    def test_dispatch_viewport_update_invalidates_absent_tanks(self) -> None:
        """Viewport update does not invalidate tanks absent from ``0x5A``."""
        from tankpit_bot.protocol import (
            MovementResponseDict,
            TankEntryDict,
            ViewportUpdateDict,
        )

        # Create self at (55, 55)
        ws = WorldService()
        self_msg = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=900,
            x=55,
            y=55,
            direction=0,
            damage_state=0,
            rank=2,
            lb_score=5,
            carrying=0,
        )
        dispatch_world_state_update(ws, self_msg)

        # Place enemy tank at (58, 58) — inside viewport
        entry = TankEntryDict(
            msg_type=0x28, team=0, tank_id=800, rank=0, damage_state=0, score=0, x=58, y=58
        )
        dispatch_world_state_update(ws, entry)
        assert ws.world_state["tanks"]["800"]["x"] == 58

        # Viewport update with only self → buffers entities
        msg = ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=46,
            viewport_top=46,
            entities=[],
        )
        dispatch_world_state_update(ws, msg)

        # Tank 800 invalidated — in viewport but not in entity list
        assert ws.world_state["tanks"]["800"]["x"] == 58
        assert ws.world_state["tanks"]["800"]["y"] == 58

    def test_dispatch_viewport_update_skips_empty_cache_rows(self) -> None:
        """Viewport rows with cache_value=0 do not affect tracked tank positions."""
        from tankpit_bot.protocol import (
            MovementResponseDict,
            TankEntryDict,
            ViewportUpdateDict,
        )
        from tankpit_bot.protocol.types import ViewportEntityDict

        # Create self at (55, 55)
        ws = WorldService()
        self_msg = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=900,
            x=55,
            y=55,
            direction=0,
            damage_state=0,
            rank=2,
            lb_score=5,
            carrying=0,
        )
        dispatch_world_state_update(ws, self_msg)

        # Place enemy at (58, 58) — inside viewport
        entry = TankEntryDict(
            msg_type=0x28, team=0, tank_id=815, rank=0, damage_state=0, score=0, x=58, y=58
        )
        dispatch_world_state_update(ws, entry)

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
        dispatch_world_state_update(ws, msg)

        # Tank position is unchanged because viewport cache rows do not track tanks
        assert ws.world_state["tanks"]["815"]["x"] == 58
        assert ws.world_state["tanks"]["815"]["y"] == 58

    def test_dispatch_viewport_update_keeps_visible_tank(self) -> None:
        """Positive ``0x5A`` rows do not overwrite tracked tank positions."""
        from tankpit_bot.protocol import TankEntryDict, ViewportUpdateDict
        from tankpit_bot.protocol.types import ViewportEntityDict

        ws = WorldService()
        ws.world_state["viewport"] = make_viewport_state(
            left=50,
            top=50,
            width=18,
            height=18,
        )

        # Tank 810 in viewport at (55, 55)
        entry = TankEntryDict(
            msg_type=0x28, team=0, tank_id=810, rank=0, damage_state=0, score=0, x=55, y=55
        )
        dispatch_world_state_update(ws, entry)

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
        dispatch_world_state_update(ws, msg)

        # Tank position preserved
        assert ws.world_state["tanks"]["810"]["x"] == 55
        assert ws.world_state["tanks"]["810"]["y"] == 55


class TestViewportContainerExtraction:
    """Tests for fuel/equipment container extraction from viewport entities."""

    def test_fuel_cache_from_viewport_entity_creates_fuel_container(self) -> None:
        """A 0x5A fuel cache byte lifts directly into world.containers."""
        from tankpit_bot.protocol import MovementResponseDict, ViewportUpdateDict
        from tankpit_bot.protocol.types import ViewportEntityDict

        ws = WorldService()
        self_msg = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=1229,
            x=60,
            y=38,
            direction=0,
            damage_state=0,
            rank=1,
            lb_score=5,
            carrying=0,
        )
        dispatch_world_state_update(ws, self_msg)

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
        dispatch_world_state_update(ws, msg)

        # Fuel container at abs (51+15-1, 29+3-1) = (65, 31) with volume 994
        container = ws.world_state["containers"]["65,31"]
        assert container["is_fuel"] is True
        assert container["volume"] == 994
        assert container["source"] == "viewport"
        assert container["refresh_kind"] == "viewport_patch"

    def test_equipment_cache_from_viewport_entity_creates_equipment_container(self) -> None:
        """A 0x5A cache_value=-1 byte creates an equipment container directly."""
        from tankpit_bot.protocol import MovementResponseDict, ViewportUpdateDict
        from tankpit_bot.protocol.types import ViewportEntityDict

        ws = WorldService()
        self_msg = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=1229,
            x=60,
            y=38,
            direction=0,
            damage_state=0,
            rank=1,
            lb_score=5,
            carrying=0,
        )
        dispatch_world_state_update(ws, self_msg)

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
        dispatch_world_state_update(ws, msg)

        # Equipment container at abs (51+4-1, 29+11-1) = (54, 39)
        container = ws.world_state["containers"]["54,39"]
        assert container["is_fuel"] is False
        assert container["volume"] == 0
        assert container["source"] == "viewport"
        assert container["refresh_kind"] == "viewport_patch"

    def test_empty_viewport_cache_removes_existing_container(self) -> None:
        """A 0x5A cache_value=0 byte removes the container at that tile.

        The 0x5A patch is per-tile authoritative; ``cache_value=0`` is the
        wire's explicit "tile is empty" signal and supersedes any prior
        radar entry (e.g. the container was picked up between the radar
        scan and the next viewport refresh).
        """
        from tankpit_bot.protocol import (
            MovementResponseDict,
            RadarContainerDict,
            ViewportUpdateDict,
        )
        from tankpit_bot.protocol.types import ViewportEntityDict
        from tankpit_bot.sniffer.world_state_radar import update_world_state_from_radar

        ws = WorldService()
        dispatch_world_state_update(
            ws,
            MovementResponseDict(
                msg_type=0x3D,
                team=1,
                tank_id=1229,
                x=60,
                y=38,
                direction=0,
                damage_state=0,
                rank=1,
                lb_score=5,
                carrying=0,
            ),
        )
        self_state = ws.world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        self_state["fuel"] = 400

        update_world_state_from_radar(ws, [RadarContainerDict(x=55, y=33, volume=900)], [], [])

        assert "55,33" in ws.world_state["containers"]

        dispatch_world_state_update(
            ws,
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

        assert "55,33" not in ws.world_state["containers"]
        self_state = ws.world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["fuel"] == 400


class TestViewportInvalidationEdgeCases:
    """Tests for viewport entity processing edge cases."""

    def test_viewport_update_applies_without_self_state(self) -> None:
        """Viewport origin updates even before self movement is known."""
        from tankpit_bot.protocol import ViewportUpdateDict

        ws = WorldService()
        dispatch_world_state_update(
            ws,
            ViewportUpdateDict(
                msg_type=0x5A,
                viewport_left=50,
                viewport_top=50,
                entities=[],
            ),
        )

        assert ws.world_state["viewport"]["left"] == 50
        assert ws.world_state["viewport"]["top"] == 50

    def test_invalidation_skips_already_zeroed_tanks(self) -> None:
        """Tanks already at (0,0) are not re-invalidated."""
        from tankpit_bot.protocol import MovementResponseDict, TankEntryDict, ViewportUpdateDict
        from tankpit_bot.protocol.types import ViewportEntityDict

        # Create self at (55, 55)
        ws = WorldService()
        self_msg = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=950,
            x=55,
            y=55,
            direction=0,
            damage_state=0,
            rank=2,
            lb_score=5,
            carrying=0,
        )
        dispatch_world_state_update(ws, self_msg)

        # Place enemy already at (0,0) — already invalidated
        entry = TankEntryDict(
            msg_type=0x28, team=0, tank_id=951, rank=0, damage_state=0, score=0, x=0, y=0
        )
        dispatch_world_state_update(ws, entry)

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
        dispatch_world_state_update(ws, msg)

        # Zeroed tank stays at (0,0) — was already invalidated, just skipped
        assert ws.world_state["tanks"]["951"]["x"] == 0
        assert ws.world_state["tanks"]["951"]["y"] == 0

    def test_invalidation_skips_tanks_outside_viewport(self) -> None:
        """Tanks outside the viewport bounds are not invalidated."""
        from tankpit_bot.protocol import MovementResponseDict, TankEntryDict, ViewportUpdateDict
        from tankpit_bot.protocol.types import ViewportEntityDict

        # Create self at (55, 55) → viewport is (46, 46) to (64, 64)
        ws = WorldService()
        self_msg = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=960,
            x=55,
            y=55,
            direction=0,
            damage_state=0,
            rank=2,
            lb_score=5,
            carrying=0,
        )
        dispatch_world_state_update(ws, self_msg)

        # Place enemy far away at (200, 200) — outside viewport
        entry = TankEntryDict(
            msg_type=0x28, team=0, tank_id=961, rank=0, damage_state=0, score=0, x=200, y=200
        )
        dispatch_world_state_update(ws, entry)

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
        dispatch_world_state_update(ws, msg)

        # Far-away tank position preserved — not in viewport bounds
        assert ws.world_state["tanks"]["961"]["x"] == 200
        assert ws.world_state["tanks"]["961"]["y"] == 200


class TestViewportPatchSweep:
    """Reset-then-apply: the 0x5A patch's silence removes stale visible entries."""

    def test_landing_patch_sweeps_silent_visible_entries(self) -> None:
        """Visible-layer entries the patch is silent about are removed.

        Mirrors the JS client's reset-then-apply (``Vg.prototype.h``
        wipes the tile grid before rebuilding from the patch). A
        container remembered from a previous visit that the landing
        0x5A does not enumerate is the server saying the tile is
        empty; keeping it produced ghost pickup targets. Radar-sourced
        entries are spared (owned by the radar omission-prune), and
        entries outside the patch bounds are untouched.
        """
        from tankpit_bot.protocol import (
            CacheUpdateDict,
            OverlayUpdateDict,
            RadarContainerDict,
            RadarScanResultDict,
            ViewportUpdateDict,
        )
        from tankpit_bot.protocol.types import ViewportEntityDict

        ws = WorldService()
        # Two visible-layer containers + a mine inside the future viewport.
        dispatch_world_state_update(
            ws, CacheUpdateDict(msg_type=0x43, updates=[(50, 50, 600), (53, 50, 250)])
        )
        dispatch_world_state_update(ws, OverlayUpdateDict(msg_type=0x40, updates=[(51, 50, 1)]))
        # Visible container at the tile the patch WILL enumerate: kept.
        dispatch_world_state_update(ws, CacheUpdateDict(msg_type=0x43, updates=[(48, 48, 100)]))
        # Radar-sourced container inside the future viewport: spared.
        dispatch_world_state_update(
            ws,
            RadarScanResultDict(
                msg_type=0x4F,
                containers=[RadarContainerDict(x=52, y=50, volume=300)],
                mines=[],
                mine_clears=[],
            ),
        )
        # Visible-layer container far outside the future viewport: untouched.
        dispatch_world_state_update(ws, CacheUpdateDict(msg_type=0x43, updates=[(200, 200, 400)]))
        assert "50,50" in ws.world_state["containers"]
        assert "51,50" in ws.world_state["mines"]

        # Landing patch for the (46,46) viewport enumerates ONE tile --
        # a fresh container at (48,48) -- and is silent about the rest.
        enumerated = ViewportEntityDict(
            col=3,
            row=3,
            cache_value=700,
            overlay_value=255,
            terrain_type=0,
        )
        dispatch_world_state_update(
            ws,
            ViewportUpdateDict(
                msg_type=0x5A,
                viewport_left=46,
                viewport_top=46,
                entities=[enumerated],
            ),
        )

        containers = ws.world_state["containers"]
        mines = ws.world_state["mines"]
        # Silent visible-layer entries inside the patch: swept.
        assert "50,50" not in containers
        assert "53,50" not in containers
        assert "51,50" not in mines
        # Enumerated tile: present with the patch's value.
        assert containers["48,48"]["volume"] == 700
        # Radar-sourced entry: spared.
        assert containers["52,50"]["source"] == "radar"
        # Outside the patch bounds: untouched.
        assert containers["200,200"]["volume"] == 400

    def test_landing_patch_with_nothing_stale_leaves_registries_alone(self) -> None:
        """A patch over ground with no stale entries changes nothing."""
        from tankpit_bot.protocol import CacheUpdateDict, ViewportUpdateDict

        ws = WorldService()
        dispatch_world_state_update(ws, CacheUpdateDict(msg_type=0x43, updates=[(200, 200, 400)]))

        dispatch_world_state_update(
            ws,
            ViewportUpdateDict(
                msg_type=0x5A,
                viewport_left=46,
                viewport_top=46,
                entities=[],
            ),
        )

        assert ws.world_state["containers"]["200,200"]["volume"] == 400

    def test_landing_patch_sweeps_stale_mine_without_stale_containers(self) -> None:
        """A sweep that only removes a mine still rewrites the registry."""
        from tankpit_bot.protocol import OverlayUpdateDict, ViewportUpdateDict

        ws = WorldService()
        dispatch_world_state_update(ws, OverlayUpdateDict(msg_type=0x40, updates=[(51, 50, 1)]))
        assert "51,50" in ws.world_state["mines"]

        dispatch_world_state_update(
            ws,
            ViewportUpdateDict(
                msg_type=0x5A,
                viewport_left=46,
                viewport_top=46,
                entities=[],
            ),
        )

        assert "51,50" not in ws.world_state["mines"]
