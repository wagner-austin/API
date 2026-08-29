"""Tests for the equipment atlas: loader narrowing and the hop planner."""

from __future__ import annotations

from pathlib import Path

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import ReadTextProtocol
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.equipment_atlas import (
    ATLAS_VISIT_TTL_MS,
    atlas_tiles_for,
    plan_atlas_equipment_hop,
)
from tankpit_bot.bot.ai.tactics import combat_radar_min
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.physics.capacity import inventory_capacity
from tankpit_bot.sniffer.world_service import WorldService
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap

_ATLAS_TEXT = (
    '{"field01.gif": [[123, 123, 104], [180, 47, 40], [10, 10, 1]], "field05.gif": [[58, 170, 8]]}'
)


def _fake_atlas_read(text: str = _ATLAS_TEXT) -> ReadTextProtocol:
    """Return a read_text fake serving the synthetic atlas payload.

    Args:
        text: Atlas JSON text to serve.

    Returns:
        The fake callable.
    """

    def read(path: Path) -> str:
        return text

    return read


def _atlas_ws(field: str = "field01.gif") -> WorldService:
    """Return a world service joined to a room on the given field.

    Args:
        field: Field image the selected room maps to.

    Returns:
        The world service.
    """
    ws = WorldService()
    ws.register_room_image("7", field)
    ws.selected_room = "7"
    return ws


def _collect_ctx(
    ws: WorldService,
    *,
    radar_count: int = 0,
    fuel: int = 900,
    terrain: InMemoryTerrainMap | None = None,
) -> DecideCtx:
    """Build a COLLECT-mode context with a radar-deficient inventory.

    Args:
        ws: World service (carries the atlas cache and visit marks).
        radar_count: Extra radars stocked.
        fuel: Current fuel.
        terrain: Terrain map; default is all-passable.

    Returns:
        Decision context standing at (100, 100).
    """
    world, self_state = make_world(fuel=fuel)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    inventory["extra_radars"]["count"] = radar_count
    return DecideCtx(
        world,
        self_state,
        ai_state,
        inventory,
        100000,
        terrain if terrain is not None else InMemoryTerrainMap(),
        "",
        ws=ws,
    )


class TestAtlasLoader:
    """Tests for atlas_tiles_for narrowing and caching."""

    def test_loads_filters_and_caches_the_field_rows(self) -> None:
        """Rows below the min-runs floor drop; the load happens once."""
        ws = _atlas_ws()
        original = _test_hooks.read_text
        _test_hooks.read_text = _fake_atlas_read()
        try:
            tiles = atlas_tiles_for(ws)
        finally:
            _test_hooks.read_text = original

        assert tiles == [(123, 123, 104), (180, 47, 40)]
        # Cached: a second call never touches the file again.
        assert atlas_tiles_for(ws) == [(123, 123, 104), (180, 47, 40)]

    def test_unknown_field_is_an_empty_atlas(self) -> None:
        """A field the corpus never mapped yields no tiles."""
        ws = _atlas_ws("field99.gif")
        original = _test_hooks.read_text
        _test_hooks.read_text = _fake_atlas_read()
        try:
            assert atlas_tiles_for(ws) == []
        finally:
            _test_hooks.read_text = original

    @pytest.mark.parametrize(
        "text",
        [
            "[1, 2]",
            '{"field01.gif": 7}',
            '{"field01.gif": [[1, 2]]}',
            '{"field01.gif": [[1, 2, true]]}',
        ],
    )
    def test_malformed_atlas_payloads_raise(self, text: str) -> None:
        """Shape violations fail loud, never load as garbage."""
        ws = _atlas_ws()
        original = _test_hooks.read_text
        _test_hooks.read_text = _fake_atlas_read(text)
        try:
            with pytest.raises(ValueError):
                atlas_tiles_for(ws)
        finally:
            _test_hooks.read_text = original


class TestAtlasHopPlanner:
    """Tests for plan_atlas_equipment_hop."""

    def test_deficient_radar_hops_to_the_best_hotspot(self) -> None:
        """Below the hunt bar, the planner teleports to atlas ground.

        Score is persistence over distance: from (100,100) the
        104-run tile at (123,123) beats the 40-run tile at (180,47).
        """
        ws = _atlas_ws()
        original = _test_hooks.read_text
        _test_hooks.read_text = _fake_atlas_read()
        try:
            ctx = _collect_ctx(ws)
            decision = plan_atlas_equipment_hop(ctx, ctx.base)
        finally:
            _test_hooks.read_text = original

        if decision is None:
            raise AssertionError("expected an atlas hop decision")
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 123
        assert decision["command"]["target_y"] == 123
        assert decision["behavior"]["reason_kind"] == "atlas_equipment_hop"
        assert "123,123" in ws.atlas_visited

    def test_restock_bars_met_means_no_hop(self) -> None:
        """With every bar met the atlas never wins a tick."""
        ws = _atlas_ws()
        ws.equipment_atlas = [(123, 123, 104)]
        ctx = _collect_ctx(ws)
        rank_cap = inventory_capacity(ctx.self_state["rank"])
        ctx.inventory["dual_shots"]["count"] = rank_cap
        ctx.inventory["homing_shots"]["count"] = rank_cap
        ctx.inventory["extra_radars"]["count"] = combat_radar_min(ctx.self_state["rank"])

        assert plan_atlas_equipment_hop(ctx, ctx.base) is None

    def test_missing_terrain_declines(self) -> None:
        """Without terrain no landing can be vetted; the hop declines."""
        ws = _atlas_ws()
        ws.equipment_atlas = [(123, 123, 104)]
        ctx = _collect_ctx(ws)
        ctx.terrain = None

        assert plan_atlas_equipment_hop(ctx, ctx.base) is None

    def test_empty_atlas_declines(self) -> None:
        """A field with no mined tiles produces no hop."""
        ws = _atlas_ws()
        ws.equipment_atlas = []
        ctx = _collect_ctx(ws)

        assert plan_atlas_equipment_hop(ctx, ctx.base) is None

    def test_visited_and_hostile_tiles_are_skipped(self) -> None:
        """Session tombstones and landing evidence both veto a hotspot."""
        ws = _atlas_ws()
        ws.equipment_atlas = [(123, 123, 104), (180, 47, 40)]
        ws.atlas_visited["123,123"] = 100000 - 1000
        ws.mark_landing_refused(180, 47, 5, 100000 - 1000)
        ctx = _collect_ctx(ws)

        assert plan_atlas_equipment_hop(ctx, ctx.base) is None

    def test_visit_tombstones_expire(self) -> None:
        """After the TTL the circuit returns to a previously-empty spot."""
        ws = _atlas_ws()
        ws.equipment_atlas = [(123, 123, 104)]
        ws.atlas_visited["123,123"] = 100000 - ATLAS_VISIT_TTL_MS - 1
        ctx = _collect_ctx(ws)

        decision = plan_atlas_equipment_hop(ctx, ctx.base)

        if decision is None:
            raise AssertionError("expected the expired tombstone to free the hotspot")
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 123
        assert decision["command"]["target_y"] == 123

    def test_impassable_and_unaffordable_hotspots_are_skipped(self) -> None:
        """Water hotspots and out-of-budget hops never dispatch."""
        ws = _atlas_ws()
        ws.equipment_atlas = [(123, 123, 104), (250, 250, 90)]
        terrain = InMemoryTerrainMap(terrain_data={(123, 123): "W"})
        # Budget: fuel 300 minus the low threshold leaves too little
        # for the far corner hop (cost ~6 * 212).
        ctx = _collect_ctx(ws, fuel=300, terrain=terrain)

        assert plan_atlas_equipment_hop(ctx, ctx.base) is None

    def test_standing_beside_an_empty_hotspot_tombstones_it(self) -> None:
        """Adjacency with nothing believed there proves the spot empty."""
        ws = _atlas_ws()
        ws.equipment_atlas = [(101, 100, 104)]
        ctx = _collect_ctx(ws)

        assert plan_atlas_equipment_hop(ctx, ctx.base) is None
        assert "101,100" in ws.atlas_visited


def test_collect_cascade_serves_the_atlas_hop() -> None:
    """End to end: with nothing believed collectible, COLLECT hops the atlas.

    The cascade position pins the doctrine: known stock preempts the
    hop, and the hop preempts the quad sweep.
    """
    from tankpit_bot.bot.ai.collect_mode import decide_collect_mode

    ws = _atlas_ws()
    ws.equipment_atlas = [(123, 123, 104)]
    ctx = _collect_ctx(ws)

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected the cascade to produce the atlas hop")
    assert decision["behavior"]["reason_kind"] == "atlas_equipment_hop"
    assert decision["command"]["cmd_type"] == "teleport"
