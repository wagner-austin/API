"""Shared builders for the resource-search tests."""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.sniffer.world_service import WorldService
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)


def _fully_cover_viewport_origins(
    origins: list[tuple[int, int]],
    *,
    timestamp_ms: int = 100000,
    width: int = 16,
    height: int = 16,
) -> dict[str, int]:
    """Return a tile-coverage dict that fully covers every supplied 16x16 viewport."""
    covered: dict[str, int] = {}
    for left, top in origins:
        for y in range(top, top + height):
            for x in range(left, left + width):
                covered[f"{x},{y}"] = timestamp_ms
    return covered


def _ctx(
    *,
    self_x: int = 100,
    self_y: int = 100,
    fuel: int = 800,
    terrain: TerrainMapProtocol | None = None,
    ai_state: AIStateDict | None = None,
    scanned_viewport_origins: list[tuple[int, int]] | None = None,
    map_fuel_dots: tuple[tuple[int, int], ...] = (),
) -> DecideCtx:
    """Build a DecideCtx with a clean default world and optional overrides."""
    ws = WorldService()
    world, self_state = make_world(self_x=self_x, self_y=self_y, fuel=fuel)
    if scanned_viewport_origins:
        world["scanned_tiles"].update(_fully_cover_viewport_origins(scanned_viewport_origins))
    return DecideCtx(
        world,
        self_state,
        ai_state if ai_state is not None else make_scanned_ai_state(),
        make_inventory(),
        100000,
        terrain,
        "",
        map_fuel_dots,
        ws=ws,
    )
