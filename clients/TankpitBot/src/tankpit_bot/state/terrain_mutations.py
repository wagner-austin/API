"""Mutations to the terrain grid.

Folds a viewport patch into the known terrain. Sibling of
:mod:`tankpit_bot.state.self_mutations`,
:mod:`tankpit_bot.state.tank_mutations`, and
:mod:`tankpit_bot.state.container_mutations`.
"""

from __future__ import annotations

from tankpit_bot.state.types import (
    WorldStateDict,
    coord_key,
    make_terrain_tile,
)
from tankpit_bot.state.viewport_geometry import (
    make_visible_viewport_state,
    viewport_patch_world_coords,
)


def update_terrain_from_viewport(
    state: WorldStateDict,
    viewport_left: int,
    viewport_top: int,
    entities: list[tuple[int, int, int, int, int]],
    timestamp_ms: int,
) -> WorldStateDict:
    """Update terrain from a visible viewport update.

    A 0x5A viewport patch carries terrain plus container (``cache_value``)
    and mine (``overlay_value``) layers; the production wire path lifts
    those into ``world.containers`` and ``world.mines`` via the per-tile
    mutators. This helper exists for tests that need to pre-seed terrain
    from a synthetic 0x5A patch -- it ignores the container / mine bytes
    so the rich registries are populated by the explicit per-tile mutators
    in tests that exercise them.

    Args:
        state: Current world state.
        viewport_left: Viewport left X coordinate.
        viewport_top: Viewport top Y coordinate.
        entities: List of ``0x5A`` patch-grid
            ``(col, row, terrain_type, cache_value, overlay_value)`` tuples.
            ``cache_value`` and ``overlay_value`` are accepted for
            wire-shape compatibility but ignored here.
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with updated terrain and viewport.
    """
    new_terrain = dict(state["terrain"])
    new_viewport = make_visible_viewport_state(viewport_left, viewport_top, timestamp_ms)

    for col, row, terrain_type, _cache_value, _overlay_value in entities:
        x, y = viewport_patch_world_coords(viewport_left, viewport_top, col, row)
        key = coord_key(x, y)
        new_terrain[key] = make_terrain_tile(
            x=x,
            y=y,
            terrain_type=terrain_type,
            observed_ms=timestamp_ms,
        )

    return WorldStateDict(
        self_state=state["self_state"],
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=new_terrain,
        viewport=new_viewport,
        scanned_tiles=state["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


__all__ = [
    "update_terrain_from_viewport",
]
