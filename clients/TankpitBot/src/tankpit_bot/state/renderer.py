"""ASCII viewport rendering for world state.

This module provides functions to render the game world state as ASCII art
for debugging and visualization purposes.
"""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.state.types import (
    ASCII_ALLY,
    ASCII_ENEMY,
    ASCII_EQUIPMENT,
    ASCII_FERRY,
    ASCII_FUEL,
    ASCII_GROUND,
    ASCII_MINE,
    ASCII_ROCK,
    ASCII_SELF,
    ASCII_UNKNOWN,
    TERRAIN_FERRY,
    TERRAIN_FERRY_ROCK,
    TERRAIN_GROUND,
    TERRAIN_ROCK_A,
    TERRAIN_ROCK_AB,
    TERRAIN_ROCK_B,
    TankStateDict,
    WorldStateDict,
    coord_key,
)

# =============================================================================
# ASCII Rendering
# =============================================================================


def terrain_to_ascii(terrain_type: int) -> str:
    """Convert terrain type to ASCII character.

    Args:
        terrain_type: Terrain type value (0-7).

    Returns:
        ASCII character for the terrain.
    """
    if terrain_type == TERRAIN_GROUND:
        return ASCII_GROUND
    if terrain_type in (TERRAIN_ROCK_A, TERRAIN_ROCK_B, TERRAIN_ROCK_AB):
        return ASCII_ROCK
    if terrain_type == TERRAIN_FERRY:
        return ASCII_FERRY
    if terrain_type == TERRAIN_FERRY_ROCK:
        return ASCII_ROCK
    return ASCII_UNKNOWN


def _render_cell(
    x: int,
    y: int,
    self_x: int,
    self_y: int,
    self_team: int,
    tank_positions: dict[str, TankStateDict],
    state: WorldStateDict,
    terrain_map: _test_hooks.TerrainMapProtocol,
) -> str:
    """Render a single cell character with priority ordering.

    Priority: self > tanks > mines > containers > terrain.
    """
    key = coord_key(x, y)

    if x == self_x and y == self_y:
        return ASCII_SELF

    if key in tank_positions:
        tank = tank_positions[key]
        return ASCII_ALLY if tank["team"] == self_team else ASCII_ENEMY

    if key in state["mines"]:
        return ASCII_MINE

    if key in state["containers"]:
        container = state["containers"][key]
        return ASCII_FUEL if container["is_fuel"] else ASCII_EQUIPMENT

    return terrain_map.get_terrain(x, y)


def render_world_ascii(state: WorldStateDict, terrain_map: _test_hooks.TerrainMapProtocol) -> str:
    """Render world state as ASCII grid.

    Renders the current viewport with all entities:
    - @ = self
    - T = enemy tank
    - A = ally tank
    - # = rock
    - . = ground
    - W = water
    - ~ = ferry
    - F = fuel container
    - E = equipment container
    - * = mine

    Args:
        state: WorldStateDict to render.
        terrain_map: TerrainMapProtocol for static terrain lookup.

    Returns:
        Multi-line ASCII string representation.
    """
    vp = state["viewport"]
    self_state = state["self_state"]
    self_team = self_state["team"] if self_state else -1
    self_x = self_state["x"] if self_state else -1
    self_y = self_state["y"] if self_state else -1

    lines: list[str] = []

    # Header
    vp_end_x = vp["left"] + vp["width"] - 1
    vp_end_y = vp["top"] + vp["height"] - 1
    lines.append(f"Viewport: ({vp['left']},{vp['top']}) to ({vp_end_x},{vp_end_y})")
    if self_state:
        rank = self_state["rank"]
        fuel = self_state["fuel"]
        lines.append(f"Self: ({self_x},{self_y}) team={self_team} rank={rank} fuel={fuel}")
    lines.append("")

    # Legend
    lines.append("Legend: @=self T=enemy A=ally #=rock .=ground")
    lines.append("        W=water ~=ferry F=fuel E=equip *=mine")
    lines.append("")

    # Column headers
    header = "    "
    for vx in range(vp["width"]):
        x = vp["left"] + vx
        header += f"{x % 10} "
    lines.append(header)

    # Build position lookups for O(1) access
    tank_positions: dict[str, TankStateDict] = {}
    for tank in state["tanks"].values():
        key = coord_key(tank["x"], tank["y"])
        tank_positions[key] = tank

    # Render grid
    for vy in range(vp["height"]):
        y = vp["top"] + vy
        row = f"{y:3d} "

        for vx in range(vp["width"]):
            x = vp["left"] + vx
            cell = _render_cell(
                x,
                y,
                self_x,
                self_y,
                self_team,
                tank_positions,
                state,
                terrain_map,
            )
            row += cell + " "

        lines.append(row)

    # Footer with tank count
    tanks = state["tanks"].values()
    enemy_count = sum(1 for t in tanks if t["team"] != self_team and not t["is_self"])
    ally_count = sum(1 for t in tanks if t["team"] == self_team and not t["is_self"])
    lines.append("")
    tank_total = len(state["tanks"])
    lines.append(f"Tanks: {tank_total} (allies={ally_count}, enemies={enemy_count})")
    lines.append(f"Containers: {len(state['containers'])} Mines: {len(state['mines'])}")

    return "\n".join(lines)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "render_world_ascii",
    "terrain_to_ascii",
]
