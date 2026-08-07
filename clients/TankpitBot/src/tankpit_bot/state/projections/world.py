"""Phase 1d: project self / mine / terrain / viewport states as Facts.

Same flat-carry pattern as :mod:`tankpit_bot.state.projections.container`
and :mod:`tankpit_bot.state.projections.tank`: the world-state TypedDicts
carry the fact metadata flat, and these projections build true
:class:`~tankpit_bot.facts.fact.Fact` views for Fact-consuming layers.

This module is intentionally NOT re-exported from
``tankpit_bot.facts`` to keep the import graph acyclic.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from tankpit_bot.facts.fact import Fact, make_fact
from tankpit_bot.state.types.mine import MineStateDict
from tankpit_bot.state.types.self_state import SelfStateDict
from tankpit_bot.state.types.terrain import TerrainTileDict
from tankpit_bot.state.types.viewport import ViewportStateDict


class SelfValueDict(TypedDict):
    """The believed value of the self-state fact (metadata stripped).

    Attributes:
        tank_id: Player's tank ID.
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        team: Team ID (0-3).
        rank: Military rank (0 recruit .. 8 general).
        fuel: Current fuel (also health).
        leaderboard_position: Position on leaderboard.
    """

    tank_id: int
    x: int
    y: int
    team: int
    rank: int
    fuel: int
    leaderboard_position: int


class MineValueDict(TypedDict):
    """The believed value of a mine fact (metadata stripped).

    Attributes:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        mine_type: Type of mine. 0 if unknown.
        tank_id: ID of placing tank. -1 if unknown.
        team: Team that owns the mine.
    """

    x: int
    y: int
    mine_type: int
    tank_id: int
    team: int


class TerrainValueDict(TypedDict):
    """The believed value of a terrain-tile fact (metadata stripped).

    Attributes:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        terrain_type: Terrain/structure type (0-7).
    """

    x: int
    y: int
    terrain_type: int


class ViewportValueDict(TypedDict):
    """The believed value of the viewport fact (metadata stripped).

    Attributes:
        left: Left edge X coordinate.
        top: Top edge Y coordinate.
        width: Viewport width in tiles.
        height: Viewport height in tiles.
    """

    left: int
    top: int
    width: int
    height: int


def self_fact(state: SelfStateDict) -> Fact[SelfValueDict]:
    """Project the self state into a Fact.

    Args:
        state: Self state carrying flat fact metadata.

    Returns:
        Fact whose value is the believed self-tank status.
    """
    return make_fact(
        value=SelfValueDict(
            tank_id=state["tank_id"],
            x=state["x"],
            y=state["y"],
            team=state["team"],
            rank=state["rank"],
            fuel=state["fuel"],
            leaderboard_position=state["leaderboard_position"],
        ),
        source=state["provenance"]["origin"],
        observed_ms=state["observed_ms"],
        confidence=state["confidence"],
        provenance=state["provenance"],
    )


def mine_fact(state: MineStateDict) -> Fact[MineValueDict]:
    """Project a mine state into a Fact.

    Args:
        state: Mine state carrying flat fact metadata.

    Returns:
        Fact whose value is the believed mine placement.
    """
    return make_fact(
        value=MineValueDict(
            x=state["x"],
            y=state["y"],
            mine_type=state["mine_type"],
            tank_id=state["tank_id"],
            team=state["team"],
        ),
        source=state["provenance"]["origin"],
        observed_ms=state["timestamp_ms"],
        confidence=state["confidence"],
        provenance=state["provenance"],
    )


def terrain_tile_fact(tile: TerrainTileDict) -> Fact[TerrainValueDict]:
    """Project a terrain tile into a Fact.

    Args:
        tile: Terrain tile carrying flat fact metadata.

    Returns:
        Fact whose value is the believed tile terrain.
    """
    return make_fact(
        value=TerrainValueDict(
            x=tile["x"],
            y=tile["y"],
            terrain_type=tile["terrain_type"],
        ),
        source=tile["provenance"]["origin"],
        observed_ms=tile["observed_ms"],
        confidence=tile["confidence"],
        provenance=tile["provenance"],
    )


def viewport_fact(state: ViewportStateDict) -> Fact[ViewportValueDict]:
    """Project the viewport state into a Fact.

    Args:
        state: Viewport state carrying flat fact metadata.

    Returns:
        Fact whose value is the believed viewport bounds.
    """
    return make_fact(
        value=ViewportValueDict(
            left=state["left"],
            top=state["top"],
            width=state["width"],
            height=state["height"],
        ),
        source=state["provenance"]["origin"],
        observed_ms=state["observed_ms"],
        confidence=state["confidence"],
        provenance=state["provenance"],
    )


__all__ = [
    "MineValueDict",
    "SelfValueDict",
    "TerrainValueDict",
    "ViewportValueDict",
    "mine_fact",
    "self_fact",
    "terrain_tile_fact",
    "viewport_fact",
]
