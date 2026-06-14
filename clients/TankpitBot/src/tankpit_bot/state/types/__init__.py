"""World-state TypedDicts, constants, factories, and serialization.

This package replaced the previous monolithic ``state/types.py``. Every
public name remains importable from ``tankpit_bot.state.types`` exactly
as before; the submodules group code by entity domain:

* :mod:`tankpit_bot.state.types.constants` -- enumerations and shared
  literal validators (entity source, container refresh kind).
* :mod:`tankpit_bot.state.types.coord` -- string-key helpers for the
  coordinate-indexed collections.
* :mod:`tankpit_bot.state.types.tank` -- :class:`TankStateDict`.
* :mod:`tankpit_bot.state.types.container` -- :class:`ContainerStateDict`.
* :mod:`tankpit_bot.state.types.mine` -- :class:`MineStateDict`.
* :mod:`tankpit_bot.state.types.terrain` -- :class:`TerrainTileDict`.
* :mod:`tankpit_bot.state.types.viewport` -- :class:`ViewportStateDict`.
* :mod:`tankpit_bot.state.types.self_state` -- :class:`SelfStateDict`.
* :mod:`tankpit_bot.state.types.world` -- :class:`WorldStateDict`.

The previously-duplicated ``_decode_dict_field_*`` per-entity helpers
collapsed into the single :func:`tankpit_bot.state.types._helpers.decode_entity_dict`.
"""

from __future__ import annotations

from tankpit_bot.state.types.constants import (
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
    ASCII_WATER,
    CONTAINER_REFRESH_KINDS,
    DAMAGE_CRITICAL,
    DAMAGE_FULL,
    DAMAGE_LIGHT,
    DAMAGE_MEDIUM,
    ENTITY_SOURCES,
    TEAM_BLUE,
    TEAM_ORANGE,
    TEAM_PURPLE,
    TEAM_RED,
    TERRAIN_FERRY,
    TERRAIN_FERRY_ROCK,
    TERRAIN_GROUND,
    TERRAIN_ROCK_A,
    TERRAIN_ROCK_AB,
    TERRAIN_ROCK_B,
    ContainerRefreshKind,
    EntitySource,
    decode_container_refresh_kind,
    encode_container_refresh_kind,
)
from tankpit_bot.state.types.container import (
    ContainerStateDict,
    decode_container_state,
    encode_container_state,
    make_container_state,
)
from tankpit_bot.state.types.coord import coord_key, parse_coord_key, viewport_scan_key
from tankpit_bot.state.types.mine import (
    MineStateDict,
    decode_mine_state,
    encode_mine_state,
    make_mine_state,
)
from tankpit_bot.state.types.self_state import (
    SelfStateDict,
    decode_self_state,
    encode_self_state,
    make_self_state,
)
from tankpit_bot.state.types.tank import (
    TankStateDict,
    decode_tank_state,
    encode_tank_state,
    make_tank_state,
)
from tankpit_bot.state.types.terrain import (
    TerrainTileDict,
    decode_terrain_tile,
    encode_terrain_tile,
    make_terrain_tile,
)
from tankpit_bot.state.types.viewport import (
    ViewportStateDict,
    decode_viewport_state,
    encode_viewport_state,
)
from tankpit_bot.state.types.world import (
    WorldStateDict,
    decode_world_state,
    encode_world_state,
    make_empty_world_state,
)

__all__ = [
    "ASCII_ALLY",
    "ASCII_ENEMY",
    "ASCII_EQUIPMENT",
    "ASCII_FERRY",
    "ASCII_FUEL",
    "ASCII_GROUND",
    "ASCII_MINE",
    "ASCII_ROCK",
    "ASCII_SELF",
    "ASCII_UNKNOWN",
    "ASCII_WATER",
    "CONTAINER_REFRESH_KINDS",
    "DAMAGE_CRITICAL",
    "DAMAGE_FULL",
    "DAMAGE_LIGHT",
    "DAMAGE_MEDIUM",
    "ENTITY_SOURCES",
    "TEAM_BLUE",
    "TEAM_ORANGE",
    "TEAM_PURPLE",
    "TEAM_RED",
    "TERRAIN_FERRY",
    "TERRAIN_FERRY_ROCK",
    "TERRAIN_GROUND",
    "TERRAIN_ROCK_A",
    "TERRAIN_ROCK_AB",
    "TERRAIN_ROCK_B",
    "ContainerRefreshKind",
    "ContainerStateDict",
    "EntitySource",
    "MineStateDict",
    "SelfStateDict",
    "TankStateDict",
    "TerrainTileDict",
    "ViewportStateDict",
    "WorldStateDict",
    "coord_key",
    "decode_container_refresh_kind",
    "decode_container_state",
    "decode_mine_state",
    "decode_self_state",
    "decode_tank_state",
    "decode_terrain_tile",
    "decode_viewport_state",
    "decode_world_state",
    "encode_container_refresh_kind",
    "encode_container_state",
    "encode_mine_state",
    "encode_self_state",
    "encode_tank_state",
    "encode_terrain_tile",
    "encode_viewport_state",
    "encode_world_state",
    "make_container_state",
    "make_empty_world_state",
    "make_mine_state",
    "make_self_state",
    "make_tank_state",
    "make_terrain_tile",
    "parse_coord_key",
    "viewport_scan_key",
]
