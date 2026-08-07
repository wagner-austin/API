"""World-state TypedDicts, constants, factories, and serialization.

This package replaced the previous monolithic ``state/types.py``. Every
public name remains importable from ``tankpit_bot.state.types`` exactly
as before; the submodules group code by entity domain:

* :mod:`tankpit_bot.types.constants` -- enumerations and shared
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
from tankpit_bot.state.types.self_account import (
    SelfAccountDict,
    decode_self_account,
    encode_self_account,
    make_empty_self_account,
)
from tankpit_bot.state.types.self_state import (
    SelfStateDict,
    decode_self_state,
    encode_self_state,
    make_self_state,
)
from tankpit_bot.state.types.tank import (
    VIEWPORT_PRESENCE_TTL_MS,
    TankStateDict,
    decode_tank_state,
    encode_tank_state,
    has_known_position,
    has_real_coordinates,
    make_tank_state,
)
from tankpit_bot.state.types.tank_observation import (
    TankObservation,
    decode_tank_observation,
    encode_tank_observation,
    make_tank_observation,
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
    make_viewport_state,
)
from tankpit_bot.state.types.world import (
    WorldStateDict,
    decode_world_state,
    encode_world_state,
    make_empty_world_state,
)

__all__ = [
    "VIEWPORT_PRESENCE_TTL_MS",
    "ContainerStateDict",
    "MineStateDict",
    "SelfAccountDict",
    "SelfStateDict",
    "TankObservation",
    "TankStateDict",
    "TerrainTileDict",
    "ViewportStateDict",
    "WorldStateDict",
    "coord_key",
    "decode_container_state",
    "decode_mine_state",
    "decode_self_account",
    "decode_self_state",
    "decode_tank_observation",
    "decode_tank_state",
    "decode_terrain_tile",
    "decode_viewport_state",
    "decode_world_state",
    "encode_container_state",
    "encode_mine_state",
    "encode_self_account",
    "encode_self_state",
    "encode_tank_observation",
    "encode_tank_state",
    "encode_terrain_tile",
    "encode_viewport_state",
    "encode_world_state",
    "has_known_position",
    "has_real_coordinates",
    "make_container_state",
    "make_empty_self_account",
    "make_empty_world_state",
    "make_mine_state",
    "make_self_state",
    "make_tank_observation",
    "make_tank_state",
    "make_terrain_tile",
    "make_viewport_state",
    "parse_coord_key",
    "viewport_scan_key",
]
