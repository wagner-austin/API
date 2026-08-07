"""World state module for game visualization and bot decision-making.

This module provides TypedDicts and functions for aggregating decoded protocol
messages into a coherent game world state suitable for ASCII rendering and
bot logic.

Submodules:
- types: TypedDicts, constants, factory functions, encode/decode
- mutations: World state update functions
- renderer: ASCII viewport rendering
"""

from tankpit_bot.state.container_mutations import (
    add_mine,
    add_mine_from_radar,
    apply_tile_cache_update,
    apply_tile_overlay_update,
    increment_container_failed_pickups,
    pickup_container,
    remove_container,
    remove_mine,
    update_container_from_radar,
)
from tankpit_bot.state.renderer import (
    render_world_ascii,
    terrain_to_ascii,
)
from tankpit_bot.state.scan_coverage import record_scanned_tiles
from tankpit_bot.state.self_mutations import (
    set_self_fuel,
    set_self_rank,
    update_self_from_movement_response,
    update_self_position,
    update_self_rank,
)
from tankpit_bot.state.tank_mutations import (
    apply_tank_observation,
    deactivate_tank,
    remove_tank,
    set_tank_last_aim,
)
from tankpit_bot.state.terrain_mutations import update_terrain_from_viewport
from tankpit_bot.state.types import (
    ContainerStateDict,
    MineStateDict,
    SelfStateDict,
    TankStateDict,
    TerrainTileDict,
    ViewportStateDict,
    WorldStateDict,
    coord_key,
    decode_container_state,
    decode_mine_state,
    decode_self_state,
    decode_tank_state,
    decode_terrain_tile,
    decode_viewport_state,
    decode_world_state,
    encode_container_state,
    encode_mine_state,
    encode_self_state,
    encode_tank_state,
    encode_terrain_tile,
    encode_viewport_state,
    encode_world_state,
    make_container_state,
    make_empty_world_state,
    make_mine_state,
    make_self_state,
    make_tank_state,
    make_terrain_tile,
    make_viewport_state,
    parse_coord_key,
    viewport_scan_key,
)

__all__ = [
    "ContainerStateDict",
    "MineStateDict",
    "SelfStateDict",
    "TankStateDict",
    "TerrainTileDict",
    "ViewportStateDict",
    "WorldStateDict",
    "add_mine",
    "add_mine_from_radar",
    "apply_tank_observation",
    "apply_tile_cache_update",
    "apply_tile_overlay_update",
    "coord_key",
    "deactivate_tank",
    "decode_container_state",
    "decode_mine_state",
    "decode_self_state",
    "decode_tank_state",
    "decode_terrain_tile",
    "decode_viewport_state",
    "decode_world_state",
    "encode_container_state",
    "encode_mine_state",
    "encode_self_state",
    "encode_tank_state",
    "encode_terrain_tile",
    "encode_viewport_state",
    "encode_world_state",
    "increment_container_failed_pickups",
    "make_container_state",
    "make_empty_world_state",
    "make_mine_state",
    "make_self_state",
    "make_tank_state",
    "make_terrain_tile",
    "make_viewport_state",
    "parse_coord_key",
    "pickup_container",
    "record_scanned_tiles",
    "remove_container",
    "remove_mine",
    "remove_tank",
    "render_world_ascii",
    "set_self_fuel",
    "set_self_rank",
    "set_tank_last_aim",
    "terrain_to_ascii",
    "update_container_from_radar",
    "update_self_from_movement_response",
    "update_self_position",
    "update_self_rank",
    "update_terrain_from_viewport",
    "viewport_scan_key",
]
