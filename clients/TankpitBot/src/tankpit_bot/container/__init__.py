"""Container message decoding module.

This module provides complete container message decoding capabilities,
organized into submodules:

- types: Enums, TypedDicts, decode level registry
- helpers: ContainerDecodeError, validation functions
- identification: Message type identification by structure
- decoders: Message decoding functions organized by category
"""

from __future__ import annotations

from tankpit_bot.container.decoders import (
    decode_container_message,
    decode_container_pickup,
    decode_mine_detonation,
    decode_mine_placement,
    decode_teleport_landed,
    decode_unknown_container,
    identify_container_type,
    is_container_pickup_structure,
    is_mine_detonation_structure,
    is_mine_placement_structure,
    is_teleport_landed_structure,
)
from tankpit_bot.container.helpers import (
    ContainerDecodeError,
    extract_uint16_le,
    require_exact_length,
    require_length_range,
    require_min_length,
)
from tankpit_bot.container.types import (
    MESSAGE_TYPE_LEVELS,
    ContainerMessage,
    ContainerMessageType,
    ContainerPickupDict,
    ContainerPickupRecordDict,
    DecodeLevel,
    MineDetonationDict,
    MinePlacementDict,
    TeleportLandedDict,
    UnknownContainerDict,
    get_decode_level,
)

__all__ = [
    "MESSAGE_TYPE_LEVELS",
    "ContainerDecodeError",
    "ContainerMessage",
    "ContainerMessageType",
    "ContainerPickupDict",
    "ContainerPickupRecordDict",
    "DecodeLevel",
    "MineDetonationDict",
    "MinePlacementDict",
    "TeleportLandedDict",
    "UnknownContainerDict",
    "decode_container_message",
    "decode_container_pickup",
    "decode_mine_detonation",
    "decode_mine_placement",
    "decode_teleport_landed",
    "decode_unknown_container",
    "extract_uint16_le",
    "get_decode_level",
    "identify_container_type",
    "is_container_pickup_structure",
    "is_mine_detonation_structure",
    "is_mine_placement_structure",
    "is_teleport_landed_structure",
    "require_exact_length",
    "require_length_range",
    "require_min_length",
]
