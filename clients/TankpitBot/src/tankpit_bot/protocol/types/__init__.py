"""Protocol message TypedDicts, one module per payload family.

The 47 payload TypedDicts were a single 959-line module; they are now
nine submodules whose membership mirrors
:mod:`tankpit_bot.protocol.decoders` -- the decoder that produces a
payload owns its definition, so a new message type has exactly one
obvious home.

This module owns what genuinely spans the families: the
:data:`TextMessage` / :data:`BinaryMessage` / :data:`DecodedMessage`
unions, which no single family can define.
"""

from __future__ import annotations

from tankpit_bot.container.types import ContainerMessage
from tankpit_bot.protocol.types.combat import (
    DeactivationDict,
    ShootEventDict,
)
from tankpit_bot.protocol.types.map_data import (
    MapDataDict,
    MapTankEntry,
)
from tankpit_bot.protocol.types.movement import (
    MovementDict,
    MovementResponseDict,
)
from tankpit_bot.protocol.types.radar import (
    EnemyDetectionDict,
    RadarContainerDict,
    RadarMineClearDict,
    RadarMineDict,
    RadarResultDict,
    RadarScanResultDict,
)
from tankpit_bot.protocol.types.resources import (
    EquipmentGainDict,
    EquipmentToggleDict,
    FuelDepositDict,
    FuelGainDict,
    InventoryDict,
)
from tankpit_bot.protocol.types.session_events import (
    ActionDoneDict,
    ActiveForcesDict,
    ActivePlayerEntry,
    ActivePlayersDict,
    BuildPickupDict,
    ChatMessageDict,
    ConnectionLostDict,
    DecorationDict,
    PingResponseDict,
    PromotionDict,
    StatisticsDict,
    Top10Dict,
    Top10EntryDict,
)
from tankpit_bot.protocol.types.tank import (
    TankEntryDict,
    TankExitDict,
    TankInfoDict,
    TankRemoveDict,
    TankStatusDict,
    TankStatusSyncDict,
)
from tankpit_bot.protocol.types.text import (
    AutoscrollAckDict,
    ChatAckDict,
    JoinConfirmDict,
    WorldInfoDict,
)
from tankpit_bot.protocol.types.world import (
    CacheUpdateDict,
    OverlayUpdateDict,
    SupervisorDict,
    SupervisorTextDict,
    SyncDict,
    TerrainUpdateDict,
    ViewportEntityDict,
    ViewportUpdateDict,
)

TextMessage = JoinConfirmDict | WorldInfoDict

BinaryMessage = (
    ShootEventDict
    | DeactivationDict
    | FuelGainDict
    | FuelDepositDict
    | RadarResultDict
    | EnemyDetectionDict
    | InventoryDict
    | EquipmentGainDict
    | EquipmentToggleDict
    | RadarScanResultDict
    | MovementDict
    | TankInfoDict
    | MovementResponseDict
    | SyncDict
    | CacheUpdateDict
    | ChatAckDict
    | AutoscrollAckDict
    | OverlayUpdateDict
    | TankEntryDict
    | TankExitDict
    | TankRemoveDict
    | PromotionDict
    | DecorationDict
    | BuildPickupDict
    | MapDataDict
    | ActionDoneDict
    | ChatMessageDict
    | StatisticsDict
    | ActiveForcesDict
    | ActivePlayersDict
    | Top10Dict
    | PingResponseDict
    | ConnectionLostDict
    | TankStatusSyncDict
    | TankStatusDict
    | SupervisorDict
    | SupervisorTextDict
    | TerrainUpdateDict
    | ViewportUpdateDict
    | ContainerMessage
)

DecodedMessage = TextMessage | BinaryMessage

__all__ = [
    "ActionDoneDict",
    "ActiveForcesDict",
    "ActivePlayerEntry",
    "ActivePlayersDict",
    "AutoscrollAckDict",
    "BinaryMessage",
    "BuildPickupDict",
    "CacheUpdateDict",
    "ChatAckDict",
    "ChatMessageDict",
    "ConnectionLostDict",
    "DeactivationDict",
    "DecodedMessage",
    "DecorationDict",
    "EnemyDetectionDict",
    "EquipmentGainDict",
    "EquipmentToggleDict",
    "FuelDepositDict",
    "FuelGainDict",
    "InventoryDict",
    "JoinConfirmDict",
    "MapDataDict",
    "MapTankEntry",
    "MovementDict",
    "MovementResponseDict",
    "OverlayUpdateDict",
    "PingResponseDict",
    "PromotionDict",
    "RadarContainerDict",
    "RadarMineClearDict",
    "RadarMineDict",
    "RadarResultDict",
    "RadarScanResultDict",
    "ShootEventDict",
    "StatisticsDict",
    "SupervisorDict",
    "SupervisorTextDict",
    "SyncDict",
    "TankEntryDict",
    "TankExitDict",
    "TankInfoDict",
    "TankRemoveDict",
    "TankStatusDict",
    "TankStatusSyncDict",
    "TerrainUpdateDict",
    "TextMessage",
    "Top10Dict",
    "Top10EntryDict",
    "ViewportEntityDict",
    "ViewportUpdateDict",
    "WorldInfoDict",
]
