"""Message trackers for WebSocket capture decoding.

This module provides specialized tracker classes for decoding different
types of TankPit protocol messages during live capture sessions.
"""

from __future__ import annotations

from tankpit_bot.capture.trackers.combat import DeactivationTracker
from tankpit_bot.capture.trackers.container import ContainerTracker
from tankpit_bot.capture.trackers.equipment import (
    EquipmentGainTracker,
    EquipmentToggleTracker,
)
from tankpit_bot.capture.trackers.fuel import FuelDepositTracker
from tankpit_bot.capture.trackers.items import ItemPickupTracker
from tankpit_bot.capture.trackers.mine import MineTracker
from tankpit_bot.capture.trackers.position import PositionTracker
from tankpit_bot.capture.trackers.radar import RadarAckTracker, RadarTracker
from tankpit_bot.capture.trackers.tank import TankExitTracker, TankTracker

__all__ = [
    "ContainerTracker",
    "DeactivationTracker",
    "EquipmentGainTracker",
    "EquipmentToggleTracker",
    "FuelDepositTracker",
    "ItemPickupTracker",
    "MineTracker",
    "PositionTracker",
    "RadarAckTracker",
    "RadarTracker",
    "TankExitTracker",
    "TankTracker",
]
