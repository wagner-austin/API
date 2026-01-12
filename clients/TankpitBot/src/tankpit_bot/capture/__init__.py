"""Capture module for WebSocket message processing.

This module provides utilities for XOR decoding, message identification,
statistics, session summary building, and message trackers.
"""

from __future__ import annotations

from tankpit_bot.capture.signature import (
    extract_message_signature,
    format_sig_key,
    identify_message,
)
from tankpit_bot.capture.stats import (
    build_message_stats,
    empty_message_stats,
)
from tankpit_bot.capture.summary import (
    build_session_summary,
)
from tankpit_bot.capture.trackers import (
    ContainerTracker,
    DeactivationTracker,
    EquipmentGainTracker,
    EquipmentToggleTracker,
    FuelDepositTracker,
    ItemPickupTracker,
    MineTracker,
    PositionTracker,
    RadarAckTracker,
    RadarTracker,
    TankExitTracker,
    TankTracker,
)
from tankpit_bot.capture.xor import (
    build_xor_table,
    decode_base64_safe,
    load_xor_static_key,
    xor_decode_body,
)

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
    "build_message_stats",
    "build_session_summary",
    "build_xor_table",
    "decode_base64_safe",
    "empty_message_stats",
    "extract_message_signature",
    "format_sig_key",
    "identify_message",
    "load_xor_static_key",
    "xor_decode_body",
]
