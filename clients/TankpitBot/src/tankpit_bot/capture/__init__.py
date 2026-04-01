"""Capture module for WebSocket message processing.

This module provides utilities for XOR decoding, message identification,
statistics, session summary building, and message trackers.
"""

from __future__ import annotations

from tankpit_bot.capture.protocol_census import (
    analyze_protocol_census,
    decode_protocol_census,
    decode_protocol_count,
    decode_protocol_sample,
    encode_protocol_census,
    encode_protocol_count,
    encode_protocol_sample,
    format_protocol_census,
)
from tankpit_bot.capture.shot_viewport_correlation import (
    analyze_shot_viewport_correlation,
    decode_shot_viewport_correlation,
    decode_shot_viewport_correlation_dump,
    encode_shot_viewport_correlation,
    encode_shot_viewport_correlation_dump,
    format_shot_viewport_correlation,
)
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
from tankpit_bot.capture.viewport_analysis import (
    analyze_capture_session,
    decode_position_viewport_evidence,
    decode_viewport_analysis,
    decode_viewport_inference,
    decode_viewport_shift,
    encode_position_viewport_evidence,
    encode_viewport_analysis,
    encode_viewport_inference,
    encode_viewport_shift,
    format_viewport_analysis,
)
from tankpit_bot.capture.viewport_entities import (
    analyze_viewport_entities,
    decode_viewport_entity_dump,
    decode_viewport_entity_row,
    decode_viewport_entity_update,
    encode_viewport_entity_dump,
    encode_viewport_entity_row,
    encode_viewport_entity_update,
    format_viewport_entity_dump,
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
    "analyze_capture_session",
    "analyze_protocol_census",
    "analyze_shot_viewport_correlation",
    "analyze_viewport_entities",
    "build_message_stats",
    "build_session_summary",
    "build_xor_table",
    "decode_base64_safe",
    "decode_position_viewport_evidence",
    "decode_protocol_census",
    "decode_protocol_count",
    "decode_protocol_sample",
    "decode_shot_viewport_correlation",
    "decode_shot_viewport_correlation_dump",
    "decode_viewport_analysis",
    "decode_viewport_entity_dump",
    "decode_viewport_entity_row",
    "decode_viewport_entity_update",
    "decode_viewport_inference",
    "decode_viewport_shift",
    "empty_message_stats",
    "encode_position_viewport_evidence",
    "encode_protocol_census",
    "encode_protocol_count",
    "encode_protocol_sample",
    "encode_shot_viewport_correlation",
    "encode_shot_viewport_correlation_dump",
    "encode_viewport_analysis",
    "encode_viewport_entity_dump",
    "encode_viewport_entity_row",
    "encode_viewport_entity_update",
    "encode_viewport_inference",
    "encode_viewport_shift",
    "extract_message_signature",
    "format_protocol_census",
    "format_shot_viewport_correlation",
    "format_sig_key",
    "format_viewport_analysis",
    "format_viewport_entity_dump",
    "identify_message",
    "load_xor_static_key",
    "xor_decode_body",
]
