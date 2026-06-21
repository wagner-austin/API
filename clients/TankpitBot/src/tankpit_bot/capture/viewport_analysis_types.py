"""TypedDicts and codecs for viewport analysis results.

Extracted from viewport_analysis.py to keep types/codecs separate from
the analysis logic.

The PositionViewportEvidence path (and related fields) was deleted
2026-06-20 along with the container PositionUpdate decoder: 13-byte
0x2E bodies are now all 0x3D MovementResponse via the protocol tunnel,
so there is nothing left to compare against viewport offsets.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    optional_int,
    require_int,
    require_list,
)

from tankpit_bot.protocol import BinaryMessage


class ViewportInferenceDict(TypedDict):
    """Viewport origin read directly from a viewport update."""

    message_index: int
    timestamp_ms: int
    viewport_left: int
    viewport_top: int


class ViewportShiftDict(TypedDict):
    """Viewport origin change detected from successive inferred origins."""

    message_index: int
    timestamp_ms: int
    old_left: int
    old_top: int
    new_left: int
    new_top: int


class ThirteenByteShapeDict(TypedDict):
    """Count of a decoded 13-byte ``0x2E`` body shape."""

    first_byte: int
    second_byte: int
    count: int


class ViewportAnalysisDict(TypedDict):
    """Evidence summary derived from a capture session."""

    self_tank_id: int | None
    viewport_inferences: list[ViewportInferenceDict]
    viewport_shifts: list[ViewportShiftDict]
    movement_response_count: int
    viewport_update_count: int
    thirteen_byte_0x2e_count: int
    thirteen_byte_shapes: list[ThirteenByteShapeDict]


class DecodedBinaryRecordDict(TypedDict):
    """Decoded received binary message record."""

    message_index: int
    timestamp_ms: int
    decoded: BinaryMessage


class ViewportAnalysisStateDict(TypedDict):
    """Mutable analysis state carried across decoded messages."""

    self_tank_id: int | None
    current_viewport_left: int | None
    current_viewport_top: int | None


def encode_viewport_inference(evidence: ViewportInferenceDict) -> JSONObject:
    """Encode viewport inference to JSON.

    Args:
        evidence: Viewport inference to encode.

    Returns:
        JSON object representation.
    """
    return {
        "message_index": evidence["message_index"],
        "timestamp_ms": evidence["timestamp_ms"],
        "viewport_left": evidence["viewport_left"],
        "viewport_top": evidence["viewport_top"],
    }


def decode_viewport_inference(data: JSONObject) -> ViewportInferenceDict:
    """Decode viewport inference from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated viewport inference.
    """
    return ViewportInferenceDict(
        message_index=require_int(data, "message_index"),
        timestamp_ms=require_int(data, "timestamp_ms"),
        viewport_left=require_int(data, "viewport_left"),
        viewport_top=require_int(data, "viewport_top"),
    )


def encode_viewport_shift(shift: ViewportShiftDict) -> JSONObject:
    """Encode viewport shift evidence to JSON.

    Args:
        shift: Viewport shift to encode.

    Returns:
        JSON object representation.
    """
    return {
        "message_index": shift["message_index"],
        "timestamp_ms": shift["timestamp_ms"],
        "old_left": shift["old_left"],
        "old_top": shift["old_top"],
        "new_left": shift["new_left"],
        "new_top": shift["new_top"],
    }


def encode_thirteen_byte_shape(shape: ThirteenByteShapeDict) -> JSONObject:
    """Encode a 13-byte ``0x2E`` shape count to JSON.

    Args:
        shape: Shape count entry to encode.

    Returns:
        JSON object representation.
    """
    return {
        "first_byte": shape["first_byte"],
        "second_byte": shape["second_byte"],
        "count": shape["count"],
    }


def decode_viewport_shift(data: JSONObject) -> ViewportShiftDict:
    """Decode viewport shift evidence from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated viewport shift.
    """
    return ViewportShiftDict(
        message_index=require_int(data, "message_index"),
        timestamp_ms=require_int(data, "timestamp_ms"),
        old_left=require_int(data, "old_left"),
        old_top=require_int(data, "old_top"),
        new_left=require_int(data, "new_left"),
        new_top=require_int(data, "new_top"),
    )


def decode_thirteen_byte_shape(data: JSONObject) -> ThirteenByteShapeDict:
    """Decode a 13-byte ``0x2E`` shape count from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated shape count entry.
    """
    return ThirteenByteShapeDict(
        first_byte=require_int(data, "first_byte"),
        second_byte=require_int(data, "second_byte"),
        count=require_int(data, "count"),
    )


def encode_viewport_analysis(result: ViewportAnalysisDict) -> JSONObject:
    """Encode viewport analysis result to JSON.

    Args:
        result: Analysis result to encode.

    Returns:
        JSON object representation.
    """
    viewport_inferences: list[JSONValue] = [
        encode_viewport_inference(entry) for entry in result["viewport_inferences"]
    ]
    viewport_shifts: list[JSONValue] = [
        encode_viewport_shift(entry) for entry in result["viewport_shifts"]
    ]
    thirteen_byte_shapes: list[JSONValue] = [
        encode_thirteen_byte_shape(entry) for entry in result["thirteen_byte_shapes"]
    ]
    return {
        "self_tank_id": result["self_tank_id"],
        "viewport_inferences": viewport_inferences,
        "viewport_shifts": viewport_shifts,
        "movement_response_count": result["movement_response_count"],
        "viewport_update_count": result["viewport_update_count"],
        "thirteen_byte_0x2e_count": result["thirteen_byte_0x2e_count"],
        "thirteen_byte_shapes": thirteen_byte_shapes,
    }


def decode_viewport_analysis(data: JSONObject) -> ViewportAnalysisDict:
    """Decode viewport analysis result from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated viewport analysis result.

    Raises:
        JSONTypeError: If nested entries are invalid.
    """
    raw_inferences = require_list(data, "viewport_inferences")
    viewport_inferences: list[ViewportInferenceDict] = []
    for idx, raw_entry in enumerate(raw_inferences):
        if not isinstance(raw_entry, dict):
            raise JSONTypeError(f"viewport_inferences[{idx}] must be an object")
        viewport_inferences.append(decode_viewport_inference(raw_entry))

    raw_viewport_shifts = require_list(data, "viewport_shifts")
    viewport_shifts: list[ViewportShiftDict] = []
    for idx, raw_entry in enumerate(raw_viewport_shifts):
        if not isinstance(raw_entry, dict):
            raise JSONTypeError(f"viewport_shifts[{idx}] must be an object")
        viewport_shifts.append(decode_viewport_shift(raw_entry))

    raw_thirteen_byte_shapes = require_list(data, "thirteen_byte_shapes")
    thirteen_byte_shapes: list[ThirteenByteShapeDict] = []
    for idx, raw_entry in enumerate(raw_thirteen_byte_shapes):
        if not isinstance(raw_entry, dict):
            raise JSONTypeError(f"thirteen_byte_shapes[{idx}] must be an object")
        thirteen_byte_shapes.append(decode_thirteen_byte_shape(raw_entry))

    return ViewportAnalysisDict(
        self_tank_id=optional_int(data, "self_tank_id"),
        viewport_inferences=viewport_inferences,
        viewport_shifts=viewport_shifts,
        movement_response_count=require_int(data, "movement_response_count"),
        viewport_update_count=require_int(data, "viewport_update_count"),
        thirteen_byte_0x2e_count=require_int(data, "thirteen_byte_0x2e_count"),
        thirteen_byte_shapes=thirteen_byte_shapes,
    )


__all__ = [
    "DecodedBinaryRecordDict",
    "ThirteenByteShapeDict",
    "ViewportAnalysisDict",
    "ViewportAnalysisStateDict",
    "ViewportInferenceDict",
    "ViewportShiftDict",
    "decode_thirteen_byte_shape",
    "decode_viewport_analysis",
    "decode_viewport_inference",
    "decode_viewport_shift",
    "encode_thirteen_byte_shape",
    "encode_viewport_analysis",
    "encode_viewport_inference",
    "encode_viewport_shift",
]
