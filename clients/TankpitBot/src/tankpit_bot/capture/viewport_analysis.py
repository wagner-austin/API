"""Viewport analysis for captured sessions.

This module derives viewport origin evidence from captured protocol traffic so
viewport semantics can be verified from real packets rather than inferred from
client behavior.
"""

from __future__ import annotations

from collections import Counter
from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    optional_int,
    require_bool,
    require_int,
    require_list,
)

from tankpit_bot.capture.xor import decode_base64_safe, xor_decode_body
from tankpit_bot.protocol import BinaryMessage, try_decode_binary_message
from tankpit_bot.types.session import CaptureSession


class ViewportInferenceDict(TypedDict):
    """Viewport origin read directly from a viewport update.

    Attributes:
        message_index: Index of the triggering message in the capture session.
        timestamp_ms: Timestamp of the triggering message.
        viewport_left: Viewport left coordinate from 0x5A.
        viewport_top: Viewport top coordinate from 0x5A.
    """

    message_index: int
    timestamp_ms: int
    viewport_left: int
    viewport_top: int


class PositionViewportEvidenceDict(TypedDict):
    """Comparison between position_update extra bytes and inferred viewport.

    Attributes:
        message_index: Index of the triggering message in the capture session.
        timestamp_ms: Timestamp of the triggering message.
        tank_id: Self tank identifier.
        x: Absolute self x coordinate from position_update.
        y: Absolute self y coordinate from position_update.
        extra_x: First extra_data byte from position_update.
        extra_y: Second extra_data byte from position_update.
        viewport_left: Inferred viewport left coordinate active at the time.
        viewport_top: Inferred viewport top coordinate active at the time.
        expected_viewport_x: Computed local x based on inferred viewport.
        expected_viewport_y: Computed local y based on inferred viewport.
        matches_x: Whether extra_x matches expected_viewport_x.
        matches_y: Whether extra_y matches expected_viewport_y.
    """

    message_index: int
    timestamp_ms: int
    tank_id: int
    x: int
    y: int
    extra_x: int
    extra_y: int
    viewport_left: int
    viewport_top: int
    expected_viewport_x: int
    expected_viewport_y: int
    matches_x: bool
    matches_y: bool


class ViewportShiftDict(TypedDict):
    """Viewport origin change detected from successive inferred origins.

    Attributes:
        message_index: Index of the message that produced the new origin.
        timestamp_ms: Timestamp of the message that produced the new origin.
        old_left: Previous viewport left coordinate.
        old_top: Previous viewport top coordinate.
        new_left: New viewport left coordinate.
        new_top: New viewport top coordinate.
    """

    message_index: int
    timestamp_ms: int
    old_left: int
    old_top: int
    new_left: int
    new_top: int


class ThirteenByteShapeDict(TypedDict):
    """Count of a decoded 13-byte ``0x2E`` body shape.

    Attributes:
        first_byte: First decoded byte of the 13-byte body.
        second_byte: Second decoded byte of the 13-byte body.
        count: Number of times this shape appeared in the capture.
    """

    first_byte: int
    second_byte: int
    count: int


class ViewportAnalysisDict(TypedDict):
    """Evidence summary derived from a capture session.

    Attributes:
        self_tank_id: Inferred self tank identifier, if known.
        viewport_inferences: Inferred viewport origins.
        position_evidence: Comparable absolute position_update samples.
        viewport_shifts: Viewport origin changes between inferences.
        movement_response_count: Count of decoded self/other movement responses.
        viewport_update_count: Count of decoded viewport updates.
        position_update_count: Count of decoded position_update messages.
        thirteen_byte_0x2e_count: Count of raw 13-byte decoded ``0x2E`` bodies.
        thirteen_byte_shapes: Frequency table for raw 13-byte decoded ``0x2E`` bodies.
        comparable_position_count: Count of comparable position_update samples.
        extra_x_match_count: Count of position_update samples where extra_x matched.
        extra_y_match_count: Count of position_update samples where extra_y matched.
    """

    self_tank_id: int | None
    viewport_inferences: list[ViewportInferenceDict]
    position_evidence: list[PositionViewportEvidenceDict]
    viewport_shifts: list[ViewportShiftDict]
    movement_response_count: int
    viewport_update_count: int
    position_update_count: int
    thirteen_byte_0x2e_count: int
    thirteen_byte_shapes: list[ThirteenByteShapeDict]
    comparable_position_count: int
    extra_x_match_count: int
    extra_y_match_count: int


class _DecodedBinaryRecordDict(TypedDict):
    """Decoded received binary message record."""

    message_index: int
    timestamp_ms: int
    decoded: BinaryMessage


class _ViewportAnalysisStateDict(TypedDict):
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


def encode_position_viewport_evidence(evidence: PositionViewportEvidenceDict) -> JSONObject:
    """Encode position-vs-viewport evidence to JSON.

    Args:
        evidence: Position evidence to encode.

    Returns:
        JSON object representation.
    """
    return {
        "message_index": evidence["message_index"],
        "timestamp_ms": evidence["timestamp_ms"],
        "tank_id": evidence["tank_id"],
        "x": evidence["x"],
        "y": evidence["y"],
        "extra_x": evidence["extra_x"],
        "extra_y": evidence["extra_y"],
        "viewport_left": evidence["viewport_left"],
        "viewport_top": evidence["viewport_top"],
        "expected_viewport_x": evidence["expected_viewport_x"],
        "expected_viewport_y": evidence["expected_viewport_y"],
        "matches_x": evidence["matches_x"],
        "matches_y": evidence["matches_y"],
    }


def decode_position_viewport_evidence(data: JSONObject) -> PositionViewportEvidenceDict:
    """Decode position-vs-viewport evidence from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated position evidence.
    """
    return PositionViewportEvidenceDict(
        message_index=require_int(data, "message_index"),
        timestamp_ms=require_int(data, "timestamp_ms"),
        tank_id=require_int(data, "tank_id"),
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        extra_x=require_int(data, "extra_x"),
        extra_y=require_int(data, "extra_y"),
        viewport_left=require_int(data, "viewport_left"),
        viewport_top=require_int(data, "viewport_top"),
        expected_viewport_x=require_int(data, "expected_viewport_x"),
        expected_viewport_y=require_int(data, "expected_viewport_y"),
        matches_x=require_bool(data, "matches_x"),
        matches_y=require_bool(data, "matches_y"),
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
    position_evidence: list[JSONValue] = [
        encode_position_viewport_evidence(entry) for entry in result["position_evidence"]
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
        "position_evidence": position_evidence,
        "viewport_shifts": viewport_shifts,
        "movement_response_count": result["movement_response_count"],
        "viewport_update_count": result["viewport_update_count"],
        "position_update_count": result["position_update_count"],
        "thirteen_byte_0x2e_count": result["thirteen_byte_0x2e_count"],
        "thirteen_byte_shapes": thirteen_byte_shapes,
        "comparable_position_count": result["comparable_position_count"],
        "extra_x_match_count": result["extra_x_match_count"],
        "extra_y_match_count": result["extra_y_match_count"],
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

    raw_position_evidence = require_list(data, "position_evidence")
    position_evidence: list[PositionViewportEvidenceDict] = []
    for idx, raw_entry in enumerate(raw_position_evidence):
        if not isinstance(raw_entry, dict):
            raise JSONTypeError(f"position_evidence[{idx}] must be an object")
        position_evidence.append(decode_position_viewport_evidence(raw_entry))

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
        position_evidence=position_evidence,
        viewport_shifts=viewport_shifts,
        movement_response_count=require_int(data, "movement_response_count"),
        viewport_update_count=require_int(data, "viewport_update_count"),
        position_update_count=require_int(data, "position_update_count"),
        thirteen_byte_0x2e_count=require_int(data, "thirteen_byte_0x2e_count"),
        thirteen_byte_shapes=thirteen_byte_shapes,
        comparable_position_count=require_int(data, "comparable_position_count"),
        extra_x_match_count=require_int(data, "extra_x_match_count"),
        extra_y_match_count=require_int(data, "extra_y_match_count"),
    )


def _split_frame_messages(frame: bytes) -> list[bytes]:
    """Split a captured WebSocket frame into message bodies.

    Args:
        frame: Raw frame bytes including 2-byte length prefixes.

    Returns:
        List of message bodies without the frame-level length prefixes.
    """
    bodies: list[bytes] = []
    offset = 0
    while offset + 2 <= len(frame):
        msg_len = frame[offset] | (frame[offset + 1] << 8)
        offset += 2
        if msg_len == 0 or offset + msg_len > len(frame):
            return bodies
        bodies.append(frame[offset : offset + msg_len])
        offset += msg_len
    return bodies


def _is_absolute_position(x: int, y: int) -> bool:
    """Return whether a position_update uses absolute coordinates.

    Args:
        x: Position update x coordinate.
        y: Position update y coordinate.

    Returns:
        True when either coordinate lies outside the 18x18 viewport-relative
        range.
    """
    return x >= 18 or y >= 18


def _decode_received_binary_records(
    session: CaptureSession,
    xor_table: bytes,
) -> list[_DecodedBinaryRecordDict]:
    """Decode all received binary messages from a capture session.

    Args:
        session: Capture session to inspect.
        xor_table: XOR table for the session magic.

    Returns:
        Decoded received binary message records.
    """
    records: list[_DecodedBinaryRecordDict] = []
    for message_index, message in enumerate(session["messages"]):
        if message["direction"] != "received":
            continue

        frame = decode_base64_safe(message["payload"])
        if frame is None:
            continue

        for body in _split_frame_messages(frame):
            msg_type = body[0]
            decoded_data = xor_decode_body(body, xor_table, offset=1)
            decoded = try_decode_binary_message(msg_type, decoded_data)
            if decoded is None:
                continue
            records.append(
                _DecodedBinaryRecordDict(
                    message_index=message_index,
                    timestamp_ms=message["timestamp_ms"],
                    decoded=decoded,
                )
            )
    return records


def _collect_thirteen_byte_shapes(
    session: CaptureSession,
    xor_table: bytes,
) -> tuple[int, list[ThirteenByteShapeDict]]:
    """Collect raw decoded 13-byte ``0x2E`` body shapes from a capture.

    Args:
        session: Capture session to inspect.
        xor_table: XOR table for the session magic.

    Returns:
        Tuple of total raw 13-byte ``0x2E`` count and sorted shape counts.
    """
    shape_counts: Counter[tuple[int, int]] = Counter()
    total_count = 0
    for message in session["messages"]:
        if message["direction"] != "received":
            continue
        frame = decode_base64_safe(message["payload"])
        if frame is None:
            continue
        for body in _split_frame_messages(frame):
            if len(body) < 1 or body[0] != 0x2E:
                continue
            decoded_data = xor_decode_body(body, xor_table, offset=1)
            if len(decoded_data) != 13:
                continue
            total_count += 1
            shape_counts[(decoded_data[0], decoded_data[1])] += 1

    shapes: list[ThirteenByteShapeDict] = []
    sorted_shape_counts = list(shape_counts.items())
    sorted_shape_counts.sort(key=_sort_thirteen_byte_shape_count)
    for (first_byte, second_byte), count in sorted_shape_counts:
        shapes.append(
            ThirteenByteShapeDict(
                first_byte=first_byte,
                second_byte=second_byte,
                count=count,
            )
        )
    return total_count, shapes


def _sort_thirteen_byte_shape_count(item: tuple[tuple[int, int], int]) -> tuple[int, int, int]:
    """Return a deterministic sort key for 13-byte shape counts.

    Args:
        item: ``((first_byte, second_byte), count)`` tuple.

    Returns:
        Sort key ordering by descending count, then ascending bytes.
    """
    return (-item[1], item[0][0], item[0][1])


def _handle_movement_response(
    state: _ViewportAnalysisStateDict,
    tank_id: int,
) -> _ViewportAnalysisStateDict:
    """Update analysis state from a MovementResponse.

    Args:
        state: Current analysis state.
        tank_id: Tank identifier from the response.

    Returns:
        Updated analysis state.
    """
    self_tank_id = state["self_tank_id"]
    if self_tank_id is None:
        self_tank_id = tank_id

    return _ViewportAnalysisStateDict(
        self_tank_id=self_tank_id,
        current_viewport_left=state["current_viewport_left"],
        current_viewport_top=state["current_viewport_top"],
    )


def _handle_viewport_update(
    state: _ViewportAnalysisStateDict,
    message_index: int,
    timestamp_ms: int,
    viewport_left: int,
    viewport_top: int,
    viewport_inferences: list[ViewportInferenceDict],
    viewport_shifts: list[ViewportShiftDict],
) -> _ViewportAnalysisStateDict:
    """Update analysis state from a ViewportUpdate.

    Args:
        state: Current analysis state.
        message_index: Capture message index.
        timestamp_ms: Capture timestamp.
        viewport_left: Viewport left coordinate from 0x5A.
        viewport_top: Viewport top coordinate from 0x5A.
        viewport_inferences: Output list for viewport origin rows.
        viewport_shifts: Output list for detected viewport shifts.

    Returns:
        Updated analysis state.
    """
    current_viewport_left = state["current_viewport_left"]
    current_viewport_top = state["current_viewport_top"]

    viewport_inferences.append(
        ViewportInferenceDict(
            message_index=message_index,
            timestamp_ms=timestamp_ms,
            viewport_left=viewport_left,
            viewport_top=viewport_top,
        )
    )
    if (
        current_viewport_left is not None
        and current_viewport_top is not None
        and (current_viewport_left != viewport_left or current_viewport_top != viewport_top)
    ):
        viewport_shifts.append(
            ViewportShiftDict(
                message_index=message_index,
                timestamp_ms=timestamp_ms,
                old_left=current_viewport_left,
                old_top=current_viewport_top,
                new_left=viewport_left,
                new_top=viewport_top,
            )
        )

    return _ViewportAnalysisStateDict(
        self_tank_id=state["self_tank_id"],
        current_viewport_left=viewport_left,
        current_viewport_top=viewport_top,
    )


def _handle_position_update(
    state: _ViewportAnalysisStateDict,
    message_index: int,
    timestamp_ms: int,
    flags: int,
    tank_id: int,
    x: int,
    y: int,
    extra_data: bytes,
    position_evidence: list[PositionViewportEvidenceDict],
) -> _ViewportAnalysisStateDict:
    """Update analysis state from a self absolute position_update.

    Args:
        state: Current analysis state.
        message_index: Capture message index.
        timestamp_ms: Capture timestamp.
        flags: Position update flags.
        tank_id: Tank identifier from the update.
        x: Absolute self x coordinate.
        y: Absolute self y coordinate.
        extra_data: Raw extra_data bytes.
        position_evidence: Output list for comparable evidence rows.

    Returns:
        Updated analysis state.
    """
    if (flags & 0x02) == 0:
        return state

    self_tank_id = state["self_tank_id"]
    if self_tank_id is None:
        self_tank_id = tank_id
    if tank_id != self_tank_id:
        return state
    if not _is_absolute_position(x, y):
        return _ViewportAnalysisStateDict(
            self_tank_id=self_tank_id,
            current_viewport_left=state["current_viewport_left"],
            current_viewport_top=state["current_viewport_top"],
        )
    if len(extra_data) < 2:
        return _ViewportAnalysisStateDict(
            self_tank_id=self_tank_id,
            current_viewport_left=state["current_viewport_left"],
            current_viewport_top=state["current_viewport_top"],
        )

    current_viewport_left = state["current_viewport_left"]
    current_viewport_top = state["current_viewport_top"]
    if current_viewport_left is None or current_viewport_top is None:
        return _ViewportAnalysisStateDict(
            self_tank_id=self_tank_id,
            current_viewport_left=current_viewport_left,
            current_viewport_top=current_viewport_top,
        )

    expected_viewport_x = x - current_viewport_left
    expected_viewport_y = y - current_viewport_top
    position_evidence.append(
        PositionViewportEvidenceDict(
            message_index=message_index,
            timestamp_ms=timestamp_ms,
            tank_id=tank_id,
            x=x,
            y=y,
            extra_x=extra_data[0],
            extra_y=extra_data[1],
            viewport_left=current_viewport_left,
            viewport_top=current_viewport_top,
            expected_viewport_x=expected_viewport_x,
            expected_viewport_y=expected_viewport_y,
            matches_x=extra_data[0] == expected_viewport_x,
            matches_y=extra_data[1] == expected_viewport_y,
        )
    )
    return _ViewportAnalysisStateDict(
        self_tank_id=self_tank_id,
        current_viewport_left=current_viewport_left,
        current_viewport_top=current_viewport_top,
    )


def _count_matches(position_evidence: list[PositionViewportEvidenceDict]) -> tuple[int, int]:
    """Count x and y matches in comparable position evidence.

    Args:
        position_evidence: Comparable position evidence rows.

    Returns:
        Tuple of (extra_x_match_count, extra_y_match_count).
    """
    extra_x_match_count = 0
    extra_y_match_count = 0
    for evidence in position_evidence:
        if evidence["matches_x"]:
            extra_x_match_count += 1
        if evidence["matches_y"]:
            extra_y_match_count += 1
    return extra_x_match_count, extra_y_match_count


def analyze_capture_session(
    session: CaptureSession,
    xor_table: bytes,
) -> ViewportAnalysisDict:
    """Analyze viewport semantics from a decoded capture session.

    Args:
        session: Capture session to analyze.
        xor_table: XOR table for the session magic.

    Returns:
        Structured viewport analysis evidence.

    """
    state = _ViewportAnalysisStateDict(
        self_tank_id=None,
        current_viewport_left=None,
        current_viewport_top=None,
    )
    viewport_inferences: list[ViewportInferenceDict] = []
    position_evidence: list[PositionViewportEvidenceDict] = []
    viewport_shifts: list[ViewportShiftDict] = []
    movement_response_count = 0
    viewport_update_count = 0
    position_update_count = 0

    for record in _decode_received_binary_records(session, xor_table):
        match record["decoded"]:
            case {
                "msg_type": 0x3D,
                "tank_id": int(tank_id),
            }:
                movement_response_count += 1
                state = _handle_movement_response(state, tank_id)
            case {
                "msg_type": 0x5A,
                "viewport_left": int(viewport_left),
                "viewport_top": int(viewport_top),
            }:
                viewport_update_count += 1
                state = _handle_viewport_update(
                    state,
                    record["message_index"],
                    record["timestamp_ms"],
                    viewport_left,
                    viewport_top,
                    viewport_inferences,
                    viewport_shifts,
                )
            case {
                "msg_type": "position_update",
                "flags": int(flags),
                "tank_id": int(tank_id),
                "x": int(x),
                "y": int(y),
                "extra_data": bytes(extra_data),
            }:
                position_update_count += 1
                state = _handle_position_update(
                    state,
                    record["message_index"],
                    record["timestamp_ms"],
                    flags,
                    tank_id,
                    x,
                    y,
                    extra_data,
                    position_evidence,
                )
            case _:
                continue

    extra_x_match_count, extra_y_match_count = _count_matches(position_evidence)
    thirteen_byte_0x2e_count, thirteen_byte_shapes = _collect_thirteen_byte_shapes(
        session,
        xor_table,
    )

    return ViewportAnalysisDict(
        self_tank_id=state["self_tank_id"],
        viewport_inferences=viewport_inferences,
        position_evidence=position_evidence,
        viewport_shifts=viewport_shifts,
        movement_response_count=movement_response_count,
        viewport_update_count=viewport_update_count,
        position_update_count=position_update_count,
        thirteen_byte_0x2e_count=thirteen_byte_0x2e_count,
        thirteen_byte_shapes=thirteen_byte_shapes,
        comparable_position_count=len(position_evidence),
        extra_x_match_count=extra_x_match_count,
        extra_y_match_count=extra_y_match_count,
    )


def _format_capture_status(result: ViewportAnalysisDict) -> str:
    """Return a bounded status line for the capture evidence quality.

    Args:
        result: Viewport analysis result.

    Returns:
        Short capture-status description.
    """
    if result["comparable_position_count"] > 0:
        return "capture_status=position_update_comparable"
    if result["movement_response_count"] == 0:
        return "capture_status=missing_movement_response"
    if result["viewport_update_count"] == 0:
        return "capture_status=missing_viewport_update"
    if result["position_update_count"] == 0:
        return "capture_status=missing_proven_position_update"
    return "capture_status=position_update_not_comparable_yet"


def format_viewport_analysis(result: ViewportAnalysisDict) -> str:
    """Format viewport analysis evidence for terminal output.

    Args:
        result: Analysis result to format.

    Returns:
        Human-readable multiline report.
    """
    lines: list[str] = []
    lines.append(f"self_tank_id={result['self_tank_id']}")
    lines.append(_format_capture_status(result))
    lines.append(f"movement_responses={result['movement_response_count']}")
    lines.append(f"viewport_updates={result['viewport_update_count']}")
    lines.append(f"position_updates={result['position_update_count']}")
    lines.append(f"raw_thirteen_byte_0x2e={result['thirteen_byte_0x2e_count']}")
    lines.append(f"viewport_inferences={len(result['viewport_inferences'])}")
    lines.append(f"viewport_shifts={len(result['viewport_shifts'])}")
    lines.append(
        "position_extra_x_matches="
        f"{result['extra_x_match_count']}/{result['comparable_position_count']}"
    )
    lines.append(
        "position_extra_y_matches="
        f"{result['extra_y_match_count']}/{result['comparable_position_count']}"
    )

    if len(result["thirteen_byte_shapes"]) > 0:
        lines.append("")
        lines.append("13-byte 0x2E shapes:")
        for shape in result["thirteen_byte_shapes"]:
            lines.append(
                "first=0x{first_byte:02X} second=0x{second_byte:02X} count={count}".format(**shape)
            )

    if len(result["viewport_inferences"]) > 0:
        lines.append("")
        lines.append("Viewport inferences:")
        for inference in result["viewport_inferences"]:
            lines.append(
                "[idx={message_index} ts={timestamp_ms}] "
                "viewport=({viewport_left},{viewport_top})".format(**inference)
            )

    if len(result["viewport_shifts"]) > 0:
        lines.append("")
        lines.append("Viewport shifts:")
        for shift in result["viewport_shifts"]:
            lines.append(
                "[idx={message_index} ts={timestamp_ms}] "
                "({old_left},{old_top}) -> ({new_left},{new_top})".format(**shift)
            )

    if len(result["position_evidence"]) > 0:
        lines.append("")
        lines.append("Comparable position updates:")
        for position_evidence in result["position_evidence"]:
            lines.append(
                "[idx={message_index} ts={timestamp_ms}] "
                "pos=({x},{y}) extra=({extra_x},{extra_y}) "
                "expected=({expected_viewport_x},{expected_viewport_y}) "
                "match_x={matches_x} match_y={matches_y}".format(**position_evidence)
            )

    if len(result["position_evidence"]) == 0:
        lines.append("")
        lines.append("No comparable absolute self position_update samples were found.")

    return "\n".join(lines)


__all__ = [
    "PositionViewportEvidenceDict",
    "ThirteenByteShapeDict",
    "ViewportAnalysisDict",
    "ViewportInferenceDict",
    "ViewportShiftDict",
    "analyze_capture_session",
    "decode_position_viewport_evidence",
    "decode_thirteen_byte_shape",
    "decode_viewport_analysis",
    "decode_viewport_inference",
    "decode_viewport_shift",
    "encode_position_viewport_evidence",
    "encode_thirteen_byte_shape",
    "encode_viewport_analysis",
    "encode_viewport_inference",
    "encode_viewport_shift",
    "format_viewport_analysis",
]
