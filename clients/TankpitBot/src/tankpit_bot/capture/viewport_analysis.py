"""Viewport analysis for captured sessions.

Derives viewport origin evidence from captured protocol traffic so
viewport semantics can be verified from real packets rather than inferred
from client behavior.
"""

from __future__ import annotations

from collections import Counter

from tankpit_bot.capture.viewport_analysis_types import (
    DecodedBinaryRecordDict,
    PositionViewportEvidenceDict,
    ThirteenByteShapeDict,
    ViewportAnalysisDict,
    ViewportAnalysisStateDict,
    ViewportInferenceDict,
    ViewportShiftDict,
    decode_position_viewport_evidence,
    decode_thirteen_byte_shape,
    decode_viewport_analysis,
    decode_viewport_inference,
    decode_viewport_shift,
    encode_position_viewport_evidence,
    encode_thirteen_byte_shape,
    encode_viewport_analysis,
    encode_viewport_inference,
    encode_viewport_shift,
)
from tankpit_bot.capture.xor import decode_base64_safe, xor_decode_body
from tankpit_bot.protocol import try_decode_binary_message
from tankpit_bot.state.viewport_geometry import VIEWPORT_PATCH_WIDTH
from tankpit_bot.types.session import CaptureSession


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
        patch grid.
    """
    return x >= VIEWPORT_PATCH_WIDTH or y >= VIEWPORT_PATCH_WIDTH


def _decode_received_binary_records(
    session: CaptureSession,
    xor_table: bytes,
) -> list[DecodedBinaryRecordDict]:
    """Decode all received binary messages from a capture session.

    Args:
        session: Capture session to inspect.
        xor_table: XOR table for the session magic.

    Returns:
        Decoded received binary message records.
    """
    records: list[DecodedBinaryRecordDict] = []
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
                DecodedBinaryRecordDict(
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
    """Return a deterministic sort key for 13-byte shape counts."""
    return (-item[1], item[0][0], item[0][1])


def _handle_movement_response(
    state: ViewportAnalysisStateDict,
    tank_id: int,
) -> ViewportAnalysisStateDict:
    """Update analysis state from a MovementResponse."""
    self_tank_id = state["self_tank_id"]
    if self_tank_id is None:
        self_tank_id = tank_id

    return ViewportAnalysisStateDict(
        self_tank_id=self_tank_id,
        current_viewport_left=state["current_viewport_left"],
        current_viewport_top=state["current_viewport_top"],
    )


def _handle_viewport_update(
    state: ViewportAnalysisStateDict,
    message_index: int,
    timestamp_ms: int,
    viewport_left: int,
    viewport_top: int,
    viewport_inferences: list[ViewportInferenceDict],
    viewport_shifts: list[ViewportShiftDict],
) -> ViewportAnalysisStateDict:
    """Update analysis state from a ViewportUpdate."""
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

    return ViewportAnalysisStateDict(
        self_tank_id=state["self_tank_id"],
        current_viewport_left=viewport_left,
        current_viewport_top=viewport_top,
    )


def _handle_position_update(
    state: ViewportAnalysisStateDict,
    message_index: int,
    timestamp_ms: int,
    flags: int,
    tank_id: int,
    x: int,
    y: int,
    extra_data: bytes,
    position_evidence: list[PositionViewportEvidenceDict],
) -> ViewportAnalysisStateDict:
    """Update analysis state from a self absolute position_update."""
    if (flags & 0x02) == 0:
        return state

    self_tank_id = state["self_tank_id"]
    if self_tank_id is None:
        self_tank_id = tank_id
    if tank_id != self_tank_id:
        return state
    if not _is_absolute_position(x, y):
        return ViewportAnalysisStateDict(
            self_tank_id=self_tank_id,
            current_viewport_left=state["current_viewport_left"],
            current_viewport_top=state["current_viewport_top"],
        )
    if len(extra_data) < 2:
        return ViewportAnalysisStateDict(
            self_tank_id=self_tank_id,
            current_viewport_left=state["current_viewport_left"],
            current_viewport_top=state["current_viewport_top"],
        )

    current_viewport_left = state["current_viewport_left"]
    current_viewport_top = state["current_viewport_top"]
    if current_viewport_left is None or current_viewport_top is None:
        return ViewportAnalysisStateDict(
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
    return ViewportAnalysisStateDict(
        self_tank_id=self_tank_id,
        current_viewport_left=current_viewport_left,
        current_viewport_top=current_viewport_top,
    )


def _count_matches(position_evidence: list[PositionViewportEvidenceDict]) -> tuple[int, int]:
    """Count x and y matches in comparable position evidence."""
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
    state = ViewportAnalysisStateDict(
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
    """Return a bounded status line for the capture evidence quality."""
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
        for pe in result["position_evidence"]:
            lines.append(
                "[idx={message_index} ts={timestamp_ms}] "
                "pos=({x},{y}) extra=({extra_x},{extra_y}) "
                "expected=({expected_viewport_x},{expected_viewport_y}) "
                "match_x={matches_x} match_y={matches_y}".format(**pe)
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
