"""Correlate sent shoot commands with raw viewport entities.

This module joins sent ``SHOOT`` commands with the most recent proven ``0x5A``
viewport update so captures can confirm whether shoot ``target_id`` values
match the identity space used inside viewport entity rows.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_bool,
    require_int,
    require_list,
)
from platform_core.logging import get_logger

from tankpit_bot.capture.viewport_entities import (
    ViewportEntityRowDict,
    ViewportEntityUpdateDict,
    analyze_viewport_entities,
    decode_viewport_entity_row,
    encode_viewport_entity_row,
)
from tankpit_bot.capture.xor import (
    build_xor_table,
    decode_base64_safe,
    load_xor_static_key,
    xor_decode_body,
)
from tankpit_bot.protocol.commands import CMD_SHOOT, deserialize_command
from tankpit_bot.protocol.framing import FramingError, split_frames
from tankpit_bot.types import CaptureSession

log = get_logger(__name__)


class ShotViewportCorrelationDict(TypedDict):
    """One shoot command correlated with the most recent viewport update.

    Attributes:
        shot_index: Capture message index of the sent shoot command.
        shot_timestamp_ms: Timestamp of the sent shoot command.
        target_x: Shot target x coordinate.
        target_y: Shot target y coordinate.
        target_id: Shot target entity id.
        viewport_found: Whether a prior viewport update was available.
        viewport_index: Capture message index of the correlated viewport update.
        viewport_timestamp_ms: Timestamp of the correlated viewport update.
        viewport_left: Absolute viewport left coordinate.
        viewport_top: Absolute viewport top coordinate.
        positive_row_count: Number of positive-id rows in the viewport update.
        anonymous_row_count: Number of ``entity_id == -1`` rows in the viewport update.
        id_match_count: Number of positive rows matching ``target_id``.
        coord_match_count: Number of rows matching ``target_x,target_y``.
        positive_rows: All positive-id rows in the correlated viewport update.
        anonymous_rows: All ``entity_id == -1`` rows in the correlated viewport update.
        id_matches: Positive rows matching ``target_id``.
        coord_matches: Rows matching the shot coordinates.
    """

    shot_index: int
    shot_timestamp_ms: int
    target_x: int
    target_y: int
    target_id: int
    viewport_found: bool
    viewport_index: int
    viewport_timestamp_ms: int
    viewport_left: int
    viewport_top: int
    positive_row_count: int
    anonymous_row_count: int
    id_match_count: int
    coord_match_count: int
    positive_rows: list[ViewportEntityRowDict]
    anonymous_rows: list[ViewportEntityRowDict]
    id_matches: list[ViewportEntityRowDict]
    coord_matches: list[ViewportEntityRowDict]


class ShotViewportCorrelationDumpDict(TypedDict):
    """Full shoot-to-viewport correlation result for a capture session.

    Attributes:
        shot_count: Number of sent shoot commands correlated.
        shots: Correlation rows in capture order.
    """

    shot_count: int
    shots: list[ShotViewportCorrelationDict]


def encode_shot_viewport_correlation(row: ShotViewportCorrelationDict) -> JSONObject:
    """Encode one correlation row to JSON.

    Args:
        row: Correlation row to encode.

    Returns:
        JSON object representation.
    """
    positive_rows_json: list[JSONValue] = [
        encode_viewport_entity_row(entry) for entry in row["positive_rows"]
    ]
    anonymous_rows_json: list[JSONValue] = [
        encode_viewport_entity_row(entry) for entry in row["anonymous_rows"]
    ]
    id_matches_json: list[JSONValue] = [
        encode_viewport_entity_row(entry) for entry in row["id_matches"]
    ]
    coord_matches_json: list[JSONValue] = [
        encode_viewport_entity_row(entry) for entry in row["coord_matches"]
    ]
    return {
        "shot_index": row["shot_index"],
        "shot_timestamp_ms": row["shot_timestamp_ms"],
        "target_x": row["target_x"],
        "target_y": row["target_y"],
        "target_id": row["target_id"],
        "viewport_found": row["viewport_found"],
        "viewport_index": row["viewport_index"],
        "viewport_timestamp_ms": row["viewport_timestamp_ms"],
        "viewport_left": row["viewport_left"],
        "viewport_top": row["viewport_top"],
        "positive_row_count": row["positive_row_count"],
        "anonymous_row_count": row["anonymous_row_count"],
        "id_match_count": row["id_match_count"],
        "coord_match_count": row["coord_match_count"],
        "positive_rows": positive_rows_json,
        "anonymous_rows": anonymous_rows_json,
        "id_matches": id_matches_json,
        "coord_matches": coord_matches_json,
    }


def _decode_entity_rows(data: JSONObject, key: str) -> list[ViewportEntityRowDict]:
    """Decode one nested list of viewport entity rows.

    Args:
        data: JSON object containing the nested list.
        key: Field name to decode.

    Returns:
        Decoded entity rows.

    Raises:
        JSONTypeError: If any nested row is not an object.
    """
    raw_rows = require_list(data, key)
    rows: list[ViewportEntityRowDict] = []
    for index, entry in enumerate(raw_rows):
        if not isinstance(entry, dict):
            raise JSONTypeError(f"{key}[{index}] must be an object")
        rows.append(decode_viewport_entity_row(entry))
    return rows


def decode_shot_viewport_correlation(data: JSONObject) -> ShotViewportCorrelationDict:
    """Decode one correlation row from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated correlation row.
    """
    return ShotViewportCorrelationDict(
        shot_index=require_int(data, "shot_index"),
        shot_timestamp_ms=require_int(data, "shot_timestamp_ms"),
        target_x=require_int(data, "target_x"),
        target_y=require_int(data, "target_y"),
        target_id=require_int(data, "target_id"),
        viewport_found=require_bool(data, "viewport_found"),
        viewport_index=require_int(data, "viewport_index"),
        viewport_timestamp_ms=require_int(data, "viewport_timestamp_ms"),
        viewport_left=require_int(data, "viewport_left"),
        viewport_top=require_int(data, "viewport_top"),
        positive_row_count=require_int(data, "positive_row_count"),
        anonymous_row_count=require_int(data, "anonymous_row_count"),
        id_match_count=require_int(data, "id_match_count"),
        coord_match_count=require_int(data, "coord_match_count"),
        positive_rows=_decode_entity_rows(data, "positive_rows"),
        anonymous_rows=_decode_entity_rows(data, "anonymous_rows"),
        id_matches=_decode_entity_rows(data, "id_matches"),
        coord_matches=_decode_entity_rows(data, "coord_matches"),
    )


def encode_shot_viewport_correlation_dump(result: ShotViewportCorrelationDumpDict) -> JSONObject:
    """Encode a correlation dump to JSON.

    Args:
        result: Correlation dump to encode.

    Returns:
        JSON object representation.
    """
    shots_json: list[JSONValue] = [
        encode_shot_viewport_correlation(entry) for entry in result["shots"]
    ]
    return {"shot_count": result["shot_count"], "shots": shots_json}


def decode_shot_viewport_correlation_dump(data: JSONObject) -> ShotViewportCorrelationDumpDict:
    """Decode a correlation dump from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated correlation dump.

    Raises:
        JSONTypeError: If any nested row is not an object.
    """
    raw_shots = require_list(data, "shots")
    shots: list[ShotViewportCorrelationDict] = []
    for index, entry in enumerate(raw_shots):
        if not isinstance(entry, dict):
            raise JSONTypeError(f"shots[{index}] must be an object")
        shots.append(decode_shot_viewport_correlation(entry))
    return ShotViewportCorrelationDumpDict(
        shot_count=require_int(data, "shot_count"),
        shots=shots,
    )


def _split_sent_frames(payload: str) -> list[bytes]:
    """Split one sent payload into logical frames.

    Args:
        payload: Base64-encoded sent payload.

    Returns:
        Logical frame bodies or an empty list when the payload is invalid.
    """
    raw_bytes = decode_base64_safe(payload)
    if raw_bytes is None:
        return []
    try:
        return split_frames(raw_bytes)
    except FramingError as exc:
        log.warning("Skipping malformed sent payload during shot correlation: %s", exc)
        return []


def _decode_shoot_command(body: bytes, xor_table: bytes) -> tuple[int, int, int] | None:
    """Decode one sent frame as a shoot command if possible.

    Args:
        body: Raw framed command body.
        xor_table: Session XOR table for sent-command decoding.

    Returns:
        Tuple of ``(target_x, target_y, target_id)`` or ``None``.
    """
    if len(body) < 3:
        return None
    decoded = body[:1] + xor_decode_body(body[1:], xor_table)
    try:
        command = deserialize_command(decoded, decoded[1])
    except ValueError as exc:
        log.warning("Skipping invalid sent command during shot correlation: %s", exc)
        return None
    match command:
        case {"kind": "action", "cmd_id": int(cmd_id), "data": str(data_hex)}:
            if cmd_id != CMD_SHOOT:
                return None
            data = bytes.fromhex(data_hex)
            if len(data) != 4:
                return None
            target_id = data[2] | (data[3] << 8)
            return data[0], data[1], target_id
        case _:
            return None


def _build_correlation(
    shot_index: int,
    shot_timestamp_ms: int,
    target_x: int,
    target_y: int,
    target_id: int,
    last_viewport: ViewportEntityUpdateDict | None,
) -> ShotViewportCorrelationDict:
    """Build one correlation row from a shot and the latest viewport update.

    Args:
        shot_index: Capture message index of the shot.
        shot_timestamp_ms: Shot timestamp.
        target_x: Shot target x coordinate.
        target_y: Shot target y coordinate.
        target_id: Shot target entity id.
        last_viewport: Most recent viewport update or ``None``.

    Returns:
        Correlation row.
    """
    if last_viewport is None:
        return ShotViewportCorrelationDict(
            shot_index=shot_index,
            shot_timestamp_ms=shot_timestamp_ms,
            target_x=target_x,
            target_y=target_y,
            target_id=target_id,
            viewport_found=False,
            viewport_index=-1,
            viewport_timestamp_ms=-1,
            viewport_left=-1,
            viewport_top=-1,
            positive_row_count=0,
            anonymous_row_count=0,
            id_match_count=0,
            coord_match_count=0,
            positive_rows=[],
            anonymous_rows=[],
            id_matches=[],
            coord_matches=[],
        )

    positive_rows = [row for row in last_viewport["entities"] if row["entity_id"] > 0]
    anonymous_rows = [row for row in last_viewport["entities"] if row["entity_id"] == -1]
    id_matches = [row for row in positive_rows if row["entity_id"] == target_id]
    coord_matches = [
        row
        for row in last_viewport["entities"]
        if row["abs_x"] == target_x and row["abs_y"] == target_y
    ]
    return ShotViewportCorrelationDict(
        shot_index=shot_index,
        shot_timestamp_ms=shot_timestamp_ms,
        target_x=target_x,
        target_y=target_y,
        target_id=target_id,
        viewport_found=True,
        viewport_index=last_viewport["message_index"],
        viewport_timestamp_ms=last_viewport["timestamp_ms"],
        viewport_left=last_viewport["viewport_left"],
        viewport_top=last_viewport["viewport_top"],
        positive_row_count=len(positive_rows),
        anonymous_row_count=len(anonymous_rows),
        id_match_count=len(id_matches),
        coord_match_count=len(coord_matches),
        positive_rows=positive_rows,
        anonymous_rows=anonymous_rows,
        id_matches=id_matches,
        coord_matches=coord_matches,
    )


def analyze_shot_viewport_correlation(session: CaptureSession) -> ShotViewportCorrelationDumpDict:
    """Correlate sent shoot commands with the latest proven viewport update.

    Args:
        session: Capture session to inspect.

    Returns:
        Shot-to-viewport correlation rows in capture order.

    Raises:
        ValueError: If viewport extraction prerequisites are missing.
    """
    magic = session["magic"]
    if magic is None:
        raise ValueError("Capture session has no magic key")
    static_key, _ = load_xor_static_key(None)
    if static_key is None:
        raise ValueError("Could not load xor_static_key.txt")
    xor_table = build_xor_table(static_key, magic)
    viewport_dump = analyze_viewport_entities(session)
    updates = viewport_dump["updates"]
    update_index = 0
    last_viewport: ViewportEntityUpdateDict | None = None
    shots: list[ShotViewportCorrelationDict] = []

    for message_index, message in enumerate(session["messages"]):
        while (
            update_index < len(updates) and updates[update_index]["message_index"] <= message_index
        ):
            last_viewport = updates[update_index]
            update_index += 1
        if message["direction"] != "sent":
            continue
        for body in _split_sent_frames(message["payload"]):
            shot = _decode_shoot_command(body, xor_table)
            if shot is None:
                continue
            target_x, target_y, target_id = shot
            shots.append(
                _build_correlation(
                    message_index,
                    message["timestamp_ms"],
                    target_x,
                    target_y,
                    target_id,
                    last_viewport,
                )
            )

    return ShotViewportCorrelationDumpDict(shot_count=len(shots), shots=shots)


def format_shot_viewport_correlation(result: ShotViewportCorrelationDumpDict) -> str:
    """Format shot-to-viewport correlations for terminal inspection.

    Args:
        result: Correlation dump to format.

    Returns:
        Human-readable multiline report.
    """
    lines = [f"shot_count={result['shot_count']}"]
    for shot in result["shots"]:
        lines.append("")
        lines.append(
            "[idx={shot_index} ts={shot_timestamp_ms}] shot=({target_x},{target_y}) "
            "target_id={target_id} viewport_found={viewport_found} "
            "viewport_idx={viewport_index} viewport=({viewport_left},{viewport_top}) "
            "id_matches={id_match_count} coord_matches={coord_match_count} "
            "positive_rows={positive_row_count} anonymous_rows={anonymous_row_count}".format(**shot)
        )
        for row in shot["id_matches"]:
            lines.append(
                "  id_match abs=({abs_x},{abs_y}) cell=({col},{row}) entity_id={entity_id} "
                "value={value} terrain={terrain_type}".format(**row)
            )
        for row in shot["coord_matches"]:
            lines.append(
                "  coord_match abs=({abs_x},{abs_y}) cell=({col},{row}) entity_id={entity_id} "
                "value={value} terrain={terrain_type}".format(**row)
            )
    return "\n".join(lines)


__all__ = [
    "ShotViewportCorrelationDict",
    "ShotViewportCorrelationDumpDict",
    "analyze_shot_viewport_correlation",
    "decode_shot_viewport_correlation",
    "decode_shot_viewport_correlation_dump",
    "encode_shot_viewport_correlation",
    "encode_shot_viewport_correlation_dump",
    "format_shot_viewport_correlation",
]
