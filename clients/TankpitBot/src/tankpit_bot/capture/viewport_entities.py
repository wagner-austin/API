"""Viewport entity extraction from captured sessions.

This module extracts raw ``0x5A`` viewport updates from capture sessions
without imposing gameplay interpretations on the entity tuples. The goal is
to inspect the actual packet classes seen in captures and compare them against
client behavior and other protocol messages.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue, require_int, require_list
from platform_core.logging import get_logger

from tankpit_bot.capture.frames import split_payload_frames
from tankpit_bot.capture.xor import (
    build_session_xor_table,
    xor_decode_body,
)
from tankpit_bot.protocol import try_decode_binary_message
from tankpit_bot.protocol.framing import FramingError
from tankpit_bot.types import CaptureSession

log = get_logger(__name__)


class ViewportEntityRowDict(TypedDict):
    """One raw entity row from a viewport update.

    Attributes:
        abs_x: Absolute x coordinate using the viewport origin from ``0x5A``.
        abs_y: Absolute y coordinate using the viewport origin from ``0x5A``.
        col: Viewport-relative column.
        row: Viewport-relative row.
        cache_value: Raw decoded cache value.
        overlay_value: Raw decoded overlay nibble (or ``255`` sentinel).
        terrain_type: Raw decoded terrain type nibble.
    """

    abs_x: int
    abs_y: int
    col: int
    row: int
    cache_value: int
    overlay_value: int
    terrain_type: int


class ViewportEntityUpdateDict(TypedDict):
    """One decoded viewport update with raw entity rows.

    Attributes:
        message_index: Capture message index.
        timestamp_ms: Capture timestamp.
        viewport_left: Absolute viewport left coordinate from ``0x5A``.
        viewport_top: Absolute viewport top coordinate from ``0x5A``.
        entity_count: Number of decoded entity rows.
        equipment_cache_count: Count of rows with ``cache_value == -1``.
        positive_cache_count: Count of rows with ``cache_value > 0``.
        zero_cache_count: Count of rows with ``cache_value == 0``.
        entities: Raw decoded rows.
    """

    message_index: int
    timestamp_ms: int
    viewport_left: int
    viewport_top: int
    entity_count: int
    equipment_cache_count: int
    positive_cache_count: int
    zero_cache_count: int
    entities: list[ViewportEntityRowDict]


class ViewportEntityDumpDict(TypedDict):
    """Viewport-entity dump derived from a capture session.

    Attributes:
        update_count: Number of decoded viewport updates.
        updates: Raw viewport updates in capture order.
    """

    update_count: int
    updates: list[ViewportEntityUpdateDict]


def encode_viewport_entity_row(row: ViewportEntityRowDict) -> JSONObject:
    """Encode one viewport entity row to JSON.

    Args:
        row: Entity row to encode.

    Returns:
        JSON object representation.
    """
    return {
        "abs_x": row["abs_x"],
        "abs_y": row["abs_y"],
        "col": row["col"],
        "row": row["row"],
        "cache_value": row["cache_value"],
        "overlay_value": row["overlay_value"],
        "terrain_type": row["terrain_type"],
    }


def decode_viewport_entity_row(data: JSONObject) -> ViewportEntityRowDict:
    """Decode one viewport entity row from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated entity row.
    """
    return ViewportEntityRowDict(
        abs_x=require_int(data, "abs_x"),
        abs_y=require_int(data, "abs_y"),
        col=require_int(data, "col"),
        row=require_int(data, "row"),
        cache_value=require_int(data, "cache_value"),
        overlay_value=require_int(data, "overlay_value"),
        terrain_type=require_int(data, "terrain_type"),
    )


def encode_viewport_entity_update(update: ViewportEntityUpdateDict) -> JSONObject:
    """Encode one viewport update to JSON.

    Args:
        update: Viewport update to encode.

    Returns:
        JSON object representation.
    """
    entities_json: list[JSONValue] = [encode_viewport_entity_row(row) for row in update["entities"]]
    return {
        "message_index": update["message_index"],
        "timestamp_ms": update["timestamp_ms"],
        "viewport_left": update["viewport_left"],
        "viewport_top": update["viewport_top"],
        "entity_count": update["entity_count"],
        "equipment_cache_count": update["equipment_cache_count"],
        "positive_cache_count": update["positive_cache_count"],
        "zero_cache_count": update["zero_cache_count"],
        "entities": entities_json,
    }


def decode_viewport_entity_update(data: JSONObject) -> ViewportEntityUpdateDict:
    """Decode one viewport update from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated viewport update.

    Raises:
        JSONTypeError: If any nested entity is not an object.
    """
    raw_entities = require_list(data, "entities")
    entities: list[ViewportEntityRowDict] = []
    for index, entry in enumerate(raw_entities):
        if not isinstance(entry, dict):
            raise JSONTypeError(f"entities[{index}] must be an object")
        entities.append(decode_viewport_entity_row(entry))
    return ViewportEntityUpdateDict(
        message_index=require_int(data, "message_index"),
        timestamp_ms=require_int(data, "timestamp_ms"),
        viewport_left=require_int(data, "viewport_left"),
        viewport_top=require_int(data, "viewport_top"),
        entity_count=require_int(data, "entity_count"),
        equipment_cache_count=require_int(data, "equipment_cache_count"),
        positive_cache_count=require_int(data, "positive_cache_count"),
        zero_cache_count=require_int(data, "zero_cache_count"),
        entities=entities,
    )


def encode_viewport_entity_dump(result: ViewportEntityDumpDict) -> JSONObject:
    """Encode viewport-entity dump to JSON.

    Args:
        result: Dump result to encode.

    Returns:
        JSON object representation.
    """
    updates_json: list[JSONValue] = [
        encode_viewport_entity_update(row) for row in result["updates"]
    ]
    return {"update_count": result["update_count"], "updates": updates_json}


def decode_viewport_entity_dump(data: JSONObject) -> ViewportEntityDumpDict:
    """Decode viewport-entity dump from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated viewport-entity dump.

    Raises:
        JSONTypeError: If any nested update is not an object.
    """
    raw_updates = require_list(data, "updates")
    updates: list[ViewportEntityUpdateDict] = []
    for index, entry in enumerate(raw_updates):
        if not isinstance(entry, dict):
            raise JSONTypeError(f"updates[{index}] must be an object")
        updates.append(decode_viewport_entity_update(entry))
    return ViewportEntityDumpDict(
        update_count=require_int(data, "update_count"),
        updates=updates,
    )


def _class_counts(entities: list[ViewportEntityRowDict]) -> tuple[int, int, int]:
    """Count raw cache-value classes in one viewport update.

    Args:
        entities: Raw entity rows.

    Returns:
        Tuple of ``(equipment_cache_count, positive_cache_count, zero_cache_count)``.
    """
    equipment_cache_count = 0
    positive_cache_count = 0
    zero_cache_count = 0
    for entity in entities:
        if entity["cache_value"] == -1:
            equipment_cache_count += 1
        elif entity["cache_value"] > 0:
            positive_cache_count += 1
        else:
            zero_cache_count += 1
    return equipment_cache_count, positive_cache_count, zero_cache_count


def _build_entity_rows(
    viewport_left: int,
    viewport_top: int,
    raw_entities: list[dict[str, int]],
) -> list[ViewportEntityRowDict]:
    """Build typed entity rows from raw decoded viewport entities.

    Args:
        viewport_left: Absolute viewport left coordinate.
        viewport_top: Absolute viewport top coordinate.
        raw_entities: Raw entity dicts from the ``0x5A`` decoder.

    Returns:
        Typed entity rows with absolute and relative coordinates.
    """
    entities: list[ViewportEntityRowDict] = []
    for raw_entity in raw_entities:
        entities.append(
            ViewportEntityRowDict(
                abs_x=viewport_left + raw_entity["col"],
                abs_y=viewport_top + raw_entity["row"],
                col=raw_entity["col"],
                row=raw_entity["row"],
                cache_value=raw_entity["cache_value"],
                overlay_value=raw_entity["overlay_value"],
                terrain_type=raw_entity["terrain_type"],
            )
        )
    return entities


def _decode_viewport_update(
    body: bytes,
    xor_table: bytes,
    message_index: int,
    timestamp_ms: int,
) -> ViewportEntityUpdateDict | None:
    """Decode one frame as a raw viewport update if it matches ``0x5A``.

    Args:
        body: Raw frame body including the type byte.
        xor_table: Session XOR table.
        message_index: Capture message index.
        timestamp_ms: Capture message timestamp.

    Returns:
        Decoded viewport update or ``None`` when the frame is not ``0x5A``.
    """
    decoded_data = xor_decode_body(body, xor_table, offset=1)
    parsed = try_decode_binary_message(body[0], decoded_data)
    match parsed:
        case {
            "msg_type": 0x5A,
            "viewport_left": int(viewport_left),
            "viewport_top": int(viewport_top),
            "entities": list(raw_entities),
        }:
            entities = _build_entity_rows(viewport_left, viewport_top, raw_entities)
            equipment_cache_count, positive_cache_count, zero_cache_count = _class_counts(entities)
            return ViewportEntityUpdateDict(
                message_index=message_index,
                timestamp_ms=timestamp_ms,
                viewport_left=viewport_left,
                viewport_top=viewport_top,
                entity_count=len(entities),
                equipment_cache_count=equipment_cache_count,
                positive_cache_count=positive_cache_count,
                zero_cache_count=zero_cache_count,
                entities=entities,
            )
        case _:
            return None


def _split_received_frames(payload: str) -> list[bytes]:
    """Split one received payload into logical frames.

    Args:
        payload: Base64-encoded payload from a received capture message.

    Returns:
        Logical frame bodies or an empty list when the payload is invalid.
    """
    try:
        return split_payload_frames(payload)
    except FramingError as exc:
        log.warning("Skipping malformed capture payload during viewport entity dump: %s", exc)
        return []


def analyze_viewport_entities(session: CaptureSession) -> ViewportEntityDumpDict:
    """Extract raw ``0x5A`` viewport updates from a capture session.

    Args:
        session: Capture session to inspect.

    Returns:
        Raw viewport-entity dump derived from decoded ``0x5A`` packets.

    Raises:
        ValueError: If the session has no magic key or static key cannot be loaded.
    """
    magic = session["magic"]
    if magic is None:
        raise ValueError("Capture session has no magic key")

    xor_table = build_session_xor_table(magic)

    updates: list[ViewportEntityUpdateDict] = []
    for message_index, message in enumerate(session["messages"]):
        if message["direction"] != "received":
            continue
        frames = _split_received_frames(message["payload"])
        for body in frames:
            update = _decode_viewport_update(
                body,
                xor_table,
                message_index,
                message["timestamp_ms"],
            )
            if update is not None:
                updates.append(update)
    return ViewportEntityDumpDict(update_count=len(updates), updates=updates)


def format_viewport_entity_dump(result: ViewportEntityDumpDict) -> str:
    """Format raw viewport-entity dump for terminal inspection.

    Args:
        result: Dump result to format.

    Returns:
        Human-readable multiline report.
    """
    lines: list[str] = [f"viewport_updates={result['update_count']}"]
    for update in result["updates"]:
        lines.append("")
        lines.append(
            "[idx={message_index} ts={timestamp_ms}] viewport=({viewport_left},{viewport_top}) "
            "entities={entity_count} equipment_cache={equipment_cache_count} "
            "positive_cache={positive_cache_count} zero_cache={zero_cache_count}".format(**update)
        )
        for entity in update["entities"]:
            lines.append(
                "  abs=({abs_x},{abs_y}) cell=({col},{row}) cache_value={cache_value} "
                "overlay_value={overlay_value} terrain={terrain_type}".format(**entity)
            )
    return "\n".join(lines)


__all__ = [
    "ViewportEntityDumpDict",
    "ViewportEntityRowDict",
    "ViewportEntityUpdateDict",
    "analyze_viewport_entities",
    "decode_viewport_entity_dump",
    "decode_viewport_entity_row",
    "decode_viewport_entity_update",
    "encode_viewport_entity_dump",
    "encode_viewport_entity_row",
    "encode_viewport_entity_update",
    "format_viewport_entity_dump",
]
