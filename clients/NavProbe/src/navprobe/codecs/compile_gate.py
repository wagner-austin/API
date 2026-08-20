"""Codec for evidence that a determinism mode compiled and stepped.

The record has no failure form, and that is deliberate. A mode Warp rejects
raises out of the gate carrying the vendor's own error -- file, line and the
reduction families that conflicted -- which is a better record of the rejection
than any summary this package could store. Catching it to write a "failed"
document would replace that with a string, and would also let a genuine bug be
filed as a determinism-mode rejection.

So this record means one thing: on this device, in this mode, the pipeline
compiled cold and stepped. Its absence is the other answer.
"""

from __future__ import annotations

from navprobe.codecs.device_conditions import (
    DEVICE_CONDITIONS_FIELD_COUNT,
    decode_device_conditions,
    encode_device_conditions,
)
from navprobe.codecs.scene import (
    SCENE_FIELD_COUNT,
    decode_scene_headers,
    encode_scene_headers,
)
from navprobe.records import CompileGateRecord
from navprobe.wireformat import (
    encode_float_field,
    header_line,
    join_document,
    require_no_body,
    require_non_negative_float_field,
    require_positive_field,
    split_document,
    split_header_line,
)

#: Banner identifying an encoded compile-gate result.
COMPILE_GATE_BANNER = "navprobe-compile-gate/1"

#: Header lines an encoded compile-gate result occupies.
COMPILE_GATE_HEADER_FIELD_COUNT = DEVICE_CONDITIONS_FIELD_COUNT + 2 + SCENE_FIELD_COUNT


def encode_compile_gate(record: CompileGateRecord) -> str:
    """Encode a compile-gate result to its text form.

    Args:
        record: The record to encode.

    Returns:
        The encoded text, newline-terminated.
    """
    return join_document(
        [
            COMPILE_GATE_BANNER,
            *encode_device_conditions(record),
            header_line("wall_seconds", encode_float_field(record["wall_seconds"])),
            header_line("world_count", str(record["world_count"])),
            *encode_scene_headers(record["scene"]),
        ]
    )


def decode_compile_gate(text: str) -> CompileGateRecord:
    """Decode a compile-gate result from its text form.

    Args:
        text: The encoded record.

    Returns:
        The decoded record.

    Raises:
        WireFormatError: When the banner is absent or belongs to another record
            type, a header field is missing or malformed, or lines trail the
            header.
    """
    header, body = split_document(text, COMPILE_GATE_BANNER, COMPILE_GATE_HEADER_FIELD_COUNT)
    require_no_body(body, COMPILE_GATE_BANNER)
    conditions = decode_device_conditions(header[:DEVICE_CONDITIONS_FIELD_COUNT])
    wall_at = DEVICE_CONDITIONS_FIELD_COUNT
    scene_at = wall_at + 2
    return CompileGateRecord(
        mode=conditions["mode"],
        device=conditions["device"],
        device_request=conditions["device_request"],
        max_records=conditions["max_records"],
        wall_seconds=require_non_negative_float_field(
            split_header_line(header[wall_at], "wall_seconds"), "wall_seconds"
        ),
        world_count=require_positive_field(
            split_header_line(header[wall_at + 1], "world_count"), "world_count"
        ),
        scene=decode_scene_headers(header[scene_at : scene_at + SCENE_FIELD_COUNT]),
    )


__all__ = [
    "COMPILE_GATE_BANNER",
    "COMPILE_GATE_HEADER_FIELD_COUNT",
    "decode_compile_gate",
    "encode_compile_gate",
]
