"""Codec for a scene-family sweep together with the conditions it ran under.

The record a cross-device comparison is actually made from. A bare sweep says
which scenes reproduced; it does not say on which card, in which Warp mode, or
under which record bound -- and every one of those changes the answer. Storing
the verdicts without them produces a file that cannot be compared against
another without trusting whoever labelled it.

The body rows are the sweep codec's own rows, reused rather than restated, so
the row layout has exactly one declaration.
"""

from __future__ import annotations

from navprobe.codecs.device_conditions import (
    DEVICE_CONDITIONS_FIELD_COUNT,
    decode_device_conditions,
    encode_device_conditions,
)
from navprobe.codecs.sweep import decode_sweep_entry, encode_sweep_entry
from navprobe.records import SweepRunRecord
from navprobe.wireformat import (
    encode_float_field,
    header_line,
    join_document,
    require_non_negative_float_field,
    require_positive_field,
    split_document,
    split_header_line,
)

#: Banner identifying an encoded sweep run. Bumped to ``/2`` on 2026-08-25 when
#: the line-search block size joined :class:`DeviceRunConditions`. A ``/1``
#: document has one fewer header line, so a ``/2`` decoder must refuse it rather
#: than read the body's first row as a condition.
SWEEP_RUN_BANNER = "navprobe-sweep-run/2"

#: Header lines an encoded sweep run occupies.
SWEEP_RUN_HEADER_FIELD_COUNT = DEVICE_CONDITIONS_FIELD_COUNT + 3


def encode_sweep_run(record: SweepRunRecord) -> str:
    """Encode a sweep run to its text form.

    Args:
        record: The record to encode.

    Returns:
        The encoded text, newline-terminated.
    """
    return join_document(
        [
            SWEEP_RUN_BANNER,
            *encode_device_conditions(record),
            header_line("world_count", str(record["world_count"])),
            header_line("perturbation", encode_float_field(record["perturbation"])),
            header_line("constraint_capacity", str(record["constraint_capacity"])),
            *(encode_sweep_entry(entry) for entry in record["entries"]),
        ]
    )


def decode_sweep_run(text: str) -> SweepRunRecord:
    """Decode a sweep run from its text form.

    Args:
        text: The encoded record.

    Returns:
        The decoded record.

    Raises:
        WireFormatError: When the banner is absent or belongs to another record
            type, a header field is missing or malformed, or a row is
            malformed.
    """
    header, body = split_document(text, SWEEP_RUN_BANNER, SWEEP_RUN_HEADER_FIELD_COUNT)
    conditions = decode_device_conditions(header[:DEVICE_CONDITIONS_FIELD_COUNT])
    return SweepRunRecord(
        mode=conditions["mode"],
        device=conditions["device"],
        device_request=conditions["device_request"],
        max_records=conditions["max_records"],
        linesearch_block_dim=conditions["linesearch_block_dim"],
        world_count=require_positive_field(
            split_header_line(header[DEVICE_CONDITIONS_FIELD_COUNT], "world_count"), "world_count"
        ),
        perturbation=require_non_negative_float_field(
            split_header_line(header[DEVICE_CONDITIONS_FIELD_COUNT + 1], "perturbation"),
            "perturbation",
        ),
        constraint_capacity=require_positive_field(
            split_header_line(header[DEVICE_CONDITIONS_FIELD_COUNT + 2], "constraint_capacity"),
            "constraint_capacity",
        ),
        entries=tuple(decode_sweep_entry(line, position) for position, line in enumerate(body)),
    )


__all__ = [
    "SWEEP_RUN_BANNER",
    "SWEEP_RUN_HEADER_FIELD_COUNT",
    "decode_sweep_run",
    "encode_sweep_run",
]
