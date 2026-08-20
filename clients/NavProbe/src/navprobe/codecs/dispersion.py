"""Codec for dispersion records.

The two spread fields are floats and are encoded exactly, through the same
hexadecimal form the scene codec uses. A dispersion figure is the number a
finding turns on — whether a disagreement is a nanometre or half a metre — so it
is stored without the rounding a decimal repr would invite.

Unlike a scene's floats, a spread may legitimately be **zero**: a deterministic
configuration has no spread at all, and that is the most important value the
record can carry. It is therefore validated as non-negative rather than
positive.
"""

from __future__ import annotations

from navprobe.records import DispersionRecord
from navprobe.wireformat import (
    encode_float_field,
    header_line,
    join_document,
    require_no_body,
    require_non_negative_field,
    require_non_negative_float_field,
    require_positive_field,
    split_document,
    split_header_line,
)

#: Banner identifying an encoded dispersion record.
DISPERSION_BANNER = "navprobe-dispersion/1"

#: Header lines an encoded dispersion record occupies.
DISPERSION_HEADER_FIELD_COUNT = 4


def encode_dispersion_record(record: DispersionRecord) -> str:
    """Encode a dispersion record to its text form.

    Args:
        record: The record to encode.

    Returns:
        The encoded text, newline-terminated.
    """
    return join_document(
        [
            DISPERSION_BANNER,
            header_line("repetitions", str(record["repetitions"])),
            header_line("observation_length", str(record["observation_length"])),
            header_line("max_spread", encode_float_field(record["max_spread"])),
            header_line("mean_spread", encode_float_field(record["mean_spread"])),
        ]
    )


def decode_dispersion_record(text: str) -> DispersionRecord:
    """Decode a dispersion record from its text form.

    Args:
        text: The encoded record.

    Returns:
        The decoded record.

    Raises:
        WireFormatError: When the banner is absent or belongs to another record
            type, a header field is missing or malformed, or lines trail the
            header.
    """
    header, body = split_document(text, DISPERSION_BANNER, DISPERSION_HEADER_FIELD_COUNT)
    require_no_body(body, DISPERSION_BANNER)
    return DispersionRecord(
        repetitions=require_positive_field(
            split_header_line(header[0], "repetitions"), "repetitions"
        ),
        observation_length=require_non_negative_field(
            split_header_line(header[1], "observation_length"), "observation_length"
        ),
        max_spread=require_non_negative_float_field(
            split_header_line(header[2], "max_spread"), "max_spread"
        ),
        mean_spread=require_non_negative_float_field(
            split_header_line(header[3], "mean_spread"), "mean_spread"
        ),
    )


__all__ = [
    "DISPERSION_BANNER",
    "DISPERSION_HEADER_FIELD_COUNT",
    "decode_dispersion_record",
    "encode_dispersion_record",
]
