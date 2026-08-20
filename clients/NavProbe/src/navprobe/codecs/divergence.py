"""Codec for divergence records.

The magnitudes are floats and are stored exactly, through the same hexadecimal
form the scene and dispersion codecs use. A divergence figure is what decides
whether a disagreement is an artefact or a finding, so it is not rounded on the
way to disk.

Both magnitudes may legitimately be **zero** — that is what two agreeing
configurations produce, and it is the value most worth being able to store — so
they are validated as non-negative rather than positive, reusing the dispersion
codec's check rather than restating it.
"""

from __future__ import annotations

from navprobe.records import DivergenceRecord
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

#: Banner identifying an encoded divergence record.
DIVERGENCE_BANNER = "navprobe-divergence/1"

#: Header lines an encoded divergence record occupies.
DIVERGENCE_HEADER_FIELD_COUNT = 4


def encode_divergence_record(record: DivergenceRecord) -> str:
    """Encode a divergence record to its text form.

    Args:
        record: The record to encode.

    Returns:
        The encoded text, newline-terminated.
    """
    return join_document(
        [
            DIVERGENCE_BANNER,
            header_line("observation_length", str(record["observation_length"])),
            header_line("differing_elements", str(record["differing_elements"])),
            header_line(
                "max_absolute_difference",
                encode_float_field(record["max_absolute_difference"]),
            ),
            header_line(
                "mean_absolute_difference",
                encode_float_field(record["mean_absolute_difference"]),
            ),
        ]
    )


def decode_divergence_record(text: str) -> DivergenceRecord:
    """Decode a divergence record from its text form.

    ``observation_length`` is required to be positive because a comparison over
    no elements is refused at the point it is made; a stored record claiming one
    describes a measurement that cannot have happened.

    Args:
        text: The encoded record.

    Returns:
        The decoded record.

    Raises:
        WireFormatError: When the banner is absent or belongs to another record
            type, a header field is missing or malformed, or lines trail the
            header.
    """
    header, body = split_document(text, DIVERGENCE_BANNER, DIVERGENCE_HEADER_FIELD_COUNT)
    require_no_body(body, DIVERGENCE_BANNER)
    return DivergenceRecord(
        observation_length=require_positive_field(
            split_header_line(header[0], "observation_length"), "observation_length"
        ),
        differing_elements=require_non_negative_field(
            split_header_line(header[1], "differing_elements"), "differing_elements"
        ),
        max_absolute_difference=require_non_negative_float_field(
            split_header_line(header[2], "max_absolute_difference"),
            "max_absolute_difference",
        ),
        mean_absolute_difference=require_non_negative_float_field(
            split_header_line(header[3], "mean_absolute_difference"),
            "mean_absolute_difference",
        ),
    )


__all__ = [
    "DIVERGENCE_BANNER",
    "DIVERGENCE_HEADER_FIELD_COUNT",
    "decode_divergence_record",
    "encode_divergence_record",
]
