"""Codec for comparison records.

A comparison record is a verdict about two rollouts and carries no body. Any
line after the header means the text is not the record it claims to be, so the
decoder refuses it rather than ignoring it.
"""

from __future__ import annotations

from navprobe.records import ComparisonRecord
from navprobe.wireformat import (
    encode_bool,
    encode_optional_int,
    header_line,
    join_document,
    require_bool_field,
    require_no_body,
    require_non_negative_field,
    require_optional_non_negative_field,
    require_text_field,
    split_document,
    split_header_line,
)

#: Banner identifying an encoded comparison record.
COMPARISON_BANNER = "navprobe-comparison/1"

#: Header lines an encoded comparison record occupies.
COMPARISON_HEADER_FIELD_COUNT = 5


def encode_comparison_record(record: ComparisonRecord) -> str:
    """Encode a comparison record to its text form.

    Args:
        record: The record to encode.

    Returns:
        The encoded text, newline-terminated.
    """
    return join_document(
        [
            COMPARISON_BANNER,
            header_line("left_label", record["left_label"]),
            header_line("right_label", record["right_label"]),
            header_line("digests_match", encode_bool(record["digests_match"])),
            header_line(
                "first_divergent_step", encode_optional_int(record["first_divergent_step"])
            ),
            header_line("compared_step_count", str(record["compared_step_count"])),
        ]
    )


def decode_comparison_record(text: str) -> ComparisonRecord:
    """Decode a comparison record from its text form.

    Args:
        text: The encoded record.

    Returns:
        The decoded record.

    Raises:
        WireFormatError: When the banner is absent or belongs to another record
            type, a header field is missing or malformed, or lines trail the
            header.
    """
    header, body = split_document(text, COMPARISON_BANNER, COMPARISON_HEADER_FIELD_COUNT)
    require_no_body(body, COMPARISON_BANNER)
    return ComparisonRecord(
        left_label=require_text_field(split_header_line(header[0], "left_label"), "left_label"),
        right_label=require_text_field(split_header_line(header[1], "right_label"), "right_label"),
        digests_match=require_bool_field(
            split_header_line(header[2], "digests_match"), "digests_match"
        ),
        first_divergent_step=require_optional_non_negative_field(
            split_header_line(header[3], "first_divergent_step"), "first_divergent_step"
        ),
        compared_step_count=require_non_negative_field(
            split_header_line(header[4], "compared_step_count"), "compared_step_count"
        ),
    )


__all__ = [
    "COMPARISON_BANNER",
    "COMPARISON_HEADER_FIELD_COUNT",
    "decode_comparison_record",
    "encode_comparison_record",
]
