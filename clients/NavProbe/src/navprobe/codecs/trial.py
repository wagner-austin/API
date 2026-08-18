"""Codec for trial records and the spec they are built from.

A trial record is the row that goes in a results table: the design, the batch
width, the reference digest, and the verdict. It carries no body, and the
per-repetition run records it summarises are persisted separately rather than
nested, so the same bytes are never stored twice.
"""

from __future__ import annotations

from navprobe.records import TrialRecord, TrialSpec
from navprobe.wireformat import (
    encode_bool,
    encode_optional_int,
    header_line,
    join_document,
    require_bool_field,
    require_no_body,
    require_non_negative_field,
    require_optional_non_negative_field,
    require_positive_field,
    require_text_field,
    split_document,
    split_header_line,
)

#: Banner identifying an encoded trial record.
TRIAL_BANNER = "navprobe-trial/1"

#: Header lines a :class:`navprobe.records.TrialSpec` occupies.
TRIAL_SPEC_FIELD_COUNT = 3

#: Header lines an encoded trial record occupies.
TRIAL_HEADER_FIELD_COUNT = TRIAL_SPEC_FIELD_COUNT + 4


def encode_trial_spec(spec: TrialSpec) -> tuple[str, ...]:
    """Encode a trial spec to its header lines.

    Args:
        spec: The spec to encode.

    Returns:
        Exactly :data:`TRIAL_SPEC_FIELD_COUNT` header lines in fixed order.
    """
    return (
        header_line("seed", str(spec["seed"])),
        header_line("step_count", str(spec["step_count"])),
        header_line("repetitions", str(spec["repetitions"])),
    )


def decode_trial_spec(lines: tuple[str, ...]) -> TrialSpec:
    """Decode a trial spec from its header lines.

    ``repetitions`` is required to be positive here and is separately required
    to be at least two by :mod:`navprobe.experiment`. The codec enforces what
    the format can represent; the experiment layer enforces what is
    scientifically meaningful.

    Args:
        lines: Exactly :data:`TRIAL_SPEC_FIELD_COUNT` header lines, as sliced by
            :func:`navprobe.wireformat.split_document`, which has already
            guaranteed the count.

    Returns:
        The decoded spec.

    Raises:
        WireFormatError: When a line is malformed, carries the wrong key, or
            holds a value outside its field's range.
    """
    return TrialSpec(
        seed=require_non_negative_field(split_header_line(lines[0], "seed"), "seed"),
        step_count=require_non_negative_field(
            split_header_line(lines[1], "step_count"), "step_count"
        ),
        repetitions=require_positive_field(
            split_header_line(lines[2], "repetitions"), "repetitions"
        ),
    )


def encode_trial_record(record: TrialRecord) -> str:
    """Encode a trial record to its text form.

    Args:
        record: The record to encode.

    Returns:
        The encoded text, newline-terminated.
    """
    return join_document(
        [
            TRIAL_BANNER,
            *encode_trial_spec(record["spec"]),
            header_line("world_count", str(record["world_count"])),
            header_line("reference_digest", record["reference_digest"]),
            header_line("deterministic", encode_bool(record["deterministic"])),
            header_line(
                "first_divergent_step", encode_optional_int(record["first_divergent_step"])
            ),
        ]
    )


def decode_trial_record(text: str) -> TrialRecord:
    """Decode a trial record from its text form.

    Args:
        text: The encoded record.

    Returns:
        The decoded record.

    Raises:
        WireFormatError: When the banner is absent or belongs to another record
            type, a header field is missing or malformed, or lines trail the
            header.
    """
    header, body = split_document(text, TRIAL_BANNER, TRIAL_HEADER_FIELD_COUNT)
    require_no_body(body, TRIAL_BANNER)
    return TrialRecord(
        spec=decode_trial_spec(header[:TRIAL_SPEC_FIELD_COUNT]),
        world_count=require_positive_field(
            split_header_line(header[3], "world_count"), "world_count"
        ),
        reference_digest=require_text_field(
            split_header_line(header[4], "reference_digest"), "reference_digest"
        ),
        deterministic=require_bool_field(
            split_header_line(header[5], "deterministic"), "deterministic"
        ),
        first_divergent_step=require_optional_non_negative_field(
            split_header_line(header[6], "first_divergent_step"), "first_divergent_step"
        ),
    )


__all__ = [
    "TRIAL_BANNER",
    "TRIAL_HEADER_FIELD_COUNT",
    "TRIAL_SPEC_FIELD_COUNT",
    "decode_trial_record",
    "decode_trial_spec",
    "encode_trial_record",
    "encode_trial_spec",
]
