"""Codec for run records and the parts they are built from.

A run record is the only probe record with a body: one tagged row per step. The
rows carry their own index and are required to be contiguous and in order, so a
truncated or reordered file cannot pass as a complete rollout.
"""

from __future__ import annotations

from navprobe.records import RunRecord, RunSpec, StepRecord
from navprobe.wireformat import (
    SEPARATOR,
    WireFormatError,
    header_line,
    join_document,
    require_non_negative_field,
    require_positive_field,
    require_text_field,
    split_document,
    split_header_line,
)

#: Banner identifying an encoded run record.
RUN_BANNER = "navprobe-run/1"

#: Leading token marking a per-step row.
STEP_TAG = "step"

#: Header lines a :class:`navprobe.records.RunSpec` occupies.
RUN_SPEC_FIELD_COUNT = 4

#: Header lines an encoded run record occupies, before its step rows.
RUN_HEADER_FIELD_COUNT = RUN_SPEC_FIELD_COUNT + 1


def encode_run_spec(spec: RunSpec) -> tuple[str, ...]:
    """Encode a run spec to its header lines.

    Args:
        spec: The spec to encode.

    Returns:
        Exactly :data:`RUN_SPEC_FIELD_COUNT` header lines in fixed order.
    """
    return (
        header_line("label", spec["label"]),
        header_line("seed", str(spec["seed"])),
        header_line("step_count", str(spec["step_count"])),
        header_line("world_count", str(spec["world_count"])),
    )


def decode_run_spec(lines: tuple[str, ...]) -> RunSpec:
    """Decode a run spec from its header lines.

    Args:
        lines: Exactly :data:`RUN_SPEC_FIELD_COUNT` header lines, as sliced by
            :func:`navprobe.wireformat.split_document`, which has already
            guaranteed the count.

    Returns:
        The decoded spec.

    Raises:
        WireFormatError: When a line is malformed, carries the wrong key, or
            holds a value outside its field's range.
    """
    return RunSpec(
        label=require_text_field(split_header_line(lines[0], "label"), "label"),
        seed=require_non_negative_field(split_header_line(lines[1], "seed"), "seed"),
        step_count=require_non_negative_field(
            split_header_line(lines[2], "step_count"), "step_count"
        ),
        world_count=require_positive_field(
            split_header_line(lines[3], "world_count"), "world_count"
        ),
    )


def encode_step_record(step: StepRecord) -> str:
    """Encode one step to its tagged row.

    Args:
        step: The step to encode.

    Returns:
        The row, without a trailing newline.
    """
    return f"{STEP_TAG}{SEPARATOR}{step['step_index']}{SEPARATOR}{step['digest']}"


def decode_step_record(line: str, position: int) -> StepRecord:
    """Decode one step row.

    Args:
        line: The line to decode.
        position: The row's position in the body, which its declared index must
            match.

    Returns:
        The decoded step.

    Raises:
        WireFormatError: When the row is malformed, or declares an index other
            than its position. Steps must be contiguous and in order, so a
            truncated or reordered record cannot pass as complete.
    """
    parts = line.split(SEPARATOR)
    if len(parts) != 3 or parts[0] != STEP_TAG:
        raise WireFormatError(
            "NP-WIRE-007",
            f"step row {position} must be {STEP_TAG!r} followed by an index and a digest, "
            f"got {line!r}",
        )
    step_index = require_non_negative_field(parts[1], f"step[{position}].step_index")
    if step_index != position:
        raise WireFormatError(
            "NP-WIRE-008",
            f"step row {position} declares index {step_index}; "
            "steps must be contiguous and in order",
        )
    return StepRecord(
        step_index=step_index,
        digest=require_text_field(parts[2], f"step[{position}].digest"),
    )


def encode_run_record(record: RunRecord) -> str:
    """Encode a run record to its text form.

    Field order is fixed, so two records with equal content encode to
    byte-identical text and file comparison is a usable second opinion on the
    digest.

    Args:
        record: The record to encode.

    Returns:
        The encoded text, newline-terminated.
    """
    return join_document(
        [
            RUN_BANNER,
            *encode_run_spec(record["spec"]),
            header_line("digest", record["digest"]),
            *(encode_step_record(step) for step in record["steps"]),
        ]
    )


def decode_run_record(text: str) -> RunRecord:
    """Decode a run record from its text form.

    The step count declared in the header is checked against the number of rows
    actually present. A record whose header and body disagree describes no real
    rollout, and accepting it would let a truncated file pass as complete.

    Args:
        text: The encoded record.

    Returns:
        The decoded record.

    Raises:
        WireFormatError: When the banner is absent or belongs to another record
            type, a header field is missing or malformed, a step row is
            malformed, or the declared step count disagrees with the rows
            present.
    """
    header, body = split_document(text, RUN_BANNER, RUN_HEADER_FIELD_COUNT)
    spec = decode_run_spec(header[:RUN_SPEC_FIELD_COUNT])
    digest = require_text_field(split_header_line(header[4], "digest"), "digest")
    if len(body) != spec["step_count"]:
        raise WireFormatError(
            "NP-WIRE-011",
            f"record declares {spec['step_count']} steps but carries {len(body)}",
        )
    steps = tuple(decode_step_record(line, position) for position, line in enumerate(body))
    return RunRecord(spec=spec, steps=steps, digest=digest)


__all__ = [
    "RUN_BANNER",
    "RUN_HEADER_FIELD_COUNT",
    "RUN_SPEC_FIELD_COUNT",
    "STEP_TAG",
    "decode_run_record",
    "decode_run_spec",
    "decode_step_record",
    "encode_run_record",
    "encode_run_spec",
    "encode_step_record",
]
