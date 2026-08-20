"""Codec for a world-count ladder together with the conditions it ran under.

Each rung is a row: the world count, its verdict, and what it cost. The scene
and the trial design are header fields rather than per-row ones, because a
ladder holds both fixed -- that is what makes the world count the only variable
and the cost curve readable as a curve.

Throughput is stored rather than derived on read. It is a function of the wall
time and the trial design, so a reader could recompute it, but storing it means
the number in the record is the number that was reported, and a later change to
the derivation cannot silently restate a published figure.
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
from navprobe.codecs.trial import (
    TRIAL_SPEC_FIELD_COUNT,
    decode_trial_spec,
    encode_trial_spec,
)
from navprobe.records import ScalingRungRecord, ScalingRunRecord
from navprobe.wireformat import (
    SEPARATOR,
    WireFormatError,
    encode_bool,
    encode_float_field,
    encode_optional_int,
    header_line,
    join_document,
    require_bool_field,
    require_non_negative_float_field,
    require_optional_non_negative_field,
    require_positive_field,
    require_positive_float_field,
    require_text_field,
    split_document,
    split_header_line,
)

#: Banner identifying an encoded scaling run.
SCALING_RUN_BANNER = "navprobe-scaling-run/1"

#: Leading token marking one rung.
RUNG_TAG = "rung"

#: Tokens one rung row carries, including its tag.
RUNG_TOKEN_COUNT = 7

#: Header lines an encoded scaling run occupies.
SCALING_RUN_HEADER_FIELD_COUNT = (
    DEVICE_CONDITIONS_FIELD_COUNT + 1 + SCENE_FIELD_COUNT + TRIAL_SPEC_FIELD_COUNT + 1
)


def encode_rung(rung: ScalingRungRecord) -> str:
    """Encode one rung to its row form.

    Args:
        rung: The rung to encode.

    Returns:
        One tab-separated row, without a trailing newline.
    """
    return SEPARATOR.join(
        (
            RUNG_TAG,
            str(rung["world_count"]),
            rung["reference_digest"],
            encode_bool(rung["deterministic"]),
            encode_optional_int(rung["first_divergent_step"]),
            encode_float_field(rung["wall_seconds"]),
            encode_float_field(rung["world_steps_per_second"]),
        )
    )


def decode_rung(line: str, position: int) -> ScalingRungRecord:
    """Decode one rung row.

    Args:
        line: The line to decode.
        position: The row's position, used in error messages.

    Returns:
        The decoded rung.

    Raises:
        WireFormatError: When the row is malformed or a field is outside its
            range.
    """
    parts = line.split(SEPARATOR)
    if len(parts) != RUNG_TOKEN_COUNT or parts[0] != RUNG_TAG:
        raise WireFormatError(
            "NP-WIRE-021",
            f"scaling row {position} must be {RUNG_TAG!r} followed by "
            f"{RUNG_TOKEN_COUNT - 1} fields, got {len(parts) - 1}",
        )
    return ScalingRungRecord(
        world_count=require_positive_field(parts[1], f"rung[{position}].world_count"),
        reference_digest=require_text_field(parts[2], f"rung[{position}].reference_digest"),
        deterministic=require_bool_field(parts[3], f"rung[{position}].deterministic"),
        first_divergent_step=require_optional_non_negative_field(
            parts[4], f"rung[{position}].first_divergent_step"
        ),
        wall_seconds=require_non_negative_float_field(parts[5], f"rung[{position}].wall_seconds"),
        world_steps_per_second=require_non_negative_float_field(
            parts[6], f"rung[{position}].world_steps_per_second"
        ),
    )


def encode_scaling_run(record: ScalingRunRecord) -> str:
    """Encode a scaling run to its text form.

    Args:
        record: The record to encode.

    Returns:
        The encoded text, newline-terminated.
    """
    return join_document(
        [
            SCALING_RUN_BANNER,
            *encode_device_conditions(record),
            header_line("capacity", str(record["capacity"])),
            *encode_scene_headers(record["scene"]),
            *encode_trial_spec(record["spec"]),
            header_line("perturbation", encode_float_field(record["perturbation"])),
            *(encode_rung(rung) for rung in record["rungs"]),
        ]
    )


def decode_scaling_run(text: str) -> ScalingRunRecord:
    """Decode a scaling run from its text form.

    Args:
        text: The encoded record.

    Returns:
        The decoded record.

    Raises:
        WireFormatError: When the banner is absent or belongs to another record
            type, a header field is missing or malformed, or a rung row is
            malformed.
    """
    header, body = split_document(text, SCALING_RUN_BANNER, SCALING_RUN_HEADER_FIELD_COUNT)
    conditions = decode_device_conditions(header[:DEVICE_CONDITIONS_FIELD_COUNT])
    capacity_at = DEVICE_CONDITIONS_FIELD_COUNT
    scene_at = capacity_at + 1
    spec_at = scene_at + SCENE_FIELD_COUNT
    perturbation_at = spec_at + TRIAL_SPEC_FIELD_COUNT
    return ScalingRunRecord(
        mode=conditions["mode"],
        device=conditions["device"],
        device_request=conditions["device_request"],
        max_records=conditions["max_records"],
        capacity=require_positive_field(
            split_header_line(header[capacity_at], "capacity"), "capacity"
        ),
        scene=decode_scene_headers(header[scene_at : scene_at + SCENE_FIELD_COUNT]),
        spec=decode_trial_spec(header[spec_at : spec_at + TRIAL_SPEC_FIELD_COUNT]),
        perturbation=require_positive_float_field(
            split_header_line(header[perturbation_at], "perturbation"), "perturbation"
        ),
        rungs=tuple(decode_rung(line, position) for position, line in enumerate(body)),
    )


__all__ = [
    "RUNG_TAG",
    "RUNG_TOKEN_COUNT",
    "SCALING_RUN_BANNER",
    "SCALING_RUN_HEADER_FIELD_COUNT",
    "decode_rung",
    "decode_scaling_run",
    "encode_rung",
    "encode_scaling_run",
]
