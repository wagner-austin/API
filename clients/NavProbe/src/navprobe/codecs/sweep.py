"""Codec for sweep results.

A sweep is the one record with a body of *composite* rows: each row carries a
whole scene and a whole trial verdict. Rather than nesting documents, the row is
flat — the scene's fields followed by the trial's — and each side's fields are
produced and consumed by its own codec.

That keeps the field order in one place per record type. A sweep row that spelled
out scene fields itself would be a second declaration of a scene's layout, free
to drift from the one in :mod:`navprobe.codecs.scene`.
"""

from __future__ import annotations

from navprobe.codecs.scene import SCENE_FIELD_COUNT, decode_scene_fields, scene_fields
from navprobe.records import SweepEntry, TrialRecord, TrialSpec
from navprobe.wireformat import (
    SEPARATOR,
    WireFormatError,
    encode_bool,
    encode_optional_int,
    join_document,
    require_bool_field,
    require_non_negative_field,
    require_optional_non_negative_field,
    require_positive_field,
    require_text_field,
    split_document,
)

#: Banner identifying an encoded sweep.
SWEEP_BANNER = "navprobe-sweep/1"

#: Leading token marking one sweep row.
ENTRY_TAG = "entry"

#: Fields a trial verdict occupies within a sweep row.
TRIAL_FIELD_COUNT = 7

#: Tokens one sweep row carries: the tag, the scene, and the trial.
ENTRY_TOKEN_COUNT = 1 + SCENE_FIELD_COUNT + TRIAL_FIELD_COUNT


def encode_sweep_entry(entry: SweepEntry) -> str:
    """Encode one sweep entry to its row form.

    Split out from :func:`encode_sweep` so a record that *embeds* a sweep --
    one carrying the conditions the sweep ran under -- reuses this row rather
    than restating the field order. Two declarations of a row layout are free
    to drift, and a drifted row would decode into the wrong fields silently.

    Args:
        entry: The entry to encode.

    Returns:
        One tab-separated row, without a trailing newline.
    """
    trial = entry["trial"]
    spec = trial["spec"]
    return SEPARATOR.join(
        (
            ENTRY_TAG,
            *scene_fields(entry["scene"]),
            str(spec["seed"]),
            str(spec["step_count"]),
            str(spec["repetitions"]),
            str(trial["world_count"]),
            trial["reference_digest"],
            encode_bool(trial["deterministic"]),
            encode_optional_int(trial["first_divergent_step"]),
        )
    )


def encode_sweep(entries: tuple[SweepEntry, ...]) -> str:
    """Encode a sweep to its text form.

    Args:
        entries: The sweep's entries, in sweep order.

    Returns:
        The encoded text, newline-terminated.
    """
    return join_document([SWEEP_BANNER, *(encode_sweep_entry(entry) for entry in entries)])


def decode_sweep_entry(line: str, position: int) -> SweepEntry:
    """Decode one sweep row.

    Args:
        line: The line to decode.
        position: The row's position, used in error messages.

    Returns:
        The decoded entry.

    Raises:
        WireFormatError: When the row is malformed or a field is outside its
            range.
    """
    parts = line.split(SEPARATOR)
    if len(parts) != ENTRY_TOKEN_COUNT or parts[0] != ENTRY_TAG:
        raise WireFormatError(
            "NP-WIRE-017",
            f"sweep row {position} must be {ENTRY_TAG!r} followed by "
            f"{ENTRY_TOKEN_COUNT - 1} fields, got {len(parts) - 1}",
        )
    scene = decode_scene_fields(tuple(parts[1 : 1 + SCENE_FIELD_COUNT]), position)
    trial = parts[1 + SCENE_FIELD_COUNT :]
    return SweepEntry(
        scene=scene,
        trial=TrialRecord(
            spec=TrialSpec(
                seed=require_non_negative_field(trial[0], f"entry[{position}].seed"),
                step_count=require_non_negative_field(trial[1], f"entry[{position}].step_count"),
                repetitions=require_positive_field(trial[2], f"entry[{position}].repetitions"),
            ),
            world_count=require_positive_field(trial[3], f"entry[{position}].world_count"),
            reference_digest=require_text_field(trial[4], f"entry[{position}].reference_digest"),
            deterministic=require_bool_field(trial[5], f"entry[{position}].deterministic"),
            first_divergent_step=require_optional_non_negative_field(
                trial[6], f"entry[{position}].first_divergent_step"
            ),
        ),
    )


def decode_sweep(text: str) -> tuple[SweepEntry, ...]:
    """Decode a sweep from its text form.

    Args:
        text: The encoded sweep.

    Returns:
        The decoded entries, in sweep order.

    Raises:
        WireFormatError: When the banner is absent or belongs to another record
            type, or a row is malformed.
    """
    _, body = split_document(text, SWEEP_BANNER, 0)
    return tuple(decode_sweep_entry(line, position) for position, line in enumerate(body))


__all__ = [
    "ENTRY_TAG",
    "ENTRY_TOKEN_COUNT",
    "SWEEP_BANNER",
    "TRIAL_FIELD_COUNT",
    "decode_sweep",
    "decode_sweep_entry",
    "encode_sweep",
    "encode_sweep_entry",
]
