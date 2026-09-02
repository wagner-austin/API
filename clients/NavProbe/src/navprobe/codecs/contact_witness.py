"""Codec for a collision-pair sweep whose verdicts carry a liveness witness.

The record this package's convex-narrowphase finding is stated from. A verdict
alone cannot distinguish a scene that reproduced from one that reproduced
because it stopped interacting, so every row here carries the contact counts
beside the verdict rather than in a separate document that could be lost,
mismatched, or quietly not written.

The row is flat -- the pair, then the trial's fields, then the witness -- and
the trial half is spelled in the same order :mod:`navprobe.codecs.sweep` uses.
It is spelled out rather than imported because a sweep row leads with a whole
:class:`navprobe.records.SceneSpec` and this one leads with a pair name, so
there is no shared prefix to reuse; what is shared is the order, and the field
count constant below is what a test can hold the two to.
"""

from __future__ import annotations

from navprobe.codecs.device_conditions import (
    DEVICE_CONDITIONS_FIELD_COUNT,
    decode_device_conditions,
    encode_device_conditions,
)
from navprobe.records import ContactWitnessEntry, ContactWitnessRunRecord, TrialRecord, TrialSpec
from navprobe.wireformat import (
    SEPARATOR,
    WireFormatError,
    encode_bool,
    encode_float_field,
    encode_optional_int,
    header_line,
    join_document,
    require_bool_field,
    require_non_negative_field,
    require_non_negative_float_field,
    require_optional_non_negative_field,
    require_positive_field,
    require_text_field,
    split_document,
    split_header_line,
)

#: Banner identifying an encoded contact-witness run.
CONTACT_WITNESS_BANNER = "navprobe-contact-witness/1"

#: Leading token marking one row.
ENTRY_TAG = "pair"

#: Fields a trial verdict occupies within a row. Same order as
#: :data:`navprobe.codecs.sweep.TRIAL_FIELD_COUNT` covers, and a test holds the
#: two constants equal so the shared order cannot drift unnoticed.
TRIAL_FIELD_COUNT = 7

#: Fields the witness occupies: the contact total and the zero-contact count.
WITNESS_FIELD_COUNT = 2

#: Tokens one row carries: the tag, the pair name, the trial, the witness.
ENTRY_TOKEN_COUNT = 1 + 1 + TRIAL_FIELD_COUNT + WITNESS_FIELD_COUNT

#: Header lines an encoded run occupies.
CONTACT_WITNESS_HEADER_FIELD_COUNT = DEVICE_CONDITIONS_FIELD_COUNT + 3


def encode_contact_witness_entry(entry: ContactWitnessEntry) -> str:
    """Encode one entry to its row form.

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
            entry["pair"],
            str(spec["seed"]),
            str(spec["step_count"]),
            str(spec["repetitions"]),
            str(trial["world_count"]),
            trial["reference_digest"],
            encode_bool(trial["deterministic"]),
            encode_optional_int(trial["first_divergent_step"]),
            str(entry["contact_total"]),
            str(entry["zero_contact_steps"]),
        )
    )


def decode_contact_witness_entry(line: str, position: int) -> ContactWitnessEntry:
    """Decode one entry from its row form.

    Args:
        line: The row.
        position: The row's index, named in any error so a bad row in a long
            document can be found without counting.

    Returns:
        The decoded entry.

    Raises:
        WireFormatError: When the row is malformed or a field is outside its
            range.
    """
    parts = line.split(SEPARATOR)
    if len(parts) != ENTRY_TOKEN_COUNT or parts[0] != ENTRY_TAG:
        raise WireFormatError(
            "NP-WIRE-022",
            f"contact-witness row {position} must be {ENTRY_TAG!r} followed by "
            f"{ENTRY_TOKEN_COUNT - 1} fields, got {len(parts) - 1}",
        )
    trial = parts[2 : 2 + TRIAL_FIELD_COUNT]
    witness = parts[2 + TRIAL_FIELD_COUNT :]
    return ContactWitnessEntry(
        pair=require_text_field(parts[1], f"entry[{position}].pair"),
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
        contact_total=require_non_negative_field(witness[0], f"entry[{position}].contact_total"),
        zero_contact_steps=require_non_negative_field(
            witness[1], f"entry[{position}].zero_contact_steps"
        ),
    )


def encode_contact_witness_run(record: ContactWitnessRunRecord) -> str:
    """Encode a contact-witness run to its text form.

    Args:
        record: The record to encode.

    Returns:
        The encoded text, newline-terminated.
    """
    return join_document(
        [
            CONTACT_WITNESS_BANNER,
            *encode_device_conditions(record),
            header_line("world_count", str(record["world_count"])),
            header_line("perturbation", encode_float_field(record["perturbation"])),
            header_line("constraint_capacity", str(record["constraint_capacity"])),
            *(encode_contact_witness_entry(entry) for entry in record["entries"]),
        ]
    )


def decode_contact_witness_run(text: str) -> ContactWitnessRunRecord:
    """Decode a contact-witness run from its text form.

    Args:
        text: The encoded record.

    Returns:
        The decoded record.

    Raises:
        WireFormatError: When the banner is absent or belongs to another record
            type, a header field is missing or malformed, or a row is
            malformed.
    """
    header, body = split_document(text, CONTACT_WITNESS_BANNER, CONTACT_WITNESS_HEADER_FIELD_COUNT)
    conditions = decode_device_conditions(header[:DEVICE_CONDITIONS_FIELD_COUNT])
    return ContactWitnessRunRecord(
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
        entries=tuple(
            decode_contact_witness_entry(line, position) for position, line in enumerate(body)
        ),
    )


__all__ = [
    "CONTACT_WITNESS_BANNER",
    "CONTACT_WITNESS_HEADER_FIELD_COUNT",
    "ENTRY_TAG",
    "ENTRY_TOKEN_COUNT",
    "TRIAL_FIELD_COUNT",
    "WITNESS_FIELD_COUNT",
    "decode_contact_witness_entry",
    "decode_contact_witness_run",
    "encode_contact_witness_entry",
    "encode_contact_witness_run",
]
