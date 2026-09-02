"""Tests for the contact-witness record codec.

The record exists so a verdict cannot be read without its witness, so what
these check hardest is that the witness survives the round trip and that a
document missing it is refused rather than decoded with the fields shifted.
"""

from __future__ import annotations

import pytest

from navprobe.codecs.contact_witness import (
    CONTACT_WITNESS_BANNER,
    ENTRY_TAG,
    ENTRY_TOKEN_COUNT,
    TRIAL_FIELD_COUNT,
    decode_contact_witness_entry,
    decode_contact_witness_run,
    encode_contact_witness_entry,
    encode_contact_witness_run,
)
from navprobe.codecs.sweep import TRIAL_FIELD_COUNT as SWEEP_TRIAL_FIELD_COUNT
from navprobe.records import (
    ContactWitnessEntry,
    ContactWitnessRunRecord,
    TrialRecord,
    TrialSpec,
)
from navprobe.wireformat import SEPARATOR, WireFormatError


def _entry(
    pair: str = "box_box",
    deterministic: bool = True,
    divergent: int | None = None,
    contact_total: int = 320,
    zero_contact_steps: int = 0,
) -> ContactWitnessEntry:
    """Build one entry.

    Args:
        pair: The collision pair.
        deterministic: The verdict.
        divergent: Where repetitions parted, if they did.
        contact_total: Contacts summed over the witness rollout.
        zero_contact_steps: Steps that reported no contact.

    Returns:
        The entry.
    """
    return ContactWitnessEntry(
        pair=pair,
        trial=TrialRecord(
            spec=TrialSpec(seed=7, step_count=40, repetitions=4),
            world_count=2,
            reference_digest="a" * 64,
            deterministic=deterministic,
            first_divergent_step=divergent,
        ),
        contact_total=contact_total,
        zero_contact_steps=zero_contact_steps,
    )


def _record(*entries: ContactWitnessEntry) -> ContactWitnessRunRecord:
    """Build a run record.

    Args:
        entries: Entries to carry, defaulting to one.

    Returns:
        The record.
    """
    return ContactWitnessRunRecord(
        mode="RUN_TO_RUN",
        device="NVIDIA GeForce RTX 3090 Ti",
        device_request="cuda:0",
        max_records=4096,
        linesearch_block_dim=None,
        world_count=2,
        perturbation=0.01,
        constraint_capacity=4096,
        entries=entries or (_entry(),),
    )


class TestRoundTrip:
    """What survives encoding and decoding."""

    def test_a_record_round_trips_exactly(self) -> None:
        """Every field returns as it went in."""
        record = _record()
        assert decode_contact_witness_run(encode_contact_witness_run(record)) == record

    def test_the_witness_survives(self) -> None:
        """The counts are carried, not recomputed or dropped."""
        record = _record(_entry(contact_total=0, zero_contact_steps=40))
        decoded = decode_contact_witness_run(encode_contact_witness_run(record))
        entry = decoded["entries"][0]
        assert (entry["contact_total"], entry["zero_contact_steps"]) == (0, 40)

    def test_a_divergent_verdict_round_trips(self) -> None:
        """The optional divergence step survives when it is present."""
        record = _record(_entry(deterministic=False, divergent=17))
        decoded = decode_contact_witness_run(encode_contact_witness_run(record))
        assert decoded["entries"][0]["trial"]["first_divergent_step"] == 17

    def test_every_pair_keeps_its_own_row(self) -> None:
        """Rows stay in order and keep their own witness."""
        record = _record(
            _entry(pair="sphere_plane", contact_total=80),
            _entry(pair="box_box", contact_total=0, zero_contact_steps=40),
        )
        decoded = decode_contact_witness_run(encode_contact_witness_run(record))
        assert [(e["pair"], e["contact_total"]) for e in decoded["entries"]] == [
            ("sphere_plane", 80),
            ("box_box", 0),
        ]


class TestRecordShape:
    """The document's own structure."""

    def test_the_banner_leads_the_document(self) -> None:
        """A decoder can identify the record type from the first line."""
        assert encode_contact_witness_run(_record()).splitlines()[0] == CONTACT_WITNESS_BANNER

    def test_a_row_carries_the_declared_token_count(self) -> None:
        """The row layout is the one the constant declares."""
        row = encode_contact_witness_entry(_entry())
        assert len(row.split(SEPARATOR)) == ENTRY_TOKEN_COUNT

    def test_the_trial_half_matches_the_sweep_row_s_width(self) -> None:
        """Both records spell a verdict with the same number of fields.

        The two row layouts are declared separately because they lead with
        different things. Holding the shared half equal here is what stops one
        from gaining a field the other does not.
        """
        assert TRIAL_FIELD_COUNT == SWEEP_TRIAL_FIELD_COUNT


class TestRejects:
    """Documents a decoder must refuse rather than misread."""

    def test_rejects_another_record_s_banner(self) -> None:
        """A sweep document does not decode as a contact-witness one."""
        text = encode_contact_witness_run(_record()).replace(
            CONTACT_WITNESS_BANNER, "navprobe-sweep-run/2", 1
        )
        with pytest.raises(WireFormatError):
            decode_contact_witness_run(text)

    def test_rejects_a_row_missing_the_witness(self) -> None:
        """A row without its counts is refused, not decoded short.

        This is the case the record exists to prevent: a verdict arriving
        without the number that says whether it meant anything.
        """
        row = encode_contact_witness_entry(_entry())
        truncated = SEPARATOR.join(row.split(SEPARATOR)[:-2])
        with pytest.raises(WireFormatError) as caught:
            decode_contact_witness_entry(truncated, 0)
        assert caught.value.code == "NP-WIRE-022"

    def test_rejects_a_row_with_the_wrong_tag(self) -> None:
        """A row that is not tagged as a pair is refused."""
        row = encode_contact_witness_entry(_entry()).replace(ENTRY_TAG, "entry", 1)
        with pytest.raises(WireFormatError) as caught:
            decode_contact_witness_entry(row, 3)
        assert caught.value.code == "NP-WIRE-022"

    def test_names_the_row_that_failed(self) -> None:
        """The error carries the row index so a long document is searchable."""
        with pytest.raises(WireFormatError) as caught:
            decode_contact_witness_entry("pair\tbox_box", 7)
        assert "row 7" in str(caught.value)

    def test_rejects_a_negative_contact_total(self) -> None:
        """A count cannot be negative, and the field is named when it is."""
        parts = encode_contact_witness_entry(_entry()).split(SEPARATOR)
        parts[-2] = "-1"
        with pytest.raises(WireFormatError) as caught:
            decode_contact_witness_entry(SEPARATOR.join(parts), 0)
        assert "contact_total" in str(caught.value)

    def test_rejects_a_negative_zero_contact_count(self) -> None:
        """The same holds for the zero-contact step count."""
        parts = encode_contact_witness_entry(_entry()).split(SEPARATOR)
        parts[-1] = "-4"
        with pytest.raises(WireFormatError) as caught:
            decode_contact_witness_entry(SEPARATOR.join(parts), 0)
        assert "zero_contact_steps" in str(caught.value)
