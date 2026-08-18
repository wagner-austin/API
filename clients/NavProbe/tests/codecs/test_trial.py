"""Tests for the trial record codec."""

from __future__ import annotations

import pytest

from navprobe.codecs.run import encode_run_record
from navprobe.codecs.trial import (
    TRIAL_BANNER,
    decode_trial_record,
    decode_trial_spec,
    encode_trial_record,
    encode_trial_spec,
)
from navprobe.records import RunRecord, RunSpec, TrialRecord, TrialSpec
from navprobe.wireformat import SEPARATOR, WireFormatError


def _record() -> TrialRecord:
    """Build a valid trial record.

    Returns:
        A record reporting a deterministic three-repetition trial.
    """
    return TrialRecord(
        spec=TrialSpec(seed=7, step_count=4, repetitions=3),
        world_count=2,
        reference_digest="cc",
        deterministic=True,
        first_divergent_step=None,
    )


def _lines() -> list[str]:
    """Encode the reference record and split it into lines.

    Returns:
        The encoded lines with the trailing blank removed.
    """
    return encode_trial_record(_record()).strip("\n").split("\n")


class TestTrialSpecCodec:
    """Tests for :func:`encode_trial_spec` and :func:`decode_trial_spec`."""

    def test_field_order_is_fixed(self) -> None:
        """Spec field order is part of the format."""
        spec = TrialSpec(seed=1, step_count=2, repetitions=3)
        keys = [line.split(SEPARATOR)[0] for line in encode_trial_spec(spec)]
        assert keys == ["seed", "step_count", "repetitions"]

    def test_round_trips(self) -> None:
        """A spec survives encoding and decoding."""
        spec = TrialSpec(seed=1, step_count=2, repetitions=3)
        assert decode_trial_spec(encode_trial_spec(spec)) == spec


class TestEncodeTrialRecord:
    """Tests for :func:`encode_trial_record`."""

    def test_starts_with_the_trial_banner(self) -> None:
        """The banner distinguishes this record from the others."""
        assert _lines()[0] == TRIAL_BANNER

    def test_writes_the_header_in_fixed_order(self) -> None:
        """Header field order is part of the format."""
        assert [line.split(SEPARATOR)[0] for line in _lines()[1:]] == [
            "seed",
            "step_count",
            "repetitions",
            "world_count",
            "reference_digest",
            "deterministic",
            "first_divergent_step",
        ]

    def test_is_byte_stable_for_equal_records(self) -> None:
        """Equal trials encode identically, so results files can be compared."""
        assert encode_trial_record(_record()) == encode_trial_record(_record())


class TestTrialRecordRoundTrip:
    """Encoding and decoding a trial record compose to the identity."""

    def test_round_trips_a_deterministic_trial(self) -> None:
        """A trial with no divergence survives the round trip."""
        assert decode_trial_record(encode_trial_record(_record())) == _record()

    def test_round_trips_a_divergent_trial(self) -> None:
        """A trial with a divergence point survives the round trip."""
        record = TrialRecord(
            spec=TrialSpec(seed=7, step_count=4, repetitions=3),
            world_count=2,
            reference_digest="cc",
            deterministic=False,
            first_divergent_step=2,
        )
        assert decode_trial_record(encode_trial_record(record)) == record

    def test_round_trips_a_divergence_at_step_zero(self) -> None:
        """A trial that diverged immediately is distinct from one that did not."""
        record = TrialRecord(
            spec=TrialSpec(seed=7, step_count=4, repetitions=2),
            world_count=1,
            reference_digest="cc",
            deterministic=False,
            first_divergent_step=0,
        )
        assert decode_trial_record(encode_trial_record(record))["first_divergent_step"] == 0


class TestTrialRecordRejections:
    """Malformed trial records the decoder refuses."""

    def test_rejects_another_record_types_banner(self) -> None:
        """A run record must not decode as a trial record."""
        other = RunRecord(
            spec=RunSpec(label="x", seed=1, step_count=0, world_count=1),
            steps=(),
            digest="cc",
        )
        with pytest.raises(WireFormatError) as caught:
            decode_trial_record(encode_run_record(other))
        assert caught.value.code == "NP-WIRE-009"

    def test_rejects_a_truncated_header(self) -> None:
        """Every header line is required."""
        with pytest.raises(WireFormatError) as caught:
            decode_trial_record(f"{TRIAL_BANNER}\nseed\t7\n")
        assert caught.value.code == "NP-WIRE-010"

    def test_rejects_trailing_lines(self) -> None:
        """A trial declares no rows, so trailing content is refused."""
        text = encode_trial_record(_record()) + "step\t0\taa\n"
        with pytest.raises(WireFormatError) as caught:
            decode_trial_record(text)
        assert caught.value.code == "NP-WIRE-013"

    def test_rejects_zero_repetitions(self) -> None:
        """A trial of no repetitions compared nothing."""
        lines = _lines()
        lines[3] = f"repetitions{SEPARATOR}0"
        with pytest.raises(WireFormatError) as caught:
            decode_trial_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-003"

    def test_rejects_a_zero_world_count(self) -> None:
        """A trial claiming no worlds observed nothing."""
        lines = _lines()
        lines[4] = f"world_count{SEPARATOR}0"
        with pytest.raises(WireFormatError) as caught:
            decode_trial_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-003"

    def test_rejects_an_empty_reference_digest(self) -> None:
        """A trial without a reference digest has nothing to compare against."""
        lines = _lines()
        lines[5] = f"reference_digest{SEPARATOR}"
        with pytest.raises(WireFormatError) as caught:
            decode_trial_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-004"

    def test_rejects_an_unspelled_determinism_verdict(self) -> None:
        """The verdict flag must be one of the two spelled tokens."""
        lines = _lines()
        lines[6] = f"deterministic{SEPARATOR}maybe"
        with pytest.raises(WireFormatError) as caught:
            decode_trial_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-012"

    def test_rejects_a_negative_seed(self) -> None:
        """A seed is drawn from the non-negative integers."""
        lines = _lines()
        lines[1] = f"seed{SEPARATOR}-1"
        with pytest.raises(WireFormatError) as caught:
            decode_trial_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-002"
