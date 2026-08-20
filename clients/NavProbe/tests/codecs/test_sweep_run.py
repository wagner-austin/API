"""Tests for the sweep-run record codec."""

from __future__ import annotations

import pytest

from navprobe.codecs.sweep import encode_sweep
from navprobe.codecs.sweep_run import (
    SWEEP_RUN_BANNER,
    decode_sweep_run,
    encode_sweep_run,
)
from navprobe.records import (
    SceneSpec,
    SweepEntry,
    SweepRunRecord,
    TrialRecord,
    TrialSpec,
)
from navprobe.wireformat import SEPARATOR, WireFormatError


def _entry(body_count: int, deterministic: bool, divergent: int | None) -> SweepEntry:
    """Build one sweep entry.

    Args:
        body_count: Bodies in the scene.
        deterministic: The verdict.
        divergent: Where repetitions parted, if they did.

    Returns:
        The entry.
    """
    return SweepEntry(
        scene=SceneSpec(
            body_count=body_count,
            lattice_width=body_count,
            spacing=0.055,
            radius=0.03,
            timestep=0.005,
        ),
        trial=TrialRecord(
            spec=TrialSpec(seed=7, step_count=150, repetitions=12),
            world_count=2,
            reference_digest="a" * 64,
            deterministic=deterministic,
            first_divergent_step=divergent,
        ),
    )


def _record() -> SweepRunRecord:
    """Build a sweep run.

    Returns:
        A record carrying both verdicts, so a codec that dropped either would
        fail the round trip.
    """
    return SweepRunRecord(
        mode="RUN_TO_RUN",
        device="NVIDIA GeForce RTX 3090 Ti",
        device_request="cuda:0",
        max_records=64,
        world_count=2,
        perturbation=0.01,
        constraint_capacity=8192,
        entries=(_entry(4, True, None), _entry(6, False, 57)),
    )


class TestEncodeSweepRun:
    """Tests for :func:`encode_sweep_run`."""

    def test_starts_with_the_sweep_run_banner(self) -> None:
        """The banner separates it from the bare sweep it embeds."""
        assert encode_sweep_run(_record()).split("\n")[0] == SWEEP_RUN_BANNER

    def test_is_byte_stable_for_equal_records(self) -> None:
        """Equal records encode identically, so two runs can be diffed."""
        assert encode_sweep_run(_record()) == encode_sweep_run(_record())

    def test_reuses_the_sweep_codec_rows(self) -> None:
        """The row layout has exactly one declaration.

        Asserted against the sweep codec's own output rather than a literal, so
        a change to the row that forgot this record would fail here.
        """
        embedded = encode_sweep(_record()["entries"]).split("\n")[1:]
        assert encode_sweep_run(_record()).split("\n")[8:] == embedded


class TestSweepRunRoundTrip:
    """Encoding and decoding compose to the identity."""

    def test_round_trips(self) -> None:
        """A record survives encoding and decoding exactly."""
        assert decode_sweep_run(encode_sweep_run(_record())) == _record()

    def test_round_trips_an_empty_sweep(self) -> None:
        """A sweep whose family was empty is a construction bug, not a crash.

        The conditions still decode, which is what lets the caller see that no
        scene ran rather than seeing a malformed file.
        """
        record = SweepRunRecord(
            mode="NOT_GUARANTEED",
            device="cpu",
            device_request="cpu",
            max_records=0,
            world_count=1,
            perturbation=0.0,
            constraint_capacity=1,
            entries=(),
        )
        assert decode_sweep_run(encode_sweep_run(record)) == record

    def test_preserves_the_divergence_point(self) -> None:
        """The step a sweep parted at is the finding, not a detail."""
        decoded = decode_sweep_run(encode_sweep_run(_record()))
        assert decoded["entries"][1]["trial"]["first_divergent_step"] == 57

    def test_preserves_a_perturbation_with_no_exact_decimal_form(self) -> None:
        """0.01 is not exactly representable; the scene must still match."""
        assert decode_sweep_run(encode_sweep_run(_record()))["perturbation"] == 0.01


class TestSweepRunRejections:
    """Malformed sweep runs the decoder refuses."""

    def test_rejects_a_bare_sweep(self) -> None:
        """A sweep without conditions must not decode as a sweep run.

        This is the whole reason the record exists: verdicts without the card
        and the mode cannot be compared against another run.
        """
        with pytest.raises(WireFormatError) as caught:
            decode_sweep_run(encode_sweep(_record()["entries"]))
        assert caught.value.code == "NP-WIRE-009"

    def test_rejects_a_truncated_header(self) -> None:
        """Every condition is required."""
        with pytest.raises(WireFormatError) as caught:
            decode_sweep_run(f"{SWEEP_RUN_BANNER}\nmode\tRUN_TO_RUN\n")
        assert caught.value.code == "NP-WIRE-010"

    def test_rejects_a_zero_world_count(self) -> None:
        """A sweep over no worlds cannot have happened."""
        lines = encode_sweep_run(_record()).strip("\n").split("\n")
        lines[5] = f"world_count{SEPARATOR}0"
        with pytest.raises(WireFormatError) as caught:
            decode_sweep_run("\n".join(lines) + "\n")
        assert caught.value.code == "NP-WIRE-003"

    def test_rejects_a_malformed_entry_row(self) -> None:
        """A short row would otherwise decode into the wrong fields."""
        lines = encode_sweep_run(_record()).strip("\n").split("\n")
        lines[8] = f"entry{SEPARATOR}2"
        with pytest.raises(WireFormatError) as caught:
            decode_sweep_run("\n".join(lines) + "\n")
        assert caught.value.code == "NP-WIRE-017"
