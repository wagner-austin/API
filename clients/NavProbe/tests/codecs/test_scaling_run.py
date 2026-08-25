"""Tests for the scaling-run record codec."""

from __future__ import annotations

import pytest

from navprobe.codecs.scaling_run import (
    RUNG_TAG,
    SCALING_RUN_BANNER,
    decode_rung,
    decode_scaling_run,
    encode_rung,
    encode_scaling_run,
)
from navprobe.records import (
    ScalingRungRecord,
    ScalingRunRecord,
    SceneSpec,
    TrialSpec,
)
from navprobe.wireformat import SEPARATOR, WireFormatError


def _rung(world_count: int, deterministic: bool) -> ScalingRungRecord:
    """Build one rung.

    Args:
        world_count: Parallel worlds the rung ran.
        deterministic: The verdict.

    Returns:
        The rung, with a throughput that has no exact decimal form.
    """
    return ScalingRungRecord(
        world_count=world_count,
        reference_digest="b" * 64,
        deterministic=deterministic,
        first_divergent_step=None if deterministic else 12,
        wall_seconds=12.634,
        world_steps_per_second=14023.7,
    )


def _record() -> ScalingRunRecord:
    """Build a scaling run.

    Returns:
        The four-rung ladder shape the published cost curve was measured on.
    """
    return ScalingRunRecord(
        mode="RUN_TO_RUN",
        device="NVIDIA GeForce RTX 3090 Ti",
        device_request="cuda:0",
        max_records=64,
        linesearch_block_dim=64,
        capacity=256,
        scene=SceneSpec(body_count=8, lattice_width=8, spacing=0.055, radius=0.03, timestep=0.005),
        spec=TrialSpec(seed=7, step_count=150, repetitions=12),
        perturbation=0.01,
        rungs=(_rung(2, True), _rung(64, True), _rung(512, True), _rung(4096, False)),
    )


class TestEncodeRung:
    """Tests for :func:`encode_rung`."""

    def test_starts_with_the_rung_tag(self) -> None:
        """The tag separates a rung from a header line."""
        assert encode_rung(_rung(2, True)).split(SEPARATOR)[0] == RUNG_TAG

    def test_round_trips_through_decode(self) -> None:
        """A rung survives its row form exactly."""
        assert decode_rung(encode_rung(_rung(64, True)), 0) == _rung(64, True)

    def test_preserves_throughput_exactly(self) -> None:
        """A throughput rounded on the way to disk is a different figure."""
        assert decode_rung(encode_rung(_rung(2, True)), 0)["world_steps_per_second"] == 14023.7


class TestDecodeRungRejections:
    """Malformed rung rows the decoder refuses."""

    def test_rejects_a_short_row(self) -> None:
        """A short row would decode into the wrong fields."""
        with pytest.raises(WireFormatError) as caught:
            decode_rung(f"{RUNG_TAG}{SEPARATOR}2", 0)
        assert caught.value.code == "NP-WIRE-021"

    def test_rejects_a_row_with_another_tag(self) -> None:
        """A sweep entry must not decode as a rung."""
        with pytest.raises(WireFormatError) as caught:
            decode_rung(SEPARATOR.join(["entry"] + ["x"] * 6), 0)
        assert caught.value.code == "NP-WIRE-021"

    def test_names_the_row_position(self) -> None:
        """The message locates the bad row in a long ladder."""
        with pytest.raises(WireFormatError) as caught:
            decode_rung(f"{RUNG_TAG}{SEPARATOR}2", 3)
        assert "row 3" in caught.value.message

    def test_rejects_a_zero_world_count(self) -> None:
        """A rung of zero worlds simulates nothing."""
        row = encode_rung(_rung(2, True)).split(SEPARATOR)
        row[1] = "0"
        with pytest.raises(WireFormatError) as caught:
            decode_rung(SEPARATOR.join(row), 0)
        assert caught.value.code == "NP-WIRE-003"

    def test_rejects_a_negative_wall_time(self) -> None:
        """Time does not run backwards, and a negative wall inverts throughput."""
        row = encode_rung(_rung(2, True)).split(SEPARATOR)
        row[5] = (-1.0).hex()
        with pytest.raises(WireFormatError) as caught:
            decode_rung(SEPARATOR.join(row), 0)
        assert caught.value.code == "NP-WIRE-016"


class TestScalingRunRoundTrip:
    """Encoding and decoding compose to the identity."""

    def test_starts_with_the_scaling_run_banner(self) -> None:
        """The banner separates it from the sweep run it resembles."""
        assert encode_scaling_run(_record()).split("\n")[0] == SCALING_RUN_BANNER

    def test_round_trips(self) -> None:
        """A ladder survives encoding and decoding exactly."""
        assert decode_scaling_run(encode_scaling_run(_record())) == _record()

    def test_preserves_the_rung_order(self) -> None:
        """Ladder order is the curve's x-axis."""
        decoded = decode_scaling_run(encode_scaling_run(_record()))
        assert [rung["world_count"] for rung in decoded["rungs"]] == [2, 64, 512, 4096]

    def test_holds_the_scene_once_rather_than_per_rung(self) -> None:
        """A per-row scene would imply the ladder could vary it."""
        assert encode_scaling_run(_record()).count("body_count") == 1

    def test_is_byte_stable_for_equal_records(self) -> None:
        """Equal ladders encode identically, so two runs can be diffed."""
        assert encode_scaling_run(_record()) == encode_scaling_run(_record())


class TestScalingRunRejections:
    """Malformed ladders the decoder refuses."""

    def test_rejects_a_truncated_header(self) -> None:
        """Every condition is required."""
        with pytest.raises(WireFormatError) as caught:
            decode_scaling_run(f"{SCALING_RUN_BANNER}\nmode\tRUN_TO_RUN\n")
        assert caught.value.code == "NP-WIRE-010"

    def test_rejects_a_zero_capacity(self) -> None:
        """A zero constraint allocation cannot hold a contact."""
        lines = encode_scaling_run(_record()).strip("\n").split("\n")
        lines[6] = f"capacity{SEPARATOR}0"
        with pytest.raises(WireFormatError) as caught:
            decode_scaling_run("\n".join(lines) + "\n")
        assert caught.value.code == "NP-WIRE-003"

    def test_rejects_a_ladder_with_no_rungs(self) -> None:
        """An empty ladder decodes, and carries no rungs.

        The conditions are still readable, so a caller sees that nothing ran
        rather than a malformed file.
        """
        record = ScalingRunRecord(
            mode=_record()["mode"],
            device=_record()["device"],
            device_request=_record()["device_request"],
            max_records=_record()["max_records"],
            linesearch_block_dim=_record()["linesearch_block_dim"],
            capacity=_record()["capacity"],
            scene=_record()["scene"],
            spec=_record()["spec"],
            perturbation=_record()["perturbation"],
            rungs=(),
        )
        assert decode_scaling_run(encode_scaling_run(record))["rungs"] == ()
