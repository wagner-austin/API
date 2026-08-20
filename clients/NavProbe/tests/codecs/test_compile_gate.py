"""Tests for the compile-gate record codec."""

from __future__ import annotations

import pytest

from navprobe.codecs.compile_gate import (
    COMPILE_GATE_BANNER,
    decode_compile_gate,
    encode_compile_gate,
)
from navprobe.records import CompileGateRecord, SceneSpec
from navprobe.wireformat import SEPARATOR, WireFormatError


def _record() -> CompileGateRecord:
    """Build a compile-gate result.

    Returns:
        The shape the tactile-patch experiment produced: the touching-6 scene,
        compiled cold on the Warp CPU device.
    """
    return CompileGateRecord(
        mode="RUN_TO_RUN",
        device="cpu",
        device_request="cpu",
        max_records=0,
        wall_seconds=19.1,
        world_count=2,
        scene=SceneSpec(body_count=6, lattice_width=6, spacing=0.055, radius=0.03, timestep=0.005),
    )


class TestCompileGateRoundTrip:
    """Encoding and decoding compose to the identity."""

    def test_starts_with_the_compile_gate_banner(self) -> None:
        """The banner separates it from every other record."""
        assert encode_compile_gate(_record()).split("\n")[0] == COMPILE_GATE_BANNER

    def test_round_trips(self) -> None:
        """A result survives encoding and decoding exactly."""
        assert decode_compile_gate(encode_compile_gate(_record())) == _record()

    def test_carries_the_scene_that_compiled(self) -> None:
        """A pass on one scene is not a pass on another."""
        assert decode_compile_gate(encode_compile_gate(_record()))["scene"] == _record()["scene"]

    def test_preserves_the_cold_codegen_time(self) -> None:
        """The wall time is the figure that says the cache was cold."""
        assert decode_compile_gate(encode_compile_gate(_record()))["wall_seconds"] == 19.1

    def test_is_byte_stable_for_equal_records(self) -> None:
        """Equal results encode identically."""
        assert encode_compile_gate(_record()) == encode_compile_gate(_record())


class TestCompileGateRejections:
    """Malformed results the decoder refuses."""

    def test_rejects_trailing_lines(self) -> None:
        """The record declares no rows, so trailing content means another file."""
        with pytest.raises(WireFormatError) as caught:
            decode_compile_gate(encode_compile_gate(_record()) + "extra\n")
        assert caught.value.code == "NP-WIRE-013"

    def test_rejects_a_truncated_header(self) -> None:
        """Every condition is required."""
        with pytest.raises(WireFormatError) as caught:
            decode_compile_gate(f"{COMPILE_GATE_BANNER}\nmode\tRUN_TO_RUN\n")
        assert caught.value.code == "NP-WIRE-010"

    def test_rejects_a_negative_wall_time(self) -> None:
        """A gate cannot have taken less than no time."""
        lines = encode_compile_gate(_record()).strip("\n").split("\n")
        lines[5] = f"wall_seconds{SEPARATOR}{(-1.0).hex()}"
        with pytest.raises(WireFormatError) as caught:
            decode_compile_gate("\n".join(lines) + "\n")
        assert caught.value.code == "NP-WIRE-016"

    def test_rejects_a_zero_world_count(self) -> None:
        """A gate that allocated no worlds compiled nothing."""
        lines = encode_compile_gate(_record()).strip("\n").split("\n")
        lines[6] = f"world_count{SEPARATOR}0"
        with pytest.raises(WireFormatError) as caught:
            decode_compile_gate("\n".join(lines) + "\n")
        assert caught.value.code == "NP-WIRE-003"
