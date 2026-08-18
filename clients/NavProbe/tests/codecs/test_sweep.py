"""Tests for the sweep codec."""

from __future__ import annotations

import pytest

from navprobe.codecs.scene import encode_scene_spec
from navprobe.codecs.sweep import (
    ENTRY_TAG,
    ENTRY_TOKEN_COUNT,
    SWEEP_BANNER,
    decode_sweep,
    decode_sweep_entry,
    encode_sweep,
)
from navprobe.records import SceneSpec, SweepEntry, TrialRecord, TrialSpec
from navprobe.wireformat import SEPARATOR, WireFormatError


def _entry(body_count: int, deterministic: bool, divergence: int | None) -> SweepEntry:
    """Build one sweep entry.

    Args:
        body_count: Bodies in the entry's scene.
        deterministic: Whether the trial reproduced.
        divergence: The first divergent step, or ``None``.

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
            reference_digest=f"{body_count:064x}",
            deterministic=deterministic,
            first_divergent_step=divergence,
        ),
    )


def _sweep() -> tuple[SweepEntry, ...]:
    """Build a sweep spanning a reproducibility boundary.

    Returns:
        Three entries, the last two of which failed.
    """
    return (_entry(5, True, None), _entry(6, False, 0), _entry(8, False, 3))


class TestEncodeSweep:
    """Tests for :func:`encode_sweep`."""

    def test_starts_with_the_sweep_banner(self) -> None:
        """The banner distinguishes a sweep from every other record."""
        assert encode_sweep(_sweep()).split("\n")[0] == SWEEP_BANNER

    def test_writes_one_row_per_entry(self) -> None:
        """Each scene's verdict is one line."""
        assert len(encode_sweep(_sweep()).strip("\n").split("\n")) == 1 + 3

    def test_every_row_carries_the_declared_token_count(self) -> None:
        """Row width is fixed, which is what lets the decoder slice it."""
        rows = encode_sweep(_sweep()).strip("\n").split("\n")[1:]
        assert [len(row.split(SEPARATOR)) for row in rows] == [ENTRY_TOKEN_COUNT] * 3

    def test_encodes_an_empty_sweep_as_a_bare_banner(self) -> None:
        """A sweep with no entries is still a well-formed document."""
        assert encode_sweep(()) == f"{SWEEP_BANNER}\n"

    def test_is_byte_stable_for_equal_sweeps(self) -> None:
        """Equal sweeps encode identically, so results files can be compared."""
        assert encode_sweep(_sweep()) == encode_sweep(_sweep())


class TestSweepRoundTrip:
    """Encoding and decoding a sweep compose to the identity."""

    def test_round_trips_a_populated_sweep(self) -> None:
        """Every entry survives, scene and verdict alike."""
        assert decode_sweep(encode_sweep(_sweep())) == _sweep()

    def test_round_trips_an_empty_sweep(self) -> None:
        """The empty case is the base case, not a special one."""
        assert decode_sweep(encode_sweep(())) == ()

    def test_preserves_entry_order(self) -> None:
        """Sweep order is what makes a boundary readable."""
        decoded = decode_sweep(encode_sweep(_sweep()))
        assert [entry["scene"]["body_count"] for entry in decoded] == [5, 6, 8]

    def test_preserves_an_absent_divergence_distinctly_from_step_zero(self) -> None:
        """Reproducing and diverging-at-step-zero must not collapse together."""
        decoded = decode_sweep(encode_sweep(_sweep()))
        assert [entry["trial"]["first_divergent_step"] for entry in decoded] == [None, 0, 3]

    def test_preserves_exact_scene_floats(self) -> None:
        """Scene geometry survives embedding in a row."""
        decoded = decode_sweep(encode_sweep(_sweep()))
        assert decoded[0]["scene"]["spacing"] == 0.055


class TestSweepRejections:
    """Malformed sweeps the decoder refuses."""

    def test_rejects_another_record_types_banner(self) -> None:
        """A scene document must not decode as a sweep."""
        scene = SceneSpec(body_count=1, lattice_width=1, spacing=0.055, radius=0.03, timestep=0.005)
        with pytest.raises(WireFormatError) as caught:
            decode_sweep(encode_scene_spec(scene))
        assert caught.value.code == "NP-WIRE-009"

    def test_rejects_a_row_with_too_few_tokens(self) -> None:
        """A truncated row is refused rather than partly read."""
        with pytest.raises(WireFormatError) as caught:
            decode_sweep_entry(f"{ENTRY_TAG}{SEPARATOR}1", 0)
        assert caught.value.code == "NP-WIRE-017"

    def test_rejects_a_row_with_the_wrong_tag(self) -> None:
        """Rows after the banner must be tagged as entries."""
        row = encode_sweep(_sweep()).strip("\n").split("\n")[1]
        with pytest.raises(WireFormatError) as caught:
            decode_sweep_entry(row.replace(ENTRY_TAG, "note", 1), 0)
        assert caught.value.code == "NP-WIRE-017"

    def test_rejects_an_unspelled_determinism_verdict(self) -> None:
        """The verdict must be one of the two spelled tokens."""
        rows = encode_sweep(_sweep()).strip("\n").split("\n")
        parts = rows[1].split(SEPARATOR)
        parts[-2] = "maybe"
        with pytest.raises(WireFormatError) as caught:
            decode_sweep_entry(SEPARATOR.join(parts), 0)
        assert caught.value.code == "NP-WIRE-012"

    def test_rejects_an_empty_reference_digest(self) -> None:
        """An entry without a digest names no measurement."""
        rows = encode_sweep(_sweep()).strip("\n").split("\n")
        parts = rows[1].split(SEPARATOR)
        parts[-3] = ""
        with pytest.raises(WireFormatError) as caught:
            decode_sweep_entry(SEPARATOR.join(parts), 0)
        assert caught.value.code == "NP-WIRE-004"

    def test_rejects_a_zero_world_count(self) -> None:
        """An entry claiming no worlds observed nothing."""
        rows = encode_sweep(_sweep()).strip("\n").split("\n")
        parts = rows[1].split(SEPARATOR)
        parts[-4] = "0"
        with pytest.raises(WireFormatError) as caught:
            decode_sweep_entry(SEPARATOR.join(parts), 0)
        assert caught.value.code == "NP-WIRE-003"

    def test_names_the_row_a_bad_field_came_from(self) -> None:
        """A sweep of many rows says which one failed."""
        rows = encode_sweep(_sweep()).strip("\n").split("\n")
        parts = rows[1].split(SEPARATOR)
        parts[-1] = "-1"
        with pytest.raises(WireFormatError) as caught:
            decode_sweep_entry(SEPARATOR.join(parts), 7)
        assert "entry[7].first_divergent_step" in caught.value.message
