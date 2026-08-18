"""Tests for the scene specification codec."""

from __future__ import annotations

import pytest

from navprobe.codecs.run import encode_run_record
from navprobe.codecs.scene import (
    SCENE_BANNER,
    SCENE_FIELD_COUNT,
    decode_scene_fields,
    decode_scene_spec,
    encode_float_field,
    encode_scene_spec,
    require_positive_float_field,
    scene_fields,
)
from navprobe.records import RunRecord, RunSpec, SceneSpec
from navprobe.wireformat import SEPARATOR, WireFormatError


def _spec() -> SceneSpec:
    """Build a scene specification.

    Returns:
        A specification whose floats are not exactly representable, so a lossy
        codec would fail the round trip.
    """
    return SceneSpec(body_count=32, lattice_width=8, spacing=0.055, radius=0.03, timestep=0.005)


def _lines() -> list[str]:
    """Encode the reference specification and split it into lines.

    Returns:
        The encoded lines with the trailing blank removed.
    """
    return encode_scene_spec(_spec()).strip("\n").split("\n")


class TestFloatFields:
    """Tests for the exact float encoding."""

    def test_round_trips_a_value_with_no_exact_decimal_form(self) -> None:
        """The hexadecimal form recovers the original bits.

        0.055 is not exactly representable in binary, so this fails for any
        codec that goes through a rounded decimal string.
        """
        assert require_positive_float_field(encode_float_field(0.055), "f") == 0.055

    def test_round_trips_a_very_small_value(self) -> None:
        """Subnormal-adjacent values survive, which decimal formatting need not."""
        assert require_positive_float_field(encode_float_field(1e-300), "f") == 1e-300

    def test_rejects_a_decimal_token(self) -> None:
        """A decimal string is refused rather than silently parsed."""
        with pytest.raises(WireFormatError) as caught:
            require_positive_float_field("0.055", "f")
        assert caught.value.code == "NP-WIRE-014"

    def test_rejects_a_zero_value(self) -> None:
        """A length or duration of zero describes no scene."""
        with pytest.raises(WireFormatError) as caught:
            require_positive_float_field(encode_float_field(0.0), "f")
        assert caught.value.code == "NP-WIRE-015"

    def test_rejects_a_negative_value(self) -> None:
        """Negative lengths are refused by the same check."""
        with pytest.raises(WireFormatError) as caught:
            require_positive_float_field(encode_float_field(-0.5), "f")
        assert caught.value.code == "NP-WIRE-015"

    def test_rejects_not_a_number(self) -> None:
        """NaN passes the prefix check and must be caught by the range check."""
        with pytest.raises(WireFormatError) as caught:
            require_positive_float_field("nan", "f")
        assert caught.value.code == "NP-WIRE-015"


class TestSceneSpecDocument:
    """The standalone document form."""

    def test_starts_with_the_scene_banner(self) -> None:
        """The banner distinguishes a scene from every other record."""
        assert _lines()[0] == SCENE_BANNER

    def test_writes_the_header_in_fixed_order(self) -> None:
        """Field order is part of the format."""
        assert [line.split(SEPARATOR)[0] for line in _lines()[1:]] == [
            "body_count",
            "lattice_width",
            "spacing",
            "radius",
            "timestep",
        ]

    def test_round_trips(self) -> None:
        """A specification survives encoding and decoding exactly."""
        assert decode_scene_spec(encode_scene_spec(_spec())) == _spec()

    def test_is_byte_stable_for_equal_specifications(self) -> None:
        """Equal scenes encode identically, so files can be compared."""
        assert encode_scene_spec(_spec()) == encode_scene_spec(_spec())

    def test_rejects_another_record_types_banner(self) -> None:
        """A run record must not decode as a scene."""
        other = RunRecord(
            spec=RunSpec(label="x", seed=1, step_count=0, world_count=1), steps=(), digest="cc"
        )
        with pytest.raises(WireFormatError) as caught:
            decode_scene_spec(encode_run_record(other))
        assert caught.value.code == "NP-WIRE-009"

    def test_rejects_a_truncated_header(self) -> None:
        """Every field is required."""
        with pytest.raises(WireFormatError) as caught:
            decode_scene_spec(f"{SCENE_BANNER}\nbody_count\t4\n")
        assert caught.value.code == "NP-WIRE-010"

    def test_rejects_trailing_lines(self) -> None:
        """A scene declares no rows."""
        with pytest.raises(WireFormatError) as caught:
            decode_scene_spec(encode_scene_spec(_spec()) + "extra\n")
        assert caught.value.code == "NP-WIRE-013"

    def test_rejects_a_zero_body_count(self) -> None:
        """A scene with no bodies observes nothing."""
        lines = _lines()
        lines[1] = f"body_count{SEPARATOR}0"
        with pytest.raises(WireFormatError) as caught:
            decode_scene_spec("\n".join(lines))
        assert caught.value.code == "NP-WIRE-003"

    def test_rejects_a_zero_lattice_width(self) -> None:
        """A lattice needs a column to place bodies in."""
        lines = _lines()
        lines[2] = f"lattice_width{SEPARATOR}0"
        with pytest.raises(WireFormatError) as caught:
            decode_scene_spec("\n".join(lines))
        assert caught.value.code == "NP-WIRE-003"

    def test_rejects_a_non_positive_spacing(self) -> None:
        """Zero spacing puts every body at one point."""
        lines = _lines()
        lines[3] = f"spacing{SEPARATOR}{encode_float_field(0.0)}"
        with pytest.raises(WireFormatError) as caught:
            decode_scene_spec("\n".join(lines))
        assert caught.value.code == "NP-WIRE-015"


class TestSceneFieldsForEmbedding:
    """The row form a sweep embeds."""

    def test_emits_the_declared_field_count(self) -> None:
        """The embedded width is what the sweep codec slices by."""
        assert len(scene_fields(_spec())) == SCENE_FIELD_COUNT

    def test_round_trips_through_the_row_form(self) -> None:
        """Embedding and extracting compose to the identity."""
        assert decode_scene_fields(scene_fields(_spec()), 0) == _spec()

    def test_the_row_form_matches_the_document_form(self) -> None:
        """One declaration of field order, used by both forms.

        The document's header values are compared against the embedded fields,
        so the two cannot drift into disagreeing about what order they are in.
        """
        header_values = [line.split(SEPARATOR)[1] for line in _lines()[1:]]
        assert header_values == list(scene_fields(_spec()))

    def test_rejects_a_bad_field_and_names_its_row(self) -> None:
        """An error from a sweep row says which row it came from."""
        fields = list(scene_fields(_spec()))
        fields[0] = "0"
        with pytest.raises(WireFormatError) as caught:
            decode_scene_fields(tuple(fields), 3)
        assert "entry[3].body_count" in caught.value.message
