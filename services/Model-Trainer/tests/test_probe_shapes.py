"""Tests for which probes exist and what identifies them.

The test that carries this file is
``test_every_axis_moves_exactly_the_field_it_claims_to``. The ladder's entire
value is that a rung where cross-card agreement breaks NAMES the axis that
broke it; a rung that quietly moved two fields at once would still produce a
number, still be reported at a position on an axis, and mean nothing. That is
the failure mode the earlier measurement already hit -- the gate probe and the
full cloze run differ in model size AND in input, which is why neither could
be blamed -- so it is checked here rather than described in a comment.
"""

from __future__ import annotations

import pytest

from model_trainer.core.services.model.model_sizes import GPT2_MODEL_SIZES
from model_trainer.core.services.model.probe_shapes import (
    GATE_RUNG,
    PROBE_AXES,
    PROBE_AXIS_FIELDS,
    PROBE_LABEL,
    PROBE_SHAPES,
    SHAPE_FIELDS,
    ProbeShape,
    differing_shape_fields,
    probe_label,
    require_probe_shape,
)

GATE_SHAPE = PROBE_SHAPES[GATE_RUNG]


def axis_rungs(name: str) -> tuple[str, ...]:
    """Return one axis's rungs, in axis order.

    Args:
        name: The axis name.

    Returns:
        Its rungs.

    Raises:
        KeyError: If no axis carries that name.
    """
    for axis in PROBE_AXES:
        if axis["name"] == name:
            return axis["rungs"]
    raise KeyError(f"no axis named {name!r}")


class TestTheTableItself:
    def test_the_gate_rung_is_in_the_table(self) -> None:
        assert GATE_RUNG in PROBE_SHAPES

    def test_every_rung_names_a_size_the_shared_table_implements(self) -> None:
        # A rung naming an absent size raises from a dict index deep inside
        # model construction, on a compute node, minutes into a job.
        unknown = [
            rung
            for rung, shape in PROBE_SHAPES.items()
            if shape["model_size"] not in GPT2_MODEL_SIZES
        ]

        assert unknown == []

    def test_no_rung_asks_for_more_tokens_than_it_has_vocabulary(self) -> None:
        # The probe input is the identity `arange`, so such a rung would index
        # embeddings that do not exist. `probe_forward_loss` refuses it; this
        # keeps the declared table from containing one at all.
        over = [
            rung
            for rung, shape in PROBE_SHAPES.items()
            if shape["sequence_len"] > shape["vocab_size"]
        ]

        assert over == []

    def test_every_rung_has_its_own_label(self) -> None:
        # Two rungs sharing a label would name one number twice, and a
        # registry could then hold two entries that can never disagree.
        labels = [probe_label(shape) for shape in PROBE_SHAPES.values()]

        assert sorted(labels) == sorted(set(labels))

    def test_the_declared_field_list_is_the_whole_shape(self) -> None:
        # The drift guard behind `differing_shape_fields`: a fourth field
        # added to ProbeShape fails here until the comparison learns to look
        # at it, so an axis cannot be silently ignored.
        assert sorted(SHAPE_FIELDS) == sorted(ProbeShape.__annotations__)


class TestTheAxes:
    def test_every_axis_names_a_field_and_every_named_field_has_an_axis(self) -> None:
        assert sorted(axis["name"] for axis in PROBE_AXES) == sorted(PROBE_AXIS_FIELDS)

    def test_every_axis_field_is_a_real_shape_field(self) -> None:
        assert [field for field in PROBE_AXIS_FIELDS.values() if field not in SHAPE_FIELDS] == []

    def test_every_axis_starts_at_the_gate_rung(self) -> None:
        # The gate rung is the one rung whose cross-card behaviour is already
        # measured, so it is the origin every other rung is read against. An
        # axis starting elsewhere would have no measured zero.
        assert [axis["rungs"][0] for axis in PROBE_AXES] == [GATE_RUNG for _ in PROBE_AXES]

    def test_every_rung_an_axis_names_exists(self) -> None:
        missing = [
            rung for axis in PROBE_AXES for rung in axis["rungs"] if rung not in PROBE_SHAPES
        ]

        assert missing == []

    def test_every_rung_in_the_table_sits_on_an_axis(self) -> None:
        # An orphan rung would be run on every card and reported by nothing,
        # because the report walks axes.
        placed = {rung for axis in PROBE_AXES for rung in axis["rungs"]}

        assert sorted(set(PROBE_SHAPES) - placed) == []

    def test_every_axis_moves_exactly_the_field_it_claims_to(self) -> None:
        # THE test in this file. See the module docstring.
        offenders: list[tuple[str, str, tuple[str, ...]]] = []
        for axis in PROBE_AXES:
            field = PROBE_AXIS_FIELDS[axis["name"]]
            for rung in axis["rungs"]:
                moved = differing_shape_fields(GATE_SHAPE, PROBE_SHAPES[rung])
                if moved not in ((), (field,)):
                    offenders.append((axis["name"], rung, moved))

        assert offenders == []

    def test_no_axis_repeats_a_rung(self) -> None:
        for axis in PROBE_AXES:
            assert sorted(axis["rungs"]) == sorted(set(axis["rungs"]))

    def test_the_model_size_axis_climbs_the_shared_size_table(self) -> None:
        # Written for this axis by name rather than generically: the ordering
        # of a size is its position in GPT2_MODEL_SIZES, and of a length is
        # the number itself. One loop over both would have to compare a str
        # to an int.
        order = list(GPT2_MODEL_SIZES)
        positions = [
            order.index(PROBE_SHAPES[rung]["model_size"]) for rung in axis_rungs("model-size")
        ]

        assert positions == sorted(positions)
        assert len(set(positions)) == len(positions)

    def test_the_sequence_length_axis_climbs(self) -> None:
        lengths = [PROBE_SHAPES[rung]["sequence_len"] for rung in axis_rungs("sequence-length")]

        assert lengths == sorted(lengths)
        assert len(set(lengths)) == len(lengths)


class TestDifferingShapeFields:
    def test_two_identical_shapes_differ_in_nothing(self) -> None:
        copied: ProbeShape = {**GATE_SHAPE}

        assert differing_shape_fields(GATE_SHAPE, copied) == ()

    def test_it_names_a_single_moved_field(self) -> None:
        moved: ProbeShape = {**GATE_SHAPE, "sequence_len": GATE_SHAPE["sequence_len"] * 2}

        assert differing_shape_fields(GATE_SHAPE, moved) == ("sequence_len",)

    def test_it_names_every_moved_field_in_declaration_order(self) -> None:
        moved: ProbeShape = {"model_size": "xl", "sequence_len": 1, "vocab_size": 2}

        assert differing_shape_fields(GATE_SHAPE, moved) == SHAPE_FIELDS


class TestLabels:
    def test_the_gate_label_is_the_one_the_deployed_registry_carries(self) -> None:
        # Asserted as a literal, not derived. Eight entries in the deployed
        # registry carry this exact string; a refactor that renames the gate
        # rung strands all of them, and a derived assertion would follow the
        # rename silently.
        assert PROBE_LABEL == "gpt2-tiny-L2-d128-h2-v512-len64-seed42"

    def test_a_label_names_every_field_of_its_shape(self) -> None:
        shape = PROBE_SHAPES["tiny-len256"]
        dims = GPT2_MODEL_SIZES[shape["model_size"]]

        assert probe_label(shape) == (
            f"gpt2-{shape['model_size']}"
            f"-L{dims['n_layer']}-d{dims['hidden_size']}-h{dims['n_head']}"
            f"-v{shape['vocab_size']}-len{shape['sequence_len']}-seed42"
        )

    def test_a_label_does_not_carry_the_rung_name(self) -> None:
        # Two rung names for one shape must produce one label. "tiny-len256"
        # is the clearest case: the rung is named for how it differs, and the
        # label is built from what it IS.
        assert "tiny-len256" not in probe_label(PROBE_SHAPES["tiny-len256"])

    def test_a_size_the_shared_table_lacks_is_refused(self) -> None:
        with pytest.raises(KeyError):
            probe_label({"model_size": "colossal", "sequence_len": 64, "vocab_size": 512})


class TestRequireProbeShape:
    def test_a_declared_rung_comes_back(self) -> None:
        assert require_probe_shape(GATE_RUNG) == GATE_SHAPE

    def test_an_unknown_rung_is_refused(self) -> None:
        with pytest.raises(KeyError, match="unknown probe rung 'huge'"):
            require_probe_shape("huge")

    def test_the_refusal_lists_what_does_exist(self) -> None:
        # The cause is nearly always a typo, and the list answers it.
        with pytest.raises(KeyError) as excinfo:
            require_probe_shape("tiny-len64")

        for rung in PROBE_SHAPES:
            assert rung in str(excinfo.value)
