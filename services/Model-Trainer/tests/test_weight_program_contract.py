"""The weight-program document: what it accepts, what it refuses, and its round trip.

Every refusal here is a program a reader could plausibly write and that the
applier could not apply as written, so each is refused at decode with the
field it concerns named, rather than reaching a model.
"""

from __future__ import annotations

import random

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from model_trainer.core.contracts.weight_program import (
    WEIGHT_OPS,
    WeightOp,
    WeightProgram,
    as_weight_op_name,
    decode_weight_op,
    decode_weight_program,
    encode_weight_op,
    encode_weight_program,
)


def _op(overrides: JSONObject) -> JSONObject:
    """Build an operation object, a valid ``scale`` unless overridden.

    Args:
        overrides: Fields to replace.

    Returns:
        The object.
    """
    base: JSONObject = {
        "op": "scale",
        "parameter": "layer.weight",
        "axis": 0,
        "indices": [1, 3],
        "values": [0.5],
    }
    return {**base, **overrides}


def test_a_scale_decodes_to_its_fields() -> None:
    assert decode_weight_op(_op({})) == WeightOp(
        op="scale", parameter="layer.weight", axis=0, indices=(1, 3), values=(0.5,)
    )


def test_every_operation_name_is_accepted_and_others_are_refused() -> None:
    assert tuple(as_weight_op_name(name) for name in WEIGHT_OPS) == WEIGHT_OPS
    with pytest.raises(JSONTypeError, match="'op' must be one of set, scale, project_out"):
        as_weight_op_name("add")


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"axis": 2}, "'axis' must be 0 or 1, got 2"),
        ({"indices": []}, "must select at least one slice"),
        ({"indices": [0, -1]}, "must be non-negative, got -1"),
        ({"indices": [2, 2]}, "must not name a slice twice"),
        ({"values": []}, "must hold at least one number"),
        ({"values": [float("inf")]}, "only finite numbers"),
        ({"values": [1.0, 2.0]}, "exactly one factor, got 2 values"),
        ({"op": "set", "values": [1.0, 2.0, 3.0]}, "2 slices carries 3 values"),
        ({"op": "project_out", "values": [0.0, 0.0]}, "must not be the zero vector"),
        ({"parameter": ""}, "'parameter' must not be empty"),
    ],
)
def test_an_operation_that_cannot_be_applied_as_written_is_refused(
    fields: JSONObject, message: str
) -> None:
    with pytest.raises(JSONTypeError, match=message):
        decode_weight_op(_op(fields))


def test_a_set_whose_values_divide_evenly_is_accepted() -> None:
    decoded = decode_weight_op(_op({"op": "set", "values": [1.0, 2.0, 3.0, 4.0]}))
    assert decoded["values"] == (1.0, 2.0, 3.0, 4.0)


def test_a_program_with_no_operations_is_refused() -> None:
    with pytest.raises(JSONTypeError, match="'ops' must hold at least one operation"):
        decode_weight_program({"program_id": "p", "written_against": "sha256:x", "ops": []})


def test_a_program_naming_no_weights_is_refused() -> None:
    with pytest.raises(JSONTypeError, match="'written_against' must not be empty"):
        decode_weight_program({"program_id": "p", "written_against": "", "ops": [_op({})]})


def _random_op(rng: random.Random) -> WeightOp:
    """Generate one valid operation of a random kind and size.

    Args:
        rng: The generator.

    Returns:
        The operation.
    """
    kind = rng.choice(WEIGHT_OPS)
    indices = tuple(rng.sample(range(50), rng.randint(1, 5)))
    width = rng.randint(1, 4)
    if kind == "scale":
        values: tuple[float, ...] = (rng.uniform(-3.0, 3.0),)
    elif kind == "set":
        values = tuple(rng.uniform(-3.0, 3.0) for _ in range(width * len(indices)))
    else:
        values = (1.0, *(rng.uniform(-3.0, 3.0) for _ in range(width - 1)))
    return WeightOp(
        op=kind,
        parameter=f"block.{rng.randint(0, 9)}.weight",
        axis=rng.randint(0, 1),
        indices=indices,
        values=values,
    )


def test_every_generated_program_round_trips_through_its_encoding() -> None:
    """Property: decode(encode(p)) == p, over generated programs of every op kind."""
    rng = random.Random(20261105)
    for case in range(300):
        program = WeightProgram(
            program_id=f"case-{case}",
            written_against=f"sha256:{case:064x}",
            ops=tuple(_random_op(rng) for _ in range(rng.randint(1, 4))),
        )
        assert decode_weight_program(encode_weight_program(program)) == program


def test_every_generated_operation_round_trips_alone() -> None:
    rng = random.Random(7)
    for _ in range(300):
        op = _random_op(rng)
        assert decode_weight_op(encode_weight_op(op)) == op
