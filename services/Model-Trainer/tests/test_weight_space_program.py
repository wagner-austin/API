"""The weight-program applier, on real torch modules with hand-worked numbers.

Every expected value below is worked by hand from the operation's definition
in :mod:`model_trainer.core.contracts.weight_program`, not read back from the
applier. The networks are real ``torch.nn`` modules; nothing stands in for a
parameter, a write or a digest.
"""

from __future__ import annotations

import pytest
import torch
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.contracts.weight_program import WeightOp, WeightProgram
from model_trainer.core.services.model.editing.apply import restore_weight, snapshot_parameter
from model_trainer.core.services.model.editing.verify import parameter_digests, weights_digest
from model_trainer.core.services.model.weight_space.program import (
    apply_op,
    apply_weight_program,
    remove_weight_program,
)


def _f(rows: list[float] | list[list[float]]) -> torch.Tensor:
    """Build a float32 tensor from hand-written numbers.

    Args:
        rows: A vector or a matrix.

    Returns:
        The tensor.
    """
    return torch.tensor(rows, dtype=torch.float32)


class TwoLayer(torch.nn.Module):
    """A 3 -> 2 -> 2 network with set, known weights.

    Attributes:
        first: The first layer.
        second: The second.
    """

    first: torch.nn.Linear
    second: torch.nn.Linear

    def __init__(self) -> None:
        """Build the layers and write ``first.weight = [[1, 2, 3], [4, 5, 6]]``."""
        super().__init__()
        self.first = torch.nn.Linear(3, 2)
        self.second = torch.nn.Linear(2, 2)
        with torch.no_grad():
            self.first.weight.copy_(torch.arange(1.0, 7.0).reshape(2, 3))
            self.first.bias.copy_(torch.tensor([10.0, 20.0]))


class Cube(torch.nn.Module):
    """A module holding one three-dimensional parameter.

    Attributes:
        cube: A 2 x 2 x 2 tensor, which no operation is defined on.
    """

    cube: torch.nn.Parameter

    def __init__(self) -> None:
        """Build the parameter."""
        super().__init__()
        self.cube = torch.nn.Parameter(torch.zeros(2, 2, 2))


def _op(
    op: str, parameter: str, axis: int, indices: tuple[int, ...], values: tuple[float, ...]
) -> WeightOp:
    """Build an operation, narrowing the name the way the decoder does.

    Args:
        op: ``set``, ``scale`` or ``project_out``.
        parameter: The parameter path.
        axis: The axis.
        indices: The selected slices.
        values: The numbers.

    Returns:
        The operation.
    """
    if op == "set":
        return WeightOp(op="set", parameter=parameter, axis=axis, indices=indices, values=values)
    if op == "scale":
        return WeightOp(op="scale", parameter=parameter, axis=axis, indices=indices, values=values)
    return WeightOp(
        op="project_out", parameter=parameter, axis=axis, indices=indices, values=values
    )


def _program(model: torch.nn.Module, ops: tuple[WeightOp, ...]) -> WeightProgram:
    """Wrap operations in a program written against the model as it is now.

    Args:
        model: The model.
        ops: The operations.

    Returns:
        The program.
    """
    return WeightProgram(
        program_id="hand", written_against=weights_digest(parameter_digests(model)), ops=ops
    )


def test_scaling_rows_multiplies_only_those_rows() -> None:
    model = TwoLayer()
    apply_op(model, _op("scale", "first.weight", 0, (1,), (0.5,)))
    assert torch.equal(model.first.weight.detach(), _f([[1.0, 2.0, 3.0], [2.0, 2.5, 3.0]]))


def test_scaling_a_column_multiplies_that_column() -> None:
    model = TwoLayer()
    apply_op(model, _op("scale", "first.weight", 1, (2,), (-1.0,)))
    assert torch.equal(model.first.weight.detach(), _f([[1.0, 2.0, -3.0], [4.0, 5.0, -6.0]]))


def test_setting_rows_writes_each_its_own_part_of_the_values() -> None:
    model = TwoLayer()
    apply_op(model, _op("set", "first.weight", 0, (1, 0), (7.0, 8.0, 9.0, 0.0, 0.0, 1.0)))
    assert torch.equal(model.first.weight.detach(), _f([[0.0, 0.0, 1.0], [7.0, 8.0, 9.0]]))


def test_setting_elements_of_a_vector() -> None:
    model = TwoLayer()
    apply_op(model, _op("set", "first.bias", 0, (1,), (-2.5,)))
    assert torch.equal(model.first.bias.detach(), _f([10.0, -2.5]))


def test_projecting_out_removes_the_component_along_the_direction() -> None:
    """Row [4, 5, 6] minus its projection on (0, 3, 4)/5: (5*3 + 6*4)/5 = 7.8.

    The removed part is 7.8 * (0, 0.6, 0.8) = (0, 4.68, 6.24), leaving
    (4, 0.32, -0.24), whose dot product with the direction is zero.
    """
    model = TwoLayer()
    apply_op(model, _op("project_out", "first.weight", 0, (1,), (0.0, 3.0, 4.0)))
    row = model.first.weight.detach()[1]
    assert torch.allclose(row, torch.tensor([4.0, 0.32, -0.24]), atol=1e-6)
    assert torch.equal(model.first.weight.detach()[0], _f([1.0, 2.0, 3.0]))


def test_projecting_out_along_columns_treats_each_column_as_the_vector() -> None:
    """Column 0 is (1, 4); removing its part along (1, 0) leaves (0, 4)."""
    model = TwoLayer()
    apply_op(model, _op("project_out", "first.weight", 1, (0,), (2.0, 0.0)))
    assert torch.equal(model.first.weight.detach(), _f([[0.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


@pytest.mark.parametrize(
    ("op", "code"),
    [
        (
            _op("scale", "first.nothing", 0, (0,), (2.0,)),
            ModelTrainerErrorCode.EDIT_MODULE_NOT_FOUND,
        ),
        (
            _op("scale", "first.bias", 1, (0,), (2.0,)),
            ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH,
        ),
        (
            _op("scale", "first.weight", 0, (2,), (2.0,)),
            ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH,
        ),
        (
            _op("set", "first.weight", 0, (0,), (1.0, 2.0)),
            ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH,
        ),
        (
            _op("project_out", "first.weight", 0, (0,), (1.0, 0.0)),
            ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH,
        ),
    ],
)
def test_an_operation_that_does_not_fit_the_parameter_is_refused(
    op: WeightOp, code: ModelTrainerErrorCode
) -> None:
    model = TwoLayer()
    before = parameter_digests(model)
    with pytest.raises(AppError) as caught:
        apply_op(model, op)
    assert caught.value.code is code
    assert parameter_digests(model) == before


def test_a_parameter_of_three_dimensions_is_refused() -> None:
    with pytest.raises(AppError) as caught:
        apply_op(Cube(), _op("scale", "cube", 0, (0,), (2.0,)))
    assert caught.value.code is ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH


def test_a_program_counts_what_it_changed_and_comes_back_out() -> None:
    """Row 1 of first.weight halved (3 elements) and bias 0 set to its own value (0).

    The bias write is equal to what was there, so it changes nothing and the
    count is three, out of 6 + 2 + 4 + 2 = 14 weights.
    """
    model = TwoLayer()
    original = parameter_digests(model)
    program = _program(
        model,
        (
            _op("scale", "first.weight", 0, (1,), (0.5,)),
            _op("set", "first.bias", 0, (0,), (10.0,)),
        ),
    )
    applied = apply_weight_program(model, program)
    effect = applied["effect"]
    assert effect["elements_changed"] == 3
    assert effect["elements_total"] == 14
    assert effect["parameters_written"] == ("first.bias", "first.weight")
    assert effect["weights_before"] == program["written_against"]
    assert effect["weights_after"] == weights_digest(parameter_digests(model))
    assert effect["weights_after"] != effect["weights_before"]
    remove_weight_program(model, applied)
    assert parameter_digests(model) == original


def test_a_program_written_against_other_weights_is_refused_before_writing() -> None:
    model = TwoLayer()
    before = parameter_digests(model)
    program = WeightProgram(
        program_id="elsewhere",
        written_against="sha256:" + "0" * 64,
        ops=(_op("scale", "first.weight", 0, (0,), (2.0,)),),
    )
    with pytest.raises(AppError) as caught:
        apply_weight_program(model, program)
    assert caught.value.code is ModelTrainerErrorCode.EDIT_VERIFICATION_FAILED
    assert parameter_digests(model) == before


def test_a_write_that_reaches_a_parameter_it_did_not_name_is_refused() -> None:
    """Two parameters sharing one storage: writing one moves the other.

    This is how weight tying looks when it is done by aliasing storage rather
    than by registering one parameter twice, and it is exactly the case where
    a program changes something it never named.
    """
    model = TwoLayer()
    model.second.weight = torch.nn.Parameter(model.first.weight.data[:, :2])
    program = _program(model, (_op("scale", "first.weight", 0, (0,), (3.0,)),))
    with pytest.raises(AppError) as caught:
        apply_weight_program(model, program)
    assert caught.value.code is ModelTrainerErrorCode.EDIT_VERIFICATION_FAILED
    assert "second.weight moved as well" in caught.value.message


def test_removing_a_program_after_something_else_moved_is_refused() -> None:
    model = TwoLayer()
    applied = apply_weight_program(
        model, _program(model, (_op("scale", "first.bias", 0, (0,), (2.0,)),))
    )
    with torch.no_grad():
        model.second.bias.add_(1.0)
    with pytest.raises(AppError) as caught:
        remove_weight_program(model, applied)
    assert caught.value.code is ModelTrainerErrorCode.EDIT_RESTORE_MISMATCH
    assert "second.bias still differ" in caught.value.message


def test_a_vector_snapshot_restores_byte_for_byte() -> None:
    """snapshot_parameter takes any shape, which the rank-one snapshot refuses."""
    model = TwoLayer()
    snapshot = snapshot_parameter(model, "first.bias")
    with torch.no_grad():
        model.first.bias.mul_(3.0)
    restore_weight(model, snapshot)
    assert torch.equal(model.first.bias.detach(), _f([10.0, 20.0]))
