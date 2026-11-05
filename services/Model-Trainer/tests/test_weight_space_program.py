"""The weight-program applier, on real torch parameters with hand-worked numbers.

Every expected value below is worked by hand from the operation's definition
in :mod:`model_trainer.core.contracts.weight_program`, not read back from the
applier. The parameters are real ``torch.nn.Parameter`` tensors held by name,
as the control policy holds its own; nothing stands in for a parameter, a
write or a digest.
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


class Weights:
    """Named parameters, in the shape :class:`ParameterHostProto` reads.

    Attributes:
        by_name: The parameters.
    """

    by_name: dict[str, torch.nn.Parameter]

    def __init__(self, by_name: dict[str, torch.nn.Parameter]) -> None:
        """Hold the given parameters.

        Args:
            by_name: The parameters, in the order to list them.
        """
        self.by_name = by_name

    def named_parameters(self) -> tuple[tuple[str, torch.nn.Parameter], ...]:
        """Every parameter with its name.

        Returns:
            ``(name, parameter)`` pairs.
        """
        return tuple(self.by_name.items())

    def get_parameter(self, target: str) -> torch.nn.Parameter:
        """One parameter by name.

        Args:
            target: The name.

        Returns:
            The parameter.
        """
        return self.by_name[target]

    def value(self, name: str) -> torch.Tensor:
        """One parameter's current values, detached.

        Args:
            name: The name.

        Returns:
            The values.
        """
        return self.by_name[name].detach()


def _two_layer() -> Weights:
    """A 3 -> 2 -> 2 network with ``first.weight = [[1, 2, 3], [4, 5, 6]]``.

    Returns:
        Its weights: 6 + 2 + 4 + 2 = 14 elements.
    """
    return Weights(
        {
            "first.weight": torch.nn.Parameter(_f([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
            "first.bias": torch.nn.Parameter(_f([10.0, 20.0])),
            "second.weight": torch.nn.Parameter(_f([[0.5, -0.5], [1.5, 2.5]])),
            "second.bias": torch.nn.Parameter(_f([0.0, 1.0])),
        }
    )


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


def _program(model: Weights, ops: tuple[WeightOp, ...]) -> WeightProgram:
    """Wrap operations in a program written against the weights as they are now.

    Args:
        model: The weights.
        ops: The operations.

    Returns:
        The program.
    """
    return WeightProgram(
        program_id="hand", written_against=weights_digest(parameter_digests(model)), ops=ops
    )


def test_scaling_rows_multiplies_only_those_rows() -> None:
    model = _two_layer()
    apply_op(model, _op("scale", "first.weight", 0, (1,), (0.5,)))
    assert torch.equal(model.value("first.weight"), _f([[1.0, 2.0, 3.0], [2.0, 2.5, 3.0]]))


def test_scaling_a_column_multiplies_that_column() -> None:
    model = _two_layer()
    apply_op(model, _op("scale", "first.weight", 1, (2,), (-1.0,)))
    assert torch.equal(model.value("first.weight"), _f([[1.0, 2.0, -3.0], [4.0, 5.0, -6.0]]))


def test_setting_rows_writes_each_its_own_part_of_the_values() -> None:
    model = _two_layer()
    apply_op(model, _op("set", "first.weight", 0, (1, 0), (7.0, 8.0, 9.0, 0.0, 0.0, 1.0)))
    assert torch.equal(model.value("first.weight"), _f([[0.0, 0.0, 1.0], [7.0, 8.0, 9.0]]))


def test_setting_elements_of_a_vector() -> None:
    model = _two_layer()
    apply_op(model, _op("set", "first.bias", 0, (1,), (-2.5,)))
    assert torch.equal(model.value("first.bias"), _f([10.0, -2.5]))


def test_projecting_out_removes_the_component_along_the_direction() -> None:
    """Row [4, 5, 6] minus its projection on (0, 3, 4)/5: (5*3 + 6*4)/5 = 7.8.

    The removed part is 7.8 * (0, 0.6, 0.8) = (0, 4.68, 6.24), leaving
    (4, 0.32, -0.24), whose dot product with the direction is zero.
    """
    model = _two_layer()
    apply_op(model, _op("project_out", "first.weight", 0, (1,), (0.0, 3.0, 4.0)))
    assert torch.allclose(model.value("first.weight")[1], _f([4.0, 0.32, -0.24]), atol=1e-6)
    assert torch.equal(model.value("first.weight")[0], _f([1.0, 2.0, 3.0]))


def test_projecting_out_along_columns_treats_each_column_as_the_vector() -> None:
    """Column 0 is (1, 4); removing its part along (1, 0) leaves (0, 4)."""
    model = _two_layer()
    apply_op(model, _op("project_out", "first.weight", 1, (0,), (2.0, 0.0)))
    assert torch.equal(model.value("first.weight"), _f([[0.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


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
    model = _two_layer()
    before = parameter_digests(model)
    with pytest.raises(AppError) as caught:
        apply_op(model, op)
    assert caught.value.code is code
    assert parameter_digests(model) == before


def test_a_parameter_of_three_dimensions_is_refused() -> None:
    cube = Weights({"cube": torch.nn.Parameter(torch.zeros(2, 2, 2))})
    with pytest.raises(AppError) as caught:
        apply_op(cube, _op("scale", "cube", 0, (0,), (2.0,)))
    assert caught.value.code is ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH


def test_a_program_counts_what_it_changed_and_comes_back_out() -> None:
    """Row 1 of first.weight halved (3 elements) and bias 0 set to its own value (0).

    The bias write is equal to what was there, so it changes nothing and the
    count is three, out of 6 + 2 + 4 + 2 = 14 weights.
    """
    model = _two_layer()
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
    model = _two_layer()
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
    model = _two_layer()
    model.by_name["second.weight"] = torch.nn.Parameter(model.value("first.weight")[:, :2])
    program = _program(model, (_op("scale", "first.weight", 0, (0,), (3.0,)),))
    with pytest.raises(AppError) as caught:
        apply_weight_program(model, program)
    assert caught.value.code is ModelTrainerErrorCode.EDIT_VERIFICATION_FAILED
    assert "second.weight moved as well" in caught.value.message


def test_removing_a_program_after_something_else_moved_is_refused() -> None:
    model = _two_layer()
    applied = apply_weight_program(
        model, _program(model, (_op("scale", "first.bias", 0, (0,), (2.0,)),))
    )
    with torch.no_grad():
        model.get_parameter("second.bias").add_(1.0)
    with pytest.raises(AppError) as caught:
        remove_weight_program(model, applied)
    assert caught.value.code is ModelTrainerErrorCode.EDIT_RESTORE_MISMATCH
    assert "second.bias still differ" in caught.value.message


def test_a_vector_snapshot_restores_byte_for_byte() -> None:
    """snapshot_parameter takes any shape, which the rank-one snapshot refuses."""
    model = _two_layer()
    snapshot = snapshot_parameter(model, "first.bias")
    with torch.no_grad():
        model.get_parameter("first.bias").mul_(3.0)
    restore_weight(model, snapshot)
    assert torch.equal(model.value("first.bias"), _f([10.0, 20.0]))
