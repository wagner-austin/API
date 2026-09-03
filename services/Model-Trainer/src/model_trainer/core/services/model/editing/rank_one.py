"""The arithmetic of a rank-one weight edit, with no model in sight.

Everything here is a function of tensors, which is what makes it the layer
whose correctness is demonstrable rather than plausible. A rank-one update has
an exact consequence and this module states it in one place:

    W' = W + u (v^T)          =>          x W' - x W = (x . u) v

for every input x, with no dependence on what W held. The value solve is the
same identity read backwards: to move the module's output at one key to a
wanted value, divide the wanted change by the key's dot product with the
current input.

WHY ORIENTATION IS A RETURNED FLAG AND NOT A SILENT TRANSPOSE. GPT-2 stores
its MLP projections as ``Conv1D``, whose weight is (input, output), the
transpose of ``nn.Linear``. The composed update matches one of those directly
and the other only transposed. Applying whichever one happens to fit is
correct and unrecordable; returning WHICH one fit puts the fact on the edit's
record, so a run that flips it between two model families is reporting a real
difference rather than hiding one.

WHY THE DEGENERATE KEY IS AN ERROR AND NOT A CLAMP. When the key is nearly
orthogonal to the module's input at the edited position, the divisor goes to
zero and the value vector's norm goes to infinity for the same requested
change. Clamping it would produce an edit that is applied, recorded, and does
not do what it says. The request is refused with the measured divisor in the
message instead, because the caller's next move is to pick a different token
or layer, and it needs the number to decide.
"""

from __future__ import annotations

import torch
from platform_core.errors import AppError, ModelTrainerErrorCode
from typing_extensions import TypedDict

#: Below this, a key is treated as orthogonal to the input it is solved
#: against and the solve is refused.
#:
#: Not a tolerance on the answer: it is a statement about which requests this
#: arithmetic can answer at all. At 1e-6 a unit-norm value vector would need a
#: norm of a million to move the output by one, which no downstream fluency
#: check would survive.
MIN_SOLVE_DENOMINATOR = 1e-6


class OrientedUpdate(TypedDict):
    """A composed update matrix, turned to face a specific weight.

    Attributes:
        matrix: The update, in the stored weight's own orientation.
        transposed: Whether the composed (input, output) form had to be
            transposed to get there.
    """

    matrix: torch.Tensor
    transposed: bool


class RightVectorSolve(TypedDict):
    """The value vector, and the divisor it came from.

    Attributes:
        vector: The value vector, in the module's output space.
        denominator: The key's dot product with the module's current input at
            the edited position. Carried out of the solve rather than
            recomputed by the caller, so the number on the edit record is the
            one the division actually used.
    """

    vector: torch.Tensor
    denominator: float


def compose_rank_one(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Build the rank-one update matrix from a key and a value.

    Args:
        left: Key vector, one dimensional, in the module's input space.
        right: Value vector, one dimensional, in the module's output space.

    Returns:
        The outer product, shaped (input, output).

    Raises:
        AppError: With ``EDIT_UPDATE_SHAPE_MISMATCH`` if either argument is
            not one dimensional. A two-dimensional "vector" would broadcast
            into a matrix of the wrong rank, and the result would still
            multiply.
    """
    _require_vector(left, "left")
    _require_vector(right, "right")
    return torch.outer(left, right)


def orient_for_weight(update: torch.Tensor, weight_shape: torch.Size) -> OrientedUpdate:
    """Turn a composed update to face a stored weight.

    Args:
        update: The composed update, shaped (input, output).
        weight_shape: Shape of the weight it will be added to.

    Returns:
        The update in the weight's orientation, and whether that needed a
        transpose.

    Raises:
        AppError: With ``EDIT_WEIGHT_NOT_MATRIX`` if the target shape is not
            two dimensional, or ``EDIT_UPDATE_SHAPE_MISMATCH`` if neither the
            update nor its transpose matches it.
    """
    if len(weight_shape) != 2:
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_WEIGHT_NOT_MATRIX,
            message=(
                f"a rank-one edit needs a two-dimensional weight, got shape {tuple(weight_shape)}"
            ),
        )
    if update.shape == weight_shape:
        return OrientedUpdate(matrix=update, transposed=False)
    if update.T.shape == weight_shape:
        return OrientedUpdate(matrix=update.T, transposed=True)
    raise AppError(
        code=ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH,
        message=(
            f"update of shape {tuple(update.shape)} fits neither {tuple(weight_shape)} "
            f"nor its transpose; the key and value were computed against a different module"
        ),
    )


def solve_right_vector(
    *,
    target_output: torch.Tensor,
    current_output: torch.Tensor,
    current_input: torch.Tensor,
    left: torch.Tensor,
) -> RightVectorSolve:
    """Solve for the value vector that moves one output to a wanted value.

    The module computes ``out = in W``. After adding ``u v^T`` it computes
    ``out + (in . u) v``, so setting that equal to the wanted output gives
    ``v = (wanted - out) / (in . u)``.

    Args:
        target_output: What the module should emit at the edited position.
        current_output: What it emits there now.
        current_input: What it receives there now.
        left: The key vector, in the module's input space.

    Returns:
        The value vector and the divisor used.

    Raises:
        AppError: With ``EDIT_UPDATE_SHAPE_MISMATCH`` if any argument is not
            one dimensional, if the two output vectors disagree in length, or
            if the key and the input disagree in length. With
            ``EDIT_KEY_ORTHOGONAL_TO_INPUT`` if the divisor is smaller than
            :data:`MIN_SOLVE_DENOMINATOR` in magnitude.
    """
    _require_vector(target_output, "target_output")
    _require_vector(current_output, "current_output")
    _require_vector(current_input, "current_input")
    _require_vector(left, "left")
    _require_same_length(target_output, current_output, "target_output", "current_output")
    _require_same_length(current_input, left, "current_input", "left")

    denominator = float(torch.dot(current_input, left).item())
    if abs(denominator) < MIN_SOLVE_DENOMINATOR:
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_KEY_ORTHOGONAL_TO_INPUT,
            message=(
                f"key is orthogonal to the module input at the edited position "
                f"(dot product {denominator:.3e}, minimum {MIN_SOLVE_DENOMINATOR:.3e}); "
                f"the value vector this implies would be arbitrarily large, so the edit "
                f"is refused rather than applied at a scale no output survives"
            ),
        )
    return RightVectorSolve(
        vector=(target_output - current_output) / denominator,
        denominator=denominator,
    )


def predicted_output_delta(
    *, left: torch.Tensor, right: torch.Tensor, probe: torch.Tensor
) -> torch.Tensor:
    """Predict how a rank-one edit moves the output for one input.

    The whole content of a rank-one edit, in one line, and the thing the
    verification pass measures the real weights against.

    Args:
        left: The key vector.
        right: The value vector.
        probe: An input in the module's input space.

    Returns:
        The predicted change in the module's output, ``(probe . left) right``.

    Raises:
        AppError: With ``EDIT_UPDATE_SHAPE_MISMATCH`` if any argument is not
            one dimensional, or if the probe and key disagree in length.
    """
    _require_vector(left, "left")
    _require_vector(right, "right")
    _require_vector(probe, "probe")
    _require_same_length(probe, left, "probe", "left")
    return torch.dot(probe, left) * right


def _require_vector(tensor: torch.Tensor, name: str) -> None:
    """Refuse a tensor that is not one dimensional.

    Args:
        tensor: The tensor to check.
        name: Argument name for the message.

    Raises:
        AppError: With ``EDIT_UPDATE_SHAPE_MISMATCH`` if the rank is not one.
    """
    if tensor.dim() != 1:
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH,
            message=(
                f"argument '{name}' must be a one-dimensional vector, got shape "
                f"{tuple(tensor.shape)}"
            ),
        )


def _require_same_length(
    first: torch.Tensor, second: torch.Tensor, first_name: str, second_name: str
) -> None:
    """Refuse two vectors of different lengths.

    Args:
        first: The first vector.
        second: The second vector.
        first_name: Argument name of the first, for the message.
        second_name: Argument name of the second, for the message.

    Raises:
        AppError: With ``EDIT_UPDATE_SHAPE_MISMATCH`` if the lengths differ.
    """
    if first.shape[0] != second.shape[0]:
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH,
            message=(
                f"'{first_name}' has length {first.shape[0]} and '{second_name}' has "
                f"length {second.shape[0]}; they describe different spaces"
            ),
        )


__all__ = [
    "MIN_SOLVE_DENOMINATOR",
    "OrientedUpdate",
    "RightVectorSolve",
    "compose_rank_one",
    "orient_for_weight",
    "predicted_output_delta",
    "solve_right_vector",
]
