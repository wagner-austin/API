"""Apply a weight program to a network, prove where it wrote, and undo it.

An edit is measured against the network it started from, so three things have
to hold for the measurement to mean anything, and each is checked rather than
assumed.

The program was written against THESE weights. Its ``written_against`` digest
is compared with the live weights' before anything is written.

It wrote only where it said. Every parameter is digested before and after,
and any parameter the program did not name that moved is a refusal.

It can be taken back out. Each parameter it names is snapshotted first, and a
restore is confirmed against the digests taken before the edit -- the whole
model, not only the parameters written -- so a run that measures the edited
network and then the original again measures the original.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from platform_core.errors import AppError, ModelTrainerErrorCode
from typing_extensions import TypedDict

from model_trainer.core.contracts.weight_program import WeightOp, WeightProgram
from model_trainer.core.services.model.editing.apply import (
    WeightSnapshot,
    restore_weight,
    snapshot_parameter,
)
from model_trainer.core.services.model.editing.sites import ParameterHostProto, require_parameter
from model_trainer.core.services.model.editing.verify import (
    changed_parameters,
    parameter_digests,
    weights_digest,
)


class ProgramEffect(TypedDict):
    """What applying a program did to the network.

    Attributes:
        program_id: The program applied.
        weights_before: The digest of the weights it was applied to, which is
            the program's own ``written_against``.
        weights_after: The digest of the weights it left.
        parameters_written: Every parameter the program named, sorted.
        elements_changed: How many individual weights now differ from their
            value before the program ran. An operation that wrote a value
            equal to the old one changed nothing and is not counted.
        elements_total: How many weights the whole network has, so the
            fraction the post reports (3.5% in Experiment 4, 0.016% in
            Experiment 8) can be read off the record.
    """

    program_id: str
    weights_before: str
    weights_after: str
    parameters_written: tuple[str, ...]
    elements_changed: int
    elements_total: int


class AppliedProgram(TypedDict):
    """An applied program and what it takes to remove it.

    Not a JSON contract. It holds tensors and exists for the length of one
    measurement.

    Attributes:
        effect: What the program did.
        snapshots: The written parameters as they were before it ran.
        digests_before: Every parameter's digest before it ran.
    """

    effect: ProgramEffect
    snapshots: tuple[WeightSnapshot, ...]
    digests_before: dict[str, float]


def _as_slices(values: torch.Tensor, axis: int, parameter: str) -> torch.Tensor:
    """View a parameter so its selected slices are the rows of a matrix.

    Args:
        values: The parameter's values, one or two dimensional.
        axis: The axis the operation selects along.
        parameter: The parameter's name, for the refusal.

    Returns:
        A view sharing ``values``'s storage: the matrix itself for axis 0, its
        transpose for axis 1, and a one-column matrix for a vector.

    Raises:
        AppError: With ``EDIT_UPDATE_SHAPE_MISMATCH`` for axis 1 of a vector,
            which has no columns, or for a parameter of more than two
            dimensions, which no operation here is defined on.
    """
    if values.dim() == 1 and axis == 0:
        return values.unsqueeze(1)
    if values.dim() == 2:
        return values if axis == 0 else values.T
    raise AppError(
        code=ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH,
        message=(
            f"parameter '{parameter}' has shape {tuple(values.shape)}, which has no axis "
            f"{axis} of slices; an operation selects rows or columns of a matrix, or "
            f"elements of a vector"
        ),
    )


def _vector(values: Sequence[float], like: torch.Tensor) -> torch.Tensor:
    """Build a tensor of the given numbers in another tensor's dtype and device.

    Filled element by element rather than through ``torch.tensor``, whose
    stubs return ``Any``.

    Args:
        values: The numbers.
        like: The tensor whose dtype and device to use.

    Returns:
        The vector.
    """
    out = torch.empty(len(values), dtype=like.dtype, device=like.device)
    for position, value in enumerate(values):
        out[position] = value
    return out


def apply_op(model: ParameterHostProto, op: WeightOp) -> None:
    """Apply one operation to the parameter it names.

    Args:
        model: The weights, changed in place.
        op: The operation.

    Raises:
        AppError: With ``EDIT_MODULE_NOT_FOUND`` if the parameter does not
            exist, or ``EDIT_UPDATE_SHAPE_MISMATCH`` if the axis does not
            exist on it, an index is past its end, or the values do not fit
            its slices.
    """
    parameter = require_parameter(model, op["parameter"])
    updated = parameter.detach().clone()
    slices = _as_slices(updated, op["axis"], op["parameter"])
    count, width = int(slices.shape[0]), int(slices.shape[1])
    if max(op["indices"]) >= count:
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH,
            message=(
                f"'{op['parameter']}' has {count} slices along axis {op['axis']} and "
                f"the {op['op']} selects index {max(op['indices'])}"
            ),
        )
    if op["op"] == "scale":
        for index in op["indices"]:
            slices[index] = slices[index] * op["values"][0]
    elif op["op"] == "set":
        _set_slices(slices, op, width)
    else:
        _project_out(slices, op, width)
    with torch.no_grad():
        parameter.copy_(updated)


def _set_slices(slices: torch.Tensor, op: WeightOp, width: int) -> None:
    """Replace each selected slice with its part of the values, in place.

    Args:
        slices: The parameter viewed as rows of slices.
        op: A ``set`` operation.
        width: The slice width.

    Raises:
        AppError: With ``EDIT_UPDATE_SHAPE_MISMATCH`` if the values are not
            exactly one slice width per index.
    """
    if len(op["values"]) != width * len(op["indices"]):
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH,
            message=(
                f"a set of {len(op['indices'])} slices of width {width} on "
                f"'{op['parameter']}' needs {width * len(op['indices'])} values and "
                f"carries {len(op['values'])}"
            ),
        )
    for position, index in enumerate(op["indices"]):
        start = position * width
        slices[index] = _vector(op["values"][start : start + width], slices)


def _project_out(slices: torch.Tensor, op: WeightOp, width: int) -> None:
    """Remove the direction's component from each selected slice, in place.

    Args:
        slices: The parameter viewed as rows of slices.
        op: A ``project_out`` operation.
        width: The slice width.

    Raises:
        AppError: With ``EDIT_UPDATE_SHAPE_MISMATCH`` if the direction is not
            one slice wide.
    """
    if len(op["values"]) != width:
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH,
            message=(
                f"a direction removed from slices of width {width} on "
                f"'{op['parameter']}' must have {width} components, got {len(op['values'])}"
            ),
        )
    direction = _vector(op["values"], slices)
    unit = direction / torch.sqrt(torch.sum(direction * direction))
    for index in op["indices"]:
        row = slices[index]
        slices[index] = row - torch.sum(row * unit) * unit


def _elements(model: ParameterHostProto) -> int:
    """Count every weight in the network.

    Args:
        model: The weights.

    Returns:
        The total number of elements across all parameters.
    """
    return sum(int(parameter.detach().numel()) for _, parameter in model.named_parameters())


def apply_weight_program(model: ParameterHostProto, program: WeightProgram) -> AppliedProgram:
    """Apply a whole program, refusing one written for other weights.

    Args:
        model: The weights, changed in place.
        program: The program.

    Returns:
        What it did, and what it takes to undo it.

    Raises:
        AppError: With ``EDIT_VERIFICATION_FAILED`` if the live weights are not
            the ones the program was written against, or if a parameter the
            program did not name moved; or propagated from
            :func:`apply_op`.
    """
    digests_before = parameter_digests(model)
    before = weights_digest(digests_before)
    if before != program["written_against"]:
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_VERIFICATION_FAILED,
            message=(
                f"program '{program['program_id']}' was written against weights "
                f"{program['written_against']} and these are {before}; an edit read off "
                f"one set of weights is a different edit on any other"
            ),
        )
    written = tuple(sorted({op["parameter"] for op in program["ops"]}))
    snapshots = tuple(snapshot_parameter(model, name) for name in written)
    for op in program["ops"]:
        apply_op(model, op)
    digests_after = parameter_digests(model)
    strays = changed_parameters(digests_before, digests_after, frozenset(written))
    if strays:
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_VERIFICATION_FAILED,
            message=(
                f"program '{program['program_id']}' names {', '.join(written)} and "
                f"{', '.join(strays)} moved as well"
            ),
        )
    changed = sum(
        int(torch.sum(require_parameter(model, s["parameter_name"]).detach() != s["values"]).item())
        for s in snapshots
    )
    return AppliedProgram(
        effect=ProgramEffect(
            program_id=program["program_id"],
            weights_before=before,
            weights_after=weights_digest(digests_after),
            parameters_written=written,
            elements_changed=changed,
            elements_total=_elements(model),
        ),
        snapshots=snapshots,
        digests_before=digests_before,
    )


def remove_weight_program(model: ParameterHostProto, applied: AppliedProgram) -> None:
    """Restore every parameter a program wrote, and confirm the whole network.

    Args:
        model: The weights, restored in place.
        applied: What :func:`apply_weight_program` returned.

    Raises:
        AppError: Propagated from the per-parameter restore, or with
            ``EDIT_RESTORE_MISMATCH`` if any parameter's digest differs from
            before the program ran.
    """
    for snapshot in applied["snapshots"]:
        restore_weight(model, snapshot)
    moved = changed_parameters(applied["digests_before"], parameter_digests(model), frozenset())
    if moved:
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_RESTORE_MISMATCH,
            message=(
                f"after removing program '{applied['effect']['program_id']}', "
                f"{', '.join(moved)} still differ from before it ran"
            ),
        )


__all__ = [
    "AppliedProgram",
    "ProgramEffect",
    "apply_op",
    "apply_weight_program",
    "remove_weight_program",
]
