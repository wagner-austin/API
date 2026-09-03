"""Install a rank-one update in a model's weights, and take it back out.

An edit is a VALUE here, not a mutation someone remembers making. Applying one
returns a record of what was written, and the snapshot taken beforehand is the
only thing that can undo it. That shape is deliberate: an experiment that
edits, measures, restores and edits again has to be able to prove the model it
measured the second time was the same model it measured the first time, and a
restore that silently half-worked would make every later number a different
experiment's.

WHY RESTORE VERIFIES ITSELF. ``copy_`` between tensors of different dtype or
device converts rather than refusing, so a snapshot taken from one and written
back to another can return a model that is close to the original and not the
original. The check after the write costs one comparison and turns that into a
named failure.
"""

from __future__ import annotations

import torch
from platform_core.errors import AppError, ModelTrainerErrorCode
from typing_extensions import TypedDict

from model_trainer.core.contracts.knowledge_edit import EditSite, RankOneEditRecord
from model_trainer.core.services.model.editing.rank_one import (
    compose_rank_one,
    orient_for_weight,
)
from model_trainer.core.services.model.editing.sites import (
    require_editable_weight,
    weight_parameter_name,
)
from model_trainer.core.services.model.tensor_digest import describe_tensor
from model_trainer.core.types import TracedLMModelProto


class WeightSnapshot(TypedDict):
    """One parameter's values, detached from the model that held them.

    Not a JSON contract. It exists in memory for the length of one edit and
    holds a tensor, which is exactly why it is here and not beside the
    serialisable records.

    Attributes:
        parameter_name: The dotted path the values came from.
        values: A clone, so later edits to the live parameter cannot reach it.
    """

    parameter_name: str
    values: torch.Tensor


def snapshot_weight(model: TracedLMModelProto, parameter_name: str) -> WeightSnapshot:
    """Clone a parameter's current values.

    Args:
        model: The model to read from.
        parameter_name: Dotted parameter path.

    Returns:
        The snapshot.

    Raises:
        AppError: With ``EDIT_MODULE_NOT_FOUND`` if the parameter does not
            exist, or ``EDIT_WEIGHT_NOT_MATRIX`` if it is not a matrix.
    """
    parameter = require_editable_weight(model, parameter_name)
    return WeightSnapshot(
        parameter_name=parameter_name,
        values=parameter.detach().clone(),
    )


def apply_rank_one_edit(
    *,
    model: TracedLMModelProto,
    site: EditSite,
    item_id: str,
    left: torch.Tensor,
    right: torch.Tensor,
    denominator: float,
) -> RankOneEditRecord:
    """Add one rank-one update to the weight the site names.

    Args:
        model: The model to edit, changed in place.
        site: Where to write.
        item_id: The request this edit satisfies, carried onto the record.
        left: Key vector, in the module's input space.
        right: Value vector, in the module's output space.
        denominator: The divisor the value solve used, recorded so a reader
            can see how close to degenerate the edit was.

    Returns:
        What was written.

    Raises:
        AppError: With ``EDIT_MODULE_NOT_FOUND`` or ``EDIT_WEIGHT_NOT_MATRIX``
            if the site does not resolve to an editable matrix, or
            ``EDIT_UPDATE_SHAPE_MISMATCH`` if the composed update fits neither
            orientation of that matrix.
    """
    parameter_name = weight_parameter_name(site)
    parameter = require_editable_weight(model, parameter_name)
    oriented = orient_for_weight(compose_rank_one(left, right), parameter.shape)
    with torch.no_grad():
        parameter.copy_(parameter.detach() + oriented["matrix"])
    left_digest, _ = describe_tensor(left)
    right_digest, _ = describe_tensor(right)
    rows, cols = parameter.shape[0], parameter.shape[1]
    return RankOneEditRecord(
        item_id=item_id,
        module=parameter_name,
        weight_rows=int(rows),
        weight_cols=int(cols),
        transposed=oriented["transposed"],
        left_digest=left_digest,
        right_digest=right_digest,
        left_norm=euclidean_norm(left),
        right_norm=euclidean_norm(right),
        denominator=denominator,
        update_norm=euclidean_norm(oriented["matrix"]),
    )


def euclidean_norm(tensor: torch.Tensor) -> float:
    """Return the square root of the sum of squares, over every element.

    Spelled out rather than called through ``torch.linalg``, whose stubs
    return ``Any``. This package types every expression, and a norm that
    arrives untyped would put an unchecked value straight onto a record other
    runs are compared against. The formula is the Euclidean norm for a vector
    and the Frobenius norm for a matrix, which is the same computation and is
    why one function serves both call sites.

    Args:
        tensor: The tensor to measure.

    Returns:
        Its norm.
    """
    return float(torch.sqrt(torch.sum(tensor * tensor)).item())


def restore_weight(model: TracedLMModelProto, snapshot: WeightSnapshot) -> None:
    """Put a snapshot's values back, and confirm they arrived.

    Args:
        model: The model to write to.
        snapshot: The values to restore.

    Raises:
        AppError: With ``EDIT_MODULE_NOT_FOUND`` or ``EDIT_WEIGHT_NOT_MATRIX``
            if the parameter is no longer resolvable, with
            ``EDIT_UPDATE_SHAPE_MISMATCH`` if the snapshot's shape no longer
            matches the parameter's, or with ``EDIT_RESTORE_MISMATCH`` if the
            parameter does not hold the snapshot's exact bytes afterwards.
    """
    parameter = require_editable_weight(model, snapshot["parameter_name"])
    if parameter.shape != snapshot["values"].shape:
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH,
            message=(
                f"snapshot of '{snapshot['parameter_name']}' has shape "
                f"{tuple(snapshot['values'].shape)} and the live parameter has "
                f"{tuple(parameter.shape)}; this snapshot came from a different model"
            ),
        )
    with torch.no_grad():
        parameter.copy_(snapshot["values"])
    if not torch.equal(parameter.detach(), snapshot["values"]):
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_RESTORE_MISMATCH,
            message=(
                f"parameter '{snapshot['parameter_name']}' does not hold the snapshot's "
                f"bytes after restore; the model in memory is no longer the one the "
                f"pre-edit measurement was taken on, so nothing measured after this "
                f"can be compared with it"
            ),
        )


__all__ = [
    "WeightSnapshot",
    "apply_rank_one_edit",
    "euclidean_norm",
    "restore_weight",
    "snapshot_weight",
]
