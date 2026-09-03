"""Check that an applied edit did what its record says, and nothing else.

Three questions, none of which requires knowing what the model means by
anything.

Did the weights move the way a rank-one update moves them? For any probe
input, the edited module's output must change by the value vector scaled by
the key's dot product with that probe, and by nothing else. The residual is
floating-point error or the edit is not what it claims.

Did the edit hit its target? At the key input the module's output must equal
the value the solve was given.

Did anything else move? Every other parameter's digest must be what it was.
This is the cheap half of the locality question and it is not the interesting
half: an edit that changes only the weight it names can still change what the
model answers about a thousand other things, which is why the item-level
before-and-after measurement exists. What this rules out is the mechanical
failure, a write that landed somewhere it was not aimed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from typing_extensions import TypedDict

from model_trainer.core.contracts.knowledge_edit import EditVerification
from model_trainer.core.services.model.editing.rank_one import predicted_output_delta
from model_trainer.core.services.model.tensor_digest import describe_tensor
from model_trainer.core.types import TracedLMModelProto


class AlgebraCheck(TypedDict):
    """How far an applied update sits from the rank-one form it claims.

    Attributes:
        max_prediction_error: Largest absolute elementwise difference between
            the measured output change and the predicted one, over every
            probe.
        key_output_error: Largest absolute elementwise difference between the
            module's output at the key input and the value the solve targeted.
    """

    max_prediction_error: float
    key_output_error: float


def as_input_output_weight(weight: torch.Tensor, transposed: bool) -> torch.Tensor:
    """View a stored weight in (input, output) orientation.

    Args:
        weight: The stored matrix.
        transposed: Whether the composed update had to be transposed to match
            it, which is the same question as whether the stored matrix is
            already (output, input).

    Returns:
        The matrix as (input, output), so ``x @ result`` is the module's own
        multiplication.
    """
    return weight.T if transposed else weight


def verify_update_algebra(
    *,
    before: torch.Tensor,
    after: torch.Tensor,
    transposed: bool,
    left: torch.Tensor,
    right: torch.Tensor,
    probes: Sequence[torch.Tensor],
    key_input: torch.Tensor,
    target_output: torch.Tensor,
) -> AlgebraCheck:
    """Measure an applied update against the rank-one identity.

    Args:
        before: The stored weight before the edit.
        after: The stored weight after it.
        transposed: Orientation flag from the edit's record.
        left: The key vector.
        right: The value vector.
        probes: Inputs to test the identity at. The key input does not need to
            be among them; it is checked separately and for a different
            property.
        key_input: The module's input at the edited position.
        target_output: What the solve was asked to make the module emit there.

    Returns:
        Both residuals.
    """
    before_io = as_input_output_weight(before, transposed)
    after_io = as_input_output_weight(after, transposed)

    worst = 0.0
    for probe in probes:
        measured = probe @ after_io - probe @ before_io
        predicted = predicted_output_delta(left=left, right=right, probe=probe)
        worst = max(worst, float(torch.max(torch.abs(measured - predicted)).item()))

    key_error = float(torch.max(torch.abs(key_input @ after_io - target_output)).item())
    return AlgebraCheck(max_prediction_error=worst, key_output_error=key_error)


def parameter_digests(model: TracedLMModelProto) -> dict[str, float]:
    """Digest every parameter in a model, by name.

    Args:
        model: The model to read.

    Returns:
        One folded digest per parameter name.
    """
    return {
        name: describe_tensor(parameter.detach())[0] for name, parameter in model.named_parameters()
    }


def changed_parameters(
    before: Mapping[str, float], after: Mapping[str, float], expected: str
) -> tuple[str, ...]:
    """Name the parameters whose digests moved, other than the edited one.

    A name present in one mapping and absent from the other counts as
    changed. A model that gained or lost a parameter during an edit is a
    stronger finding than one whose values moved, and reporting it as
    "unchanged" because there is nothing to compare against would hide it.

    Args:
        before: Digests taken before the edit.
        after: Digests taken after it.
        expected: The parameter the edit was aimed at, which is allowed to
            differ.

    Returns:
        Sorted names of every other parameter that differs.
    """
    names = set(before) | set(after)
    return tuple(
        sorted(name for name in names if name != expected and before.get(name) != after.get(name))
    )


def verify_rank_one_edit(
    *,
    module: str,
    before: torch.Tensor,
    after: torch.Tensor,
    transposed: bool,
    left: torch.Tensor,
    right: torch.Tensor,
    probes: Sequence[torch.Tensor],
    key_input: torch.Tensor,
    target_output: torch.Tensor,
    digests_before: Mapping[str, float],
    digests_after: Mapping[str, float],
) -> EditVerification:
    """Assemble the full evidence that one edit behaved.

    Args:
        module: The edited parameter's dotted path.
        before: Its values before the edit.
        after: Its values after.
        transposed: Orientation flag from the edit's record.
        left: The key vector.
        right: The value vector.
        probes: Inputs to test the rank-one identity at.
        key_input: The module's input at the edited position.
        target_output: What the solve targeted there.
        digests_before: Every parameter's digest before the edit.
        digests_after: Every parameter's digest after it.

    Returns:
        The verification record.
    """
    algebra = verify_update_algebra(
        before=before,
        after=after,
        transposed=transposed,
        left=left,
        right=right,
        probes=probes,
        key_input=key_input,
        target_output=target_output,
    )
    return EditVerification(
        module=module,
        max_prediction_error=algebra["max_prediction_error"],
        key_output_error=algebra["key_output_error"],
        other_parameters_changed=changed_parameters(digests_before, digests_after, module),
    )


__all__ = [
    "AlgebraCheck",
    "as_input_output_weight",
    "changed_parameters",
    "parameter_digests",
    "verify_rank_one_edit",
    "verify_update_algebra",
]
