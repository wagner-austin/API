"""Find the output a module must emit for the model to say the target.

THE PIECE THE RANK-ONE ARITHMETIC CANNOT SUPPLY. :mod:`rank_one` solves for
the value vector once somebody names ``target_output`` -- what the edited
module should emit at the keyed position. Nothing derives that from the
target string, because the map from "the model should continue with
``ClearGBM``" to "this 768-vector at layer 6" is the model's own function and
has no closed form. Locate-then-edit methods find it by gradient descent on
that one vector with every weight frozen, and so does this.

WHAT IS OPTIMISED, AND WHAT IS NOT. One vector, added to the module's output
at ONE token position. The model's parameters are frozen before the first
step and are not restored to trainable afterwards, deliberately: this runs
inside an arm that edits weights in place, and a base left quietly trainable
is how a later stage trains what it meant to read.

THE SWAP RATHER THAN A HOOK. torch's forward hooks can substitute a module's
output, but this repository's
:class:`~model_trainer.core.types.ForwardHookProto` returns ``None`` and its
docstring says it observes and never substitutes. Widening that protocol for
one caller would weaken every hook in the codebase. Replacing the module for
the duration is the mechanism the kernel arms already use for the same
problem (:mod:`~model_trainer.core.services.model.kernel_arm_modules`), it
goes through the typed ``set_submodule``, and it is put back in a ``finally``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.services.model.editing.sites import require_edit_module
from model_trainer.core.types import TracedLMModelProto

#: Ignored label id. What torch's cross entropy skips, and what marks every
#: position whose prediction this optimisation is not about.
IGNORED_LABEL = -100


def module_class() -> type[torch.nn.Module]:
    """Return ``torch.nn.Module``, typed, for an ``isinstance`` narrowing.

    Reached through ``__import__`` for the reason
    :func:`~model_trainer.core.services.model.kernel_arm_modules.linear_class`
    is: the bare attribute expression is untyped, and this repository refuses
    an ``Any`` even inside an ``isinstance``. Annotating the binding is what
    turns it back into a type.

    Returns:
        The class.
    """
    module = __import__("torch.nn", fromlist=["Module"])
    cls: type[torch.nn.Module] = module.Module
    return cls


class DeltaAtPosition(torch.nn.Module):
    """A module, plus a learnable vector added to its output at one position.

    The original is held as a registered submodule rather than a plain
    attribute, so the model's parameter list is unchanged while the swap is
    installed. A vanished parameter would be a difference this arm did not
    intend and could not see.

    Attributes:
        delta: The vector being learned. The only trainable tensor in the
            model while this is installed.
        position: Which token of the sequence it is added at.
    """

    delta: torch.nn.Parameter
    position: int

    def __init__(self, original: torch.nn.Module, *, width: int, position: int) -> None:
        """Wrap one module.

        Args:
            original: The module being stood in for.
            width: Output width, which is the length of the learned vector.
            position: Which token position to add it at.
        """
        super().__init__()
        self.original = original
        self.position = position
        self.delta = torch.nn.Parameter(torch.zeros(width))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the original and add the delta at one position.

        Written as an addition of a selector-scaled outer product rather than
        an indexed assignment: assigning into a tensor that carries gradient
        history is a mutation autograd has to undo, and the functional form
        says the same thing without one.

        Args:
            hidden: What the original module receives, shaped
                ``(batch, sequence, width)``.

        Returns:
            The original's output with the delta added at ``position``.
        """
        out: torch.Tensor = self.original(hidden)
        selector = torch.zeros(out.shape[1], 1, dtype=out.dtype, device=out.device)
        selector[self.position, 0] = 1.0
        return out + selector * self.delta


def target_labels(*, prompt_length: int, target_ids: Sequence[int]) -> torch.Tensor:
    """Build labels that score the target tokens and nothing else.

    Args:
        prompt_length: How many tokens precede the target.
        target_ids: The target's token ids.

    Returns:
        Labels shaped ``(1, prompt_length + len(target_ids))``, every prompt
        position :data:`IGNORED_LABEL`.
    """
    rows: list[list[int]] = [[IGNORED_LABEL] * prompt_length + list(target_ids)]
    return torch.tensor(rows, dtype=torch.long)


def target_token_nll(
    *,
    model: TracedLMModelProto,
    prompt_ids: Sequence[int],
    target_ids: Sequence[int],
    device: str,
) -> float:
    """Total surprise the model assigns to the TARGET tokens alone.

    HERE RATHER THAN IN THE SCORER, because it shares :func:`target_labels`
    with the optimisation above and asks the identical question: how likely is
    this continuation, given this prompt. A second implementation of the mask
    would be free to disagree with the one being optimised against.

    AND TARGET-ONLY RATHER THAN WHOLE-SEQUENCE, which is not a refinement but
    a correction. Scoring ``prompt + target`` as one string was the first
    version, and it reported every edit as a large REGRESSION -- 82 to 203
    across thirteen edits -- while the same edits were driving their targets'
    own likelihood to near certainty. A rank-one edit at one position changes
    the model's predictions for everything downstream of it, including the
    remaining PROMPT tokens, and a whole-sequence total is dominated by that
    collateral rather than by the association the edit wrote. Both numbers are
    worth having and they are different measurements; this one is the edit's.

    Args:
        model: The model to ask.
        prompt_ids: Token ids of the rendered prompt.
        target_ids: Token ids of the wanted continuation.
        device: Device to run on.

    Returns:
        Summed negative log-likelihood over the target's tokens.
    """
    sequence: list[list[int]] = [list(prompt_ids) + list(target_ids)]
    input_ids = torch.tensor(sequence, dtype=torch.long).to(device)
    labels = target_labels(prompt_length=len(prompt_ids), target_ids=target_ids).to(device)
    with torch.no_grad():
        mean = float(model.forward(input_ids=input_ids, labels=labels).loss.item())
    # The model reports a MEAN over the positions it scored, and the scored
    # positions are exactly the target's, so the total is that mean times the
    # target's length -- the same multiplication `cloze.score` does for the
    # same reason.
    return mean * float(len(target_ids))


def optimise_target_output(
    *,
    model: TracedLMModelProto,
    module_name: str,
    prompt_ids: Sequence[int],
    target_ids: Sequence[int],
    position: int,
    current_output: torch.Tensor,
    steps: int,
    learning_rate: float,
    device: str,
) -> torch.Tensor:
    """Learn what the module must emit for the model to continue with the target.

    Args:
        model: The model, whose parameters are frozen here and left frozen.
        module_name: Dotted path of the module to edit.
        prompt_ids: Token ids of the rendered prompt.
        target_ids: Token ids of the wanted continuation.
        position: Which token of the prompt the fact is keyed on, indexed
            from the start of the sequence.
        current_output: What the module emits there now, one dimensional.
        steps: Optimisation steps. Fixed rather than early-stopped on a
            threshold, so two runs of one plan spend the same compute and the
            record's cost line means the same thing in both.
        learning_rate: AdamW step size for the delta.
        device: Device to run on.

    Returns:
        ``current_output`` plus the learned delta, detached, ready for
        :func:`~model_trainer.core.services.model.editing.rank_one.solve_right_vector`.

    Raises:
        AppError: With ``EDIT_MODULE_NOT_FOUND`` if the module does not exist,
            or ``EDIT_UPDATE_SHAPE_MISMATCH`` if the module cannot be swapped,
            if ``current_output`` is not one dimensional, or if the target is
            empty -- an empty target names no continuation to optimise
            towards, and the loss over zero scored positions is not a number.
    """
    if current_output.dim() != 1:
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH,
            message=(
                f"current_output must be one dimensional, got shape "
                f"{tuple(current_output.shape)}; it is the module's emission at one "
                f"token, not a sequence of them"
            ),
        )
    if not target_ids:
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH,
            message=(
                "the target encodes to no tokens, so there is nothing for the delta to "
                "make more likely and the loss would average over an empty set"
            ),
        )

    original = require_edit_module(model, module_name)
    if not isinstance(original, module_class()):
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH,
            message=(
                f"module '{module_name}' is not a torch module, so it cannot be stood in "
                f"for while a value vector is learned"
            ),
        )

    for _, parameter in model.named_parameters():
        parameter.requires_grad = False

    sequence: list[list[int]] = [list(prompt_ids) + list(target_ids)]
    input_ids = torch.tensor(sequence, dtype=torch.long).to(device)
    labels = target_labels(prompt_length=len(prompt_ids), target_ids=target_ids).to(device)
    wrapper = DeltaAtPosition(original, width=int(current_output.shape[0]), position=position).to(
        device
    )
    optimiser = torch.optim.AdamW([wrapper.delta], lr=learning_rate)

    model.set_submodule(module_name, wrapper)
    try:
        for _ in range(steps):
            optimiser.zero_grad()
            model.forward(input_ids=input_ids, labels=labels).loss.backward()
            optimiser.step()
    finally:
        model.set_submodule(module_name, original)
    return (current_output + wrapper.delta.detach().to(current_output.device)).detach()


__all__ = [
    "IGNORED_LABEL",
    "DeltaAtPosition",
    "module_class",
    "optimise_target_output",
    "target_labels",
    "target_token_nll",
]
