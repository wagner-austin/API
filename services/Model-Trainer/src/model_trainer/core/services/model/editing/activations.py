"""Read what one module sees and emits at one token position.

An edit needs two vectors from the live model before it can be solved: what
the edited module receives at the token the fact is keyed on, and what it
currently produces there. Both are read with a forward hook on the real
module during a real forward pass, because the alternative -- recomputing the
module's input from its inputs -- would be a second implementation of the
model's own arithmetic, free to disagree with it.

WHY ONE SEQUENCE AT A TIME. The position argument indexes tokens, and a batch
would make it index tokens of an unstated sequence. Capturing one sequence per
call keeps the returned vectors unambiguous; a caller with many prompts calls
many times, which is also how the cost is visible rather than hidden.
"""

from __future__ import annotations

import torch
from platform_core.errors import AppError, ModelTrainerErrorCode
from typing_extensions import TypedDict

from model_trainer.core.services.model.editing import _test_hooks
from model_trainer.core.services.model.editing.sites import require_edit_module
from model_trainer.core.types import HookValue, TracedLMModelProto, TracedModuleProto


class CapturedActivation(TypedDict):
    """One module's input and output at one token position.

    Attributes:
        module_input: What the module received, one dimensional.
        module_output: What it emitted, one dimensional.
    """

    module_input: torch.Tensor
    module_output: torch.Tensor


def capture_module_io(
    *,
    model: TracedLMModelProto,
    module_name: str,
    input_ids: torch.Tensor,
    position: int,
) -> CapturedActivation:
    """Run one forward pass and read a module's input and output at a token.

    Args:
        model: The model to run.
        module_name: Dotted path of the module to listen to.
        input_ids: Token ids for exactly one sequence, shaped (1, sequence).
        position: Which token to read, negative counting from the end.

    Returns:
        The module's input and output vectors at that position, detached.

    Raises:
        AppError: With ``EDIT_MODULE_NOT_FOUND`` if the module does not exist.
            With ``EDIT_UPDATE_SHAPE_MISMATCH`` if the tokens are not one
            sequence, or the position lies outside it. With
            ``EDIT_ACTIVATION_NOT_CAPTURED`` if the forward pass did not run
            the module, or ran it with something other than tensors.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH,
            message=(
                f"capture expects exactly one sequence shaped (1, tokens), got "
                f"{tuple(input_ids.shape)}; a position index would otherwise name "
                f"a token of an unstated sequence"
            ),
        )
    sequence_len = int(input_ids.shape[1])
    if not -sequence_len <= position < sequence_len:
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH,
            message=(f"position {position} lies outside a sequence of {sequence_len} tokens"),
        )

    module = require_edit_module(model, module_name)
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _record(
        hooked: TracedModuleProto, args: tuple[HookValue, ...], output: HookValue, /
    ) -> None:
        """Record one call's input and output tensors.

        Args:
            hooked: The module that ran. Unread: the hook is attached to one
                module, so its identity is already known.
            args: The module's positional arguments.
            output: What it returned.

        Raises:
            AppError: With ``EDIT_ACTIVATION_NOT_CAPTURED`` if the first
                argument or the output is not a tensor.
        """
        first = args[0] if args else None
        if not torch.is_tensor(first) or not torch.is_tensor(output):
            raise AppError(
                code=ModelTrainerErrorCode.EDIT_ACTIVATION_NOT_CAPTURED,
                message=(
                    f"module '{module_name}' ran with a non-tensor input or output, so "
                    f"there is no activation to key an edit on"
                ),
            )
        seen.append((first.detach(), output.detach()))

    handle = module.register_forward_hook(_record)
    _test_hooks.run_capture_forward(model, input_ids)
    handle.remove()

    if not seen:
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_ACTIVATION_NOT_CAPTURED,
            message=(
                f"module '{module_name}' did not run during the forward pass, so no "
                f"activation was captured; the module exists but this input does not "
                f"reach it"
            ),
        )
    captured_input, captured_output = seen[-1]
    return CapturedActivation(
        module_input=captured_input[0, position],
        module_output=captured_output[0, position],
    )


__all__ = [
    "CapturedActivation",
    "capture_module_io",
]
