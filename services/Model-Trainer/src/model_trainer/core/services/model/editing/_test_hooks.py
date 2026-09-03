"""Internal hooks for the knowledge-editing services.

Production binds these to real implementations at import. Tests assign a fake
and call :func:`reset_hooks` afterwards. No conditionals: call the hook.

There is exactly one seam here, and it exists for a reason that could not be
covered any other way. Capturing a module's activations means running a
forward pass with a hook attached, and on a real module that hook always
fires -- so the arm that reports a capture which did not happen is
unreachable with real weights. A fake forward that runs nothing reaches it,
which is the difference between a guard that is believed and a guard that is
checked.
"""

from __future__ import annotations

from typing import Protocol

import torch

from model_trainer.core.types import LMModelProto


class RunCaptureForwardProto(Protocol):
    """Protocol for running the forward pass a capture listens to."""

    def __call__(self, model: LMModelProto, input_ids: torch.Tensor) -> None:
        """Run one forward pass over the given tokens.

        Args:
            model: The model to run.
            input_ids: Token ids, shaped (batch, sequence).
        """
        ...


def _default_run_capture_forward(model: LMModelProto, input_ids: torch.Tensor) -> None:
    """Production implementation: one forward pass under no_grad.

    The labels argument is the model's own input. Nothing here reads the
    loss, and the protocol this package types models by declares ``forward``
    with both arguments, so passing the tokens twice is how a capture asks
    for a plain forward without widening that protocol for one caller.

    ``no_grad`` because a capture is a measurement. Building a graph over a
    model whose weights are about to be edited in place would hold the
    pre-edit values alive for no reader.

    Args:
        model: The model to run.
        input_ids: Token ids, shaped (batch, sequence).
    """
    with torch.no_grad():
        model.forward(input_ids=input_ids, labels=input_ids)


run_capture_forward: RunCaptureForwardProto = _default_run_capture_forward


def reset_hooks() -> None:
    """Restore every hook to the production implementation it is bound to."""
    global run_capture_forward
    run_capture_forward = _default_run_capture_forward


__all__ = [
    "RunCaptureForwardProto",
    "reset_hooks",
    "run_capture_forward",
]
