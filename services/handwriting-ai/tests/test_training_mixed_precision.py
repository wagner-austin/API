"""The fp16 mixed-precision path, exercised without a CUDA device.

These lines were the whole of this service's CI coverage gap. A gradient
scaler exists only when ``precision == "fp16" and device.type == "cuda"``,
so every line behind ``scaler is not None`` -- and the two hook defaults
that build the autocast context and the scaler -- ran on a machine with a
graphics card and nowhere else. The 100% gate was reporting on code no CPU
run had executed.

Nothing here needs a GPU:

  * ``torch.amp.autocast(device_type="cuda", ...)`` CONSTRUCTS anywhere; it
    only warns when entered without CUDA.
  * ``torch.amp.GradScaler()`` constructs anywhere too, disabling itself
    when there is no device.
  * :func:`apply_gradient` takes its scaler as an argument, so a fake plus
    real CPU tensors reaches both arms.

What is verified is ORDER AND WIRING -- ``unscale_`` before ``step``, the
optimiser stepped through the scaler rather than also directly, the scaler
consulted only when one exists. fp16 arithmetic is not verified here and is
not claimed to be.

Strict typing only: no Any, casts, or type: ignore.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from handwriting_ai._hook_defaults_training import (
    _default_create_grad_scaler,
    _default_get_autocast_context,
)
from handwriting_ai.training.loops import apply_gradient


class _RecordingScaler:
    """A gradient scaler that records the sequence it was driven in."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.scale_factor = 2.0

    def scale(self, loss: Tensor) -> Tensor:
        self.calls.append("scale")
        return loss * self.scale_factor

    def unscale_(self, optimizer: Optimizer) -> None:
        self.calls.append("unscale_")

    def step(self, optimizer: Optimizer) -> None:
        self.calls.append("step")

    def update(self) -> None:
        self.calls.append("update")


def _leaf_loss() -> tuple[Tensor, Tensor]:
    """Build a real CPU loss with a real gradient path.

    Returns:
        The parameter (value 1.0) and a scalar loss whose derivative with
        respect to it is exactly 3.0, so every assertion below can name a
        number rather than a shape.
    """
    param = torch.ones(1, requires_grad=True)
    loss = (param * 3.0).sum()
    return param, loss


def _grad_of(param: Tensor) -> float:
    """Read the gradient backward populated.

    Args:
        param: The leaf parameter.

    Returns:
        The gradient as a float.

    Raises:
        AssertionError: If backward populated no gradient at all, which is a
            different failure from populating the wrong one and should say
            so.
    """
    grad = param.grad
    if grad is None:
        raise AssertionError("backward populated no gradient")
    return float(grad.item())


def _sgd(param: Tensor) -> Optimizer:
    """A real optimiser with a step size that makes the arithmetic legible.

    torch's ``Optimizer`` is a concrete class rather than a protocol, so a
    double cannot stand in for it. That is fine and better: a real SGD on a
    real CPU parameter means "was it stepped" is answered by the
    parameter's value rather than by a call counter.

    Args:
        param: The parameter to optimise.

    Returns:
        SGD at lr=0.1, so one step on a gradient of 3.0 moves 1.0 to 0.7.
    """
    # Annotated so the list is list[Tensor] rather than list[Any]: SGD's
    # params argument is loosely typed, and this package forbids Any.
    params: list[Tensor] = [param]
    return torch.optim.SGD(params, lr=0.1)


class TestApplyGradientWithoutScaler:
    """The fp32/bf16 path: straight through the optimiser."""

    def test_backpropagates_and_steps_the_optimizer(self) -> None:
        """One step at lr=0.1 on a gradient of 3.0 takes 1.0 to 0.7."""
        param, loss = _leaf_loss()
        apply_gradient(scaler=None, optimizer=_sgd(param), loss=loss)
        assert float(param.item()) == pytest.approx(0.7)

    def test_gradient_is_unscaled(self) -> None:
        """d/dparam of (param * 3).sum() is 3, with no scaler in the path."""
        param, loss = _leaf_loss()
        apply_gradient(scaler=None, optimizer=_sgd(param), loss=loss)
        assert _grad_of(param) == 3.0


class TestApplyGradientWithScaler:
    """The fp16 path: through the scaler, in a specific order."""

    def test_drives_the_scaler_in_order(self) -> None:
        """scale -> unscale_ -> step -> update.

        The order is the contract. ``unscale_`` after ``step`` would hand
        the optimiser gradients still multiplied by the scale factor, which
        trains at the wrong rate and shows up as nothing but a worse curve.
        """
        param, loss = _leaf_loss()
        scaler = _RecordingScaler()
        apply_gradient(scaler=scaler, optimizer=_sgd(param), loss=loss)
        assert scaler.calls == ["scale", "unscale_", "step", "update"]

    def test_the_optimizer_is_not_also_stepped_directly(self) -> None:
        """Stepping both would apply two updates for one batch.

        The fake scaler's ``step`` is a no-op, so the parameter moving at
        all would mean this function stepped the optimiser itself as well.
        """
        param, loss = _leaf_loss()
        apply_gradient(scaler=_RecordingScaler(), optimizer=_sgd(param), loss=loss)
        assert float(param.item()) == 1.0

    def test_the_scaled_loss_is_what_backpropagates(self) -> None:
        """The scale factor reaches the gradient, which is the point of it."""
        param, loss = _leaf_loss()
        scaler = _RecordingScaler()
        apply_gradient(scaler=scaler, optimizer=_sgd(param), loss=loss)
        assert _grad_of(param) == 3.0 * scaler.scale_factor


class TestAutocastContextDefault:
    """The production hook that names the forward pass's context."""

    def test_fp32_gets_a_context_that_does_nothing(self) -> None:
        ctx = _default_get_autocast_context("fp32", torch.device("cpu"))
        body_ran = False
        with ctx:
            body_ran = True
        assert body_ran

    def test_fp16_builds_an_enterable_autocast(self) -> None:
        """Built for CUDA and entered here, on a machine that may have none.

        ``torch.device("cuda")`` is a descriptor, not an allocation, and
        autocast off CUDA warns rather than raising -- which is what lets
        this default be covered without a card in the machine.
        """
        ctx = _default_get_autocast_context("fp16", torch.device("cuda"))
        body_ran = False
        with ctx:
            body_ran = True
        assert body_ran

    def test_bf16_builds_an_enterable_autocast(self) -> None:
        ctx = _default_get_autocast_context("bf16", torch.device("cuda"))
        body_ran = False
        with ctx:
            body_ran = True
        assert body_ran


class TestGradScalerDefault:
    """The production hook that builds the scaler."""

    def test_constructs_a_real_grad_scaler(self) -> None:
        assert type(_default_create_grad_scaler()).__name__ == "GradScaler"
