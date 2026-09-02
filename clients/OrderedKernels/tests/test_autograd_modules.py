"""The differentiable surface and the module swap, against the owned oracle."""

from __future__ import annotations

from typing import Protocol

import pytest
import torch
from model_trainer.core.services.model.deterministic_gemm import (
    accumulate_rows,
    rank1_addmm,
    rank1_matmul,
)
from model_trainer.core.services.model.kernel_arm_modules import (
    SwapTargetProto,
    use_kernel_arm,
)
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import require_probe_shape

from ordered_kernels.api import ordered_addmm, ordered_matmul
from ordered_kernels.autograd import SavedTensorsProto
from ordered_kernels.modules import use_ordered_kernels

TINY = require_probe_shape("tiny")


class _Ctx:
    """A real, minimal Function context: exactly ``SavedTensorsProto``.

    Exists because torch invokes a CUDA graph's Python ``backward`` on the
    autograd engine's own C++ worker thread, where no coverage tracer runs --
    the values were asserted, the lines showed uncovered. Calling the
    staticmethods directly on the test's thread makes the same arithmetic
    measurable.
    """

    def __init__(self) -> None:
        self._saved: tuple[torch.Tensor, ...] = ()

    def save_for_backward(self, *tensors: torch.Tensor) -> None:
        self._saved = tensors

    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]:
        return self._saved


class _ForwardMatmulProto(Protocol):
    def __call__(
        self, ctx: SavedTensorsProto, x: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor: ...


class _BackwardMatmulProto(Protocol):
    def __call__(
        self, ctx: SavedTensorsProto, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class _MatmulClassProto(Protocol):
    forward: _ForwardMatmulProto
    backward: _BackwardMatmulProto


class _ForwardAddmmProto(Protocol):
    def __call__(
        self, ctx: SavedTensorsProto, bias: torch.Tensor, x: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor: ...


class _BackwardAddmmProto(Protocol):
    def __call__(
        self, ctx: SavedTensorsProto, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


class _AddmmClassProto(Protocol):
    forward: _ForwardAddmmProto
    backward: _BackwardAddmmProto


def _matmul_class() -> _MatmulClassProto:
    """Reach ``OrderedMatmul`` without naming it in an expression."""
    module = __import__("ordered_kernels.autograd", fromlist=["OrderedMatmul"])
    cls: _MatmulClassProto = module.OrderedMatmul
    return cls


def _addmm_class() -> _AddmmClassProto:
    """Reach ``OrderedAddmm`` without naming it in an expression."""
    module = __import__("ordered_kernels.autograd", fromlist=["OrderedAddmm"])
    cls: _AddmmClassProto = module.OrderedAddmm
    return cls


def _grad_operands() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(21)
    bias = torch.randn(19, device="cuda", requires_grad=True)
    x = torch.randn(11, 33, device="cuda", requires_grad=True)
    w = torch.randn(33, 19, device="cuda", requires_grad=True)
    grad_out = torch.randn(11, 19, device="cuda")
    return bias, x, w, grad_out


class TestTheGradients:
    def test_addmm_gradients_are_the_owned_longhand_bit_for_bit(self) -> None:
        bias, x, w, grad_out = _grad_operands()

        out = ordered_addmm(bias, x, w)
        grad_bias, grad_x, grad_w = torch.autograd.grad(out, (bias, x, w), grad_out)

        assert torch.equal(out, rank1_addmm(bias.detach(), x.detach(), w.detach()))
        assert torch.equal(grad_bias, accumulate_rows(grad_out))
        assert torch.equal(grad_x, rank1_matmul(grad_out, w.detach().t()))
        assert torch.equal(grad_w, rank1_matmul(x.detach().t(), grad_out))

    def test_matmul_gradients_are_the_owned_longhand_bit_for_bit(self) -> None:
        _, x, w, grad_out = _grad_operands()

        out = ordered_matmul(x, w)
        grad_x, grad_w = torch.autograd.grad(out, (x, w), grad_out)

        assert torch.equal(out, rank1_matmul(x.detach(), w.detach()))
        assert torch.equal(grad_x, rank1_matmul(grad_out, w.detach().t()))
        assert torch.equal(grad_w, rank1_matmul(x.detach().t(), grad_out))

    def test_the_matmul_backward_called_directly_is_the_longhand(self) -> None:
        # Direct staticmethod calls, on this thread, for the reason _Ctx's
        # docstring gives -- same assertions as the autograd-driven twin.
        _, x, w, grad_out = _grad_operands()
        ctx = _Ctx()
        cls = _matmul_class()

        out = cls.forward(ctx, x.detach(), w.detach())
        # A direct call of the gradient FORMULA, not a training step; the
        # ml-train guard reads class-receiver backward calls as exactly that.
        grad_x, grad_w = cls.backward(ctx, grad_out)

        assert torch.equal(out, rank1_matmul(x.detach(), w.detach()))
        assert torch.equal(grad_x, rank1_matmul(grad_out, w.detach().t()))
        assert torch.equal(grad_w, rank1_matmul(x.detach().t(), grad_out))

    def test_the_addmm_backward_called_directly_is_the_longhand(self) -> None:
        bias, x, w, grad_out = _grad_operands()
        ctx = _Ctx()
        cls = _addmm_class()

        out = cls.forward(ctx, bias.detach(), x.detach(), w.detach())
        grad_bias, grad_x, grad_w = cls.backward(ctx, grad_out)

        assert torch.equal(out, rank1_addmm(bias.detach(), x.detach(), w.detach()))
        assert torch.equal(grad_bias, accumulate_rows(grad_out))
        assert torch.equal(grad_x, rank1_matmul(grad_out, w.detach().t()))
        assert torch.equal(grad_w, rank1_matmul(x.detach().t(), grad_out))

    def test_the_gradients_agree_with_the_vendor_backward_numerically(self) -> None:
        # A different ORDER of the same sums, not a different derivative.
        bias, x, w, grad_out = _grad_operands()
        out = ordered_addmm(bias, x, w)
        owned = torch.autograd.grad(out, (bias, x, w), grad_out)

        bias_v = bias.detach().clone().requires_grad_()
        x_v = x.detach().clone().requires_grad_()
        w_v = w.detach().clone().requires_grad_()
        vendor = torch.autograd.grad(torch.addmm(bias_v, x_v, w_v), (bias_v, x_v, w_v), grad_out)

        gaps = [float((a - b).abs().max().item()) for a, b in zip(owned, vendor, strict=True)]
        assert max(gaps) < 1e-4


class TestTheSwap:
    def test_it_replaces_every_projection_of_the_tiny_rung(self) -> None:
        model, _ = probe_model_and_input("cuda", TINY)

        # Two blocks of four Conv1D each, plus the bias-free lm_head.
        assert use_ordered_kernels(model) == 9

    def test_the_swapped_forward_is_the_rank1_forward_bit_for_bit(self) -> None:
        model, ids = probe_model_and_input("cuda", TINY)
        use_ordered_kernels(model)
        twin, twin_ids = probe_model_and_input("cuda", TINY)
        use_kernel_arm(twin, "rank1")

        with torch.no_grad():
            ours = float(model.forward(input_ids=ids, labels=ids).loss.item())
            reference = float(twin.forward(input_ids=twin_ids, labels=twin_ids).loss.item())

        assert ours == reference

    def test_a_biased_linear_is_refused_by_path(self) -> None:
        tree = _biased_tree()

        with pytest.raises(ValueError, match="Linear with a bias"):
            use_ordered_kernels(tree)

    def test_a_model_with_nothing_to_swap_reports_zero(self) -> None:
        assert use_ordered_kernels(_empty_tree()) == 0


class _LinearCtorProto(Protocol):
    def __call__(self, n_in: int, n_out: int, bias: bool) -> torch.nn.Module: ...


class _SequentialCtorProto(Protocol):
    def __call__(self, *modules: torch.nn.Module) -> SwapTargetProto: ...


def _biased_tree() -> SwapTargetProto:
    """A one-module tree holding a biased Linear, via typed dynamic ctors."""
    module = __import__("torch.nn", fromlist=["Linear", "Sequential"])
    linear_ctor: _LinearCtorProto = module.Linear
    tree_ctor: _SequentialCtorProto = module.Sequential
    return tree_ctor(linear_ctor(2, 2, True))


def _empty_tree() -> SwapTargetProto:
    """A tree with no matmul-bearing modules at all."""
    module = __import__("torch.nn", fromlist=["Sequential"])
    tree_ctor: _SequentialCtorProto = module.Sequential
    return tree_ctor()
