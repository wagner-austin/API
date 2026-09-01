"""Swap a GPT-2's matmul-bearing modules onto the ordered kernels.

The client-side twin of Model-Trainer's ``kernel_arm_modules``: the same two
wrapper shapes, the same original-parameters-by-reference discipline, the
same materialise-then-swap walk -- built on that module's own exported
Protocols so the two cannot drift apart in what they consider a Conv1D.
It lives here rather than there because Model-Trainer's suite is CPU-only
and these forwards exist only on CUDA.
"""

from __future__ import annotations

import torch
from model_trainer.core.services.model.kernel_arm_modules import (
    Conv1DProto,
    LinearProto,
    SwapTargetProto,
)

from ordered_kernels.api import ordered_addmm, ordered_matmul


def _conv1d_class() -> type[Conv1DProto]:
    """Return ``transformers.pytorch_utils.Conv1D``, typed for isinstance.

    The same dynamic-import dance ``kernel_arm_modules`` performs; restated
    here rather than imported because that module keeps its getters private,
    and eight lines of the established pattern beat a cross-package private
    import.
    """
    module = __import__("transformers.pytorch_utils", fromlist=["Conv1D"])
    cls: type[Conv1DProto] = module.Conv1D
    return cls


def _linear_class() -> type[LinearProto]:
    """Return ``torch.nn.Linear``, typed for isinstance."""
    module = __import__("torch.nn", fromlist=["Linear"])
    cls: type[LinearProto] = module.Linear
    return cls


class OrderedConv1D(torch.nn.Module):
    """``transformers`` ``Conv1D`` with its matmul on the ordered kernel."""

    def __init__(self, original: Conv1DProto) -> None:
        """Wrap one Conv1D, holding its ORIGINAL parameters by reference."""
        super().__init__()
        self.nf = original.nf
        self.weight = original.weight
        self.bias = original.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``bias + x @ weight``, flattened exactly as Conv1D does."""
        leading = [x.size(i) for i in range(x.dim() - 1)]
        flat = ordered_addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return flat.view(*leading, self.nf)


class OrderedLinear(torch.nn.Module):
    """A bias-free ``nn.Linear`` with its matmul on the ordered kernel."""

    def __init__(self, original: LinearProto) -> None:
        """Wrap one bias-free Linear."""
        super().__init__()
        self.weight = original.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``x @ weight.T`` by the ordered kernel."""
        weight = self.weight.t()
        leading = [x.size(i) for i in range(x.dim() - 1)]
        flat = ordered_matmul(x.view(-1, x.size(-1)), weight)
        return flat.view(*leading, weight.size(1))


def use_ordered_kernels(model: SwapTargetProto) -> int:
    """Replace every matmul-bearing module with its ordered version.

    Args:
        model: The model to rewrite in place.

    Returns:
        How many modules were replaced -- the caller's check that a swap
        that matched nothing cannot masquerade as a treated run.

    Raises:
        ValueError: If a ``Linear`` carries a bias, for the reason
            ``kernel_arm_modules`` refuses one: where ``F.linear`` adds its
            bias has not been measured.
    """
    conv1d_class = _conv1d_class()
    linear_class = _linear_class()
    graph = [(path, module) for path, module in model.named_modules() if path]
    replaced = 0
    for path, module in graph:
        if isinstance(module, conv1d_class):
            model.set_submodule(path, OrderedConv1D(module))
            replaced += 1
        elif isinstance(module, linear_class):
            if module.bias is not None:
                raise ValueError(
                    f"{path} is a Linear with a bias; the ordered swap only replaces "
                    "bias-free Linears, and where F.linear adds a bias has not been measured"
                )
            model.set_submodule(path, OrderedLinear(module))
            replaced += 1
    return replaced


__all__ = ["OrderedConv1D", "OrderedLinear", "use_ordered_kernels"]
