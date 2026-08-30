"""Put a kernel arm inside a model, by replacing the modules that multiply.

WHY THIS EXISTS. The 2026-08-30 kernel-arm run answered its question about
fifty-nine ISOLATED matmuls: under a fixed reduction order four cards agree on
every one, including the shape the forward trace had named. That is a strong
indication and not a demonstration that a MODEL reproduces, and the page said
so. A forward pass also runs layer norms, a softmax and a GELU, each with its
own reduction and none of them touched by a GEMM probe. The only way to find
out is to run the model with its matmuls replaced and trace it.

WHAT IS REPLACED, AND WHAT IS DELIBERATELY NOT.

* Every ``Conv1D`` -- ``c_attn``, ``c_proj``, ``mlp.c_fc``, ``mlp.c_proj``,
  four per block. This is where the trace's first divergence lives:
  ``h.0.attn.c_attn`` at ``large`` and ``xl``.
* ``lm_head``, an ``nn.Linear(n_embd, vocab, bias=False)``. Included because
  it is the residual the sm_80+ cards were left with, and because it is the
  one matmul ``CUBLASLT_WORKSPACE_SIZE`` structurally cannot reach.
* NOT the two matmuls inside attention. Under the math pin, SDPA decomposes
  into ``torch.matmul`` calls that are not modules and so cannot be swapped
  this way. Leaving them is a measured decision rather than an oversight: the
  backend probe found the ``math`` digest to be ONE value across every card,
  and the four-card trace found attention never appearing as a first
  divergence once pinned. If attention were still carrying a difference, this
  trace will say so by naming it -- which is the point of running it rather
  than arguing about it.
* NOT the layer norms, the softmax or the GELU. Same reason in reverse:
  nothing here can fix them, and if one of them carries a difference the trace
  will name it and that is a finding.

WHY A MODULE SWAP RATHER THAN AN INTERCEPT. ``torch.overrides.TorchFunctionMode``
would catch every ``addmm`` including attention's, and it would do it by
intercepting a call the model makes rather than by changing the model. Two
reasons against it here. The record would then be unable to show that the arm
was applied -- every module class would read exactly as it does untreated --
whereas a swapped module puts its own class name into every observation the
trace writes, so the artifact carries its own evidence. And an intercept
applies to whatever else happens to run inside the block, which for a probe
that is trying to isolate one variable is the wrong default.

THE CLASS NAMES CHANGE, AND THAT IS THE POINT. A traced observation is named
``<rung>|<step>|<kind>|<index>|<module_class>|<path>``, so a swapped model
writes ``ArmConv1D`` where an untreated one writes ``Conv1D``. Records from
two different arms therefore do not share observation names and cannot be
compared name-for-name -- which is correct, because they did not compute the
same thing. Four cards under the SAME arm do share them, and that is the
comparison this exists for.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Protocol

import torch

from model_trainer.core.services.model.deterministic_gemm import (
    CUBLAS_ARM,
    gemm_by_arm,
    matmul_by_arm,
    require_kernel_arm,
)
from model_trainer.core.types import TracedModuleProto


class Conv1DProto(Protocol):
    """What this module needs from a ``transformers`` ``Conv1D``.

    ``transformers.pytorch_utils`` ships no type information, so importing
    ``Conv1D`` directly lands an untyped symbol in a package configured with
    ``disallow_any_unimported``. The dynamic-import-behind-a-Protocol shape is
    the one ``backends/gpt2/hf_gpt2.py`` already uses for ``GPT2Config`` and
    ``GPT2LMHeadModel``, and this follows it rather than inventing a second
    way to reach the same library.

    Attributes:
        nf: Output width.
        weight: ``[K, nf]``.
        bias: ``[nf]``.
    """

    @property
    def nf(self) -> int:
        """Output width."""
        ...

    @property
    def weight(self) -> torch.Tensor:
        """``[K, nf]``."""
        ...

    @property
    def bias(self) -> torch.Tensor:
        """``[nf]``."""
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``bias + x @ weight``, reshaping either side."""
        ...


class SwapTargetProto(Protocol):
    """The two members :func:`use_kernel_arm` needs from a model.

    Narrower than :class:`~model_trainer.core.types.TracedModuleProto` on
    purpose. That protocol also declares ``children`` and
    ``register_forward_hook`` with the signatures the forward trace wants, and
    a concrete ``torch.nn.Sequential`` does NOT match those -- torch's real
    hook signature is broader. Declaring only what is used lets both the
    traced model and a plain module tree be swapped, which is what makes the
    refusal path testable without building a language model to reach it.

    ``named_modules`` yields :class:`~model_trainer.core.types.TracedModuleProto`
    -- the name this codebase already uses for "a module in the graph" -- and
    every use here narrows it with ``isinstance``.
    """

    def named_modules(self) -> Generator[tuple[str, TracedModuleProto], None, None]:
        """Yield every module in the graph with its dotted path, root first."""
        ...

    def set_submodule(self, target: str, module: torch.nn.Module) -> None:
        """Replace the submodule at a dotted path."""
        ...


class LinearProto(Protocol):
    """What this module needs from a ``torch.nn.Linear``.

    Reached the same way ``Conv1D`` is, and for a related reason: under
    ``disallow_any_expr`` the expression ``torch.nn.Linear`` is itself typed
    ``type[Linear]`` containing ``Any``, so naming the class inline puts an
    ``Any`` in an ``isinstance``. Behind a Protocol the check is exact.

    Attributes:
        weight: ``[out, in]`` -- note the orientation is the transpose of
            ``Conv1D``'s, which is why :class:`ArmLinear` transposes.
        bias: ``None`` for the one this exists to replace.
    """

    @property
    def weight(self) -> torch.Tensor:
        """``[out, in]``."""
        ...

    @property
    def bias(self) -> torch.Tensor | None:
        """``None`` for the one this exists to replace."""
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``x @ weight.T``, plus a bias if there is one."""
        ...


def _linear_class() -> type[LinearProto]:
    """Return ``torch.nn.Linear``, typed.

    Returns:
        The class, for :func:`isinstance`.
    """
    module = __import__("torch.nn", fromlist=["Linear"])
    cls: type[LinearProto] = module.Linear
    return cls


def _conv1d_class() -> type[Conv1DProto]:
    """Return ``transformers.pytorch_utils.Conv1D``, typed.

    Returns:
        The class, for :func:`isinstance`.
    """
    module = __import__("transformers.pytorch_utils", fromlist=["Conv1D"])
    cls: type[Conv1DProto] = module.Conv1D
    return cls


class ArmConv1D(torch.nn.Module):
    """``transformers`` ``Conv1D``, with its matmul taken by a named arm.

    Holds the ORIGINAL parameter objects rather than copies, so the swapped
    model has the same weights bit-for-bit and a difference in the trace
    cannot be a difference in what was multiplied.

    Attributes:
        nf: Output width, as the original names it.
        arm: Which arithmetic, by :data:`~deterministic_gemm.KERNEL_ARMS` name.
    """

    def __init__(self, original: Conv1DProto, arm: str) -> None:
        """Wrap one ``Conv1D``.

        Args:
            original: The module being replaced.
            arm: One of :data:`~deterministic_gemm.KERNEL_ARMS`.

        Raises:
            ValueError: Propagated from
                :func:`~deterministic_gemm.require_kernel_arm`.
        """
        super().__init__()
        self.arm = require_kernel_arm(arm)
        self.nf = int(original.nf)
        self.weight = original.weight
        self.bias = original.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute what ``Conv1D.forward`` computes, by this module's arm.

        The reshape either side is copied from ``transformers`` 4.46.3
        ``pytorch_utils.Conv1D.forward`` deliberately: a probe that flattened
        differently would be measuring a different operation and the
        comparison against an untreated run would mean nothing.

        Args:
            x: ``[..., K]``.

        Returns:
            ``[..., nf]``.
        """
        leading = [x.size(i) for i in range(x.dim() - 1)]
        flat = gemm_by_arm(self.arm, self.bias, x.view(-1, x.size(-1)), self.weight)
        return flat.view(*leading, self.nf)


class ArmLinear(torch.nn.Module):
    """A bias-free ``nn.Linear``, with its matmul taken by a named arm.

    Bias-free only, because the one this exists for is ``lm_head`` and GPT-2
    declares it ``bias=False``. A biased ``Linear`` would need its bias added
    at the same point ``F.linear`` adds it, and asserting where that is
    without measuring it is the kind of guess this module is built to avoid.
    :func:`use_kernel_arm` refuses a biased one rather than assuming.

    Attributes:
        arm: Which arithmetic, by :data:`~deterministic_gemm.KERNEL_ARMS` name.
    """

    def __init__(self, original: LinearProto, arm: str) -> None:
        """Wrap one bias-free ``Linear``.

        Args:
            original: The module being replaced.
            arm: One of :data:`~deterministic_gemm.KERNEL_ARMS`.

        Raises:
            ValueError: Propagated from
                :func:`~deterministic_gemm.require_kernel_arm`.
        """
        super().__init__()
        self.arm = require_kernel_arm(arm)
        self.weight = original.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``x @ weight.T``, by this module's arm.

        ``Linear`` stores ``[out, in]`` and the arms take ``[in, out]``, so
        the weight is transposed rather than the input -- a view, not a copy,
        and the arms are written to accept a strided ``w``.

        Args:
            x: ``[..., in]``.

        Returns:
            ``[..., out]``.
        """
        weight = self.weight.t()
        leading = [x.size(i) for i in range(x.dim() - 1)]
        flat = matmul_by_arm(self.arm, x.view(-1, x.size(-1)), weight)
        return flat.view(*leading, weight.size(1))


def use_kernel_arm(model: SwapTargetProto, arm: str) -> int:
    """Replace every matmul-bearing module in ``model`` with the arm's version.

    Mutates ``model`` in place and returns the count, because the count is
    what a caller checks: a swap that silently matched nothing would produce a
    record labelled with an arm it did not run, which is the defect every
    other arm on this command already refuses.

    :data:`~deterministic_gemm.CUBLAS_ARM` is a no-op by construction. It is
    the untreated path, and rebuilding it out of wrapper modules would change
    the class names in the record and make it incomparable with every trace
    taken before today for no gain.

    The graph is materialised BEFORE anything is replaced. ``named_modules``
    walks the live tree, and swapping during the walk would hand the iterator
    modules that are no longer attached to anything.

    Args:
        model: The model to rewrite.
        arm: One of :data:`~deterministic_gemm.KERNEL_ARMS`.

    Returns:
        How many modules were replaced. Zero for the cuBLAS arm.

    Raises:
        ValueError: For an unknown arm, or if a ``Linear`` carries a bias --
            see :class:`ArmLinear` for why that case is refused rather than
            guessed at.
    """
    named = require_kernel_arm(arm)
    if named == CUBLAS_ARM:
        return 0

    conv1d_class = _conv1d_class()
    linear_class = _linear_class()
    graph = [(path, module) for path, module in model.named_modules() if path]

    replaced = 0
    for path, module in graph:
        if isinstance(module, conv1d_class):
            model.set_submodule(path, ArmConv1D(module, named))
            replaced += 1
        elif isinstance(module, linear_class):
            if module.bias is not None:
                raise ValueError(
                    f"{path} is a Linear with a bias; this arm only replaces bias-free "
                    "Linears, and where F.linear adds a bias has not been measured"
                )
            model.set_submodule(path, ArmLinear(module, named))
            replaced += 1
    return replaced


__all__ = [
    "ArmConv1D",
    "ArmLinear",
    "Conv1DProto",
    "LinearProto",
    "SwapTargetProto",
    "use_kernel_arm",
]
