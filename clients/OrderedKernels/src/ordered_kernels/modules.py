"""Swap a GPT-2's matmul-bearing modules onto the ordered kernels.

The client-side twin of Model-Trainer's ``kernel_arm_modules``: the same two
wrapper shapes, the same original-parameters-by-reference discipline, the
same materialise-then-swap walk -- built on that module's own exported
Protocols and class getters so the two cannot drift apart in what they
consider a Conv1D. It lives here rather than there because Model-Trainer's
suite is CPU-only and these forwards exist only on CUDA.
"""

from __future__ import annotations

from typing import Protocol

import torch
from model_trainer.core.services.model.kernel_arm_modules import (
    Conv1DProto,
    LinearProto,
    SwapTargetProto,
    conv1d_class,
    linear_class,
)

from ordered_kernels.api import ordered_addmm, ordered_matmul
from ordered_kernels.attention import ordered_causal_attention
from ordered_kernels.torch_surface import split_three


class _ProjectionProto(Protocol):
    """A projection module as the attention wrapper calls it."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class SdpaAttentionProto(Protocol):
    """The members of ``GPT2SdpaAttention`` the ordered wrapper takes over."""

    c_attn: _ProjectionProto
    c_proj: _ProjectionProto

    @property
    def num_heads(self) -> int: ...
    @property
    def head_dim(self) -> int: ...
    @property
    def split_size(self) -> int: ...
    @property
    def embed_dim(self) -> int: ...


def sdpa_attention_class() -> type[SdpaAttentionProto]:
    """Return ``transformers``' ``GPT2SdpaAttention``, typed for isinstance."""
    module = __import__("transformers.models.gpt2.modeling_gpt2", fromlist=["GPT2SdpaAttention"])
    cls: type[SdpaAttentionProto] = module.GPT2SdpaAttention
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


class OrderedSdpaAttention(torch.nn.Module):
    """GPT-2's SDPA attention with all three of its reductions owned.

    Mirrors ``GPT2SdpaAttention.forward``'s scoring path -- the split, the
    head reshape, the merge, the output projection, in its exact order --
    with the attention core on :func:`ordered_causal_attention` instead of
    the dispatcher. The projections are held BY REFERENCE as submodules, so
    a later :func:`use_ordered_kernels` walk still finds and swaps them; the
    residual dropout is not reproduced because this module refuses training
    mode outright, where eval-mode dropout is the identity.

    Every path the scorer does not take is refused rather than approximated:
    a cache to extend, a padding mask, a head mask, attention outputs and
    train mode all raise, because an arm that silently fell back to vendor
    arithmetic on some calls would write records claiming an ownership it
    did not have.
    """

    def __init__(self, original: SdpaAttentionProto) -> None:
        """Take over one attention module, holding its projections."""
        super().__init__()
        self.c_attn = original.c_attn
        self.c_proj = original.c_proj
        self.num_heads = original.num_heads
        self.head_dim = original.head_dim
        self.split_size = original.split_size
        self.embed_dim = original.embed_dim
        # Born in eval: a fresh nn.Module defaults to training mode, and a
        # swap into an already-eval model would otherwise refuse its first
        # forward. A later model.train() still flips this flag and the
        # refusal still fires -- which is the contract.
        self.eval()

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: tuple[torch.Tensor, ...] | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, None]:
        """The causal scoring path, owned; everything else refused.

        Args:
            hidden_states: ``[batch, length, embed]``.
            layer_past: Refused when present.
            attention_mask: Refused when present.
            head_mask: Refused when present.
            encoder_hidden_states: Refused when present.
            encoder_attention_mask: Ignored -- the original only reads it
                under cross-attention, which is refused above it.
            use_cache: When true, the split key and value ride along, as the
                original returns them.
            output_attentions: Refused when true.

        Returns:
            ``(attn_output, present, None)``, shaped as the original's.

        Raises:
            ValueError: For any refused path, or in training mode.
        """
        if self.training:
            raise ValueError("OrderedSdpaAttention is eval-only; dropout would entangle the RNG")
        if layer_past is not None or attention_mask is not None or head_mask is not None:
            raise ValueError(
                "OrderedSdpaAttention owns only the causal scoring path; "
                "caches, padding masks and head masks are refused, not approximated"
            )
        if encoder_hidden_states is not None or bool(output_attentions):
            raise ValueError(
                "OrderedSdpaAttention does not do cross-attention or attention outputs"
            )
        bsz, q_len, _ = hidden_states.size()
        query, key, value = split_three(self.c_attn(hidden_states), self.split_size, 2)
        query = query.view(bsz, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(bsz, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(bsz, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_output = ordered_causal_attention(query, key, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.embed_dim)
        attn_output = self.c_proj(attn_output)
        present = (key, value) if use_cache else None
        return attn_output, present, None


def use_ordered_attention(model: SwapTargetProto) -> int:
    """Replace every SDPA attention module with its ordered version.

    The companion walk to :func:`use_ordered_kernels`, and deliberately a
    SEPARATE one: the projections-only swap is what the seven-GPU record
    corpus pins, and a flag that quietly widened it would change what those
    records mean. A caller wanting the fully-owned model runs this walk
    first, then the projections walk, and checks both counts.

    Args:
        model: The model to rewrite in place.

    Returns:
        How many attention modules were replaced.
    """
    sdpa = sdpa_attention_class()
    graph = [(path, module) for path, module in model.named_modules() if path]
    replaced = 0
    for path, module in graph:
        if isinstance(module, sdpa):
            model.set_submodule(path, OrderedSdpaAttention(module))
            replaced += 1
    return replaced


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
    conv1d = conv1d_class()
    linear = linear_class()
    graph = [(path, module) for path, module in model.named_modules() if path]
    replaced = 0
    for path, module in graph:
        if isinstance(module, conv1d):
            model.set_submodule(path, OrderedConv1D(module))
            replaced += 1
        elif isinstance(module, linear):
            if module.bias is not None:
                raise ValueError(
                    f"{path} is a Linear with a bias; the ordered swap only replaces "
                    "bias-free Linears, and where F.linear adds a bias has not been measured"
                )
            model.set_submodule(path, OrderedLinear(module))
            replaced += 1
    return replaced


__all__ = [
    "OrderedConv1D",
    "OrderedLinear",
    "OrderedSdpaAttention",
    "SdpaAttentionProto",
    "sdpa_attention_class",
    "use_ordered_attention",
    "use_ordered_kernels",
]
