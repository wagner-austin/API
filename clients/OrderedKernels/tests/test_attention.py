"""Owned attention against its longhand oracle, and the module swap.

The oracle recomputes every owned reduction by a route that shares no kernel
with the implementation: rank-one matmuls per head slice, and the softmax
denominator as an ascending column-by-column accumulation of elementwise
adds -- the same order, different code. ``torch.equal`` throughout, because
bit-for-bit is the product; the dispatcher's own math kernel is held to
``allclose`` only, since different arithmetic for the same function is the
entire point.
"""

from __future__ import annotations

import math

import pytest
import torch
from model_trainer.core.services.model.deterministic_gemm import rank1_matmul
from model_trainer.core.services.model.kernel_arm_modules import SwapTargetProto
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import require_probe_shape

from ordered_kernels.attention import (
    CAUSAL_FILL,
    causal_bias,
    ordered_causal_attention,
    ordered_softmax,
)
from ordered_kernels.modules import (
    OrderedSdpaAttention,
    SdpaAttentionProto,
    sdpa_attention_class,
    use_ordered_attention,
    use_ordered_kernels,
)
from ordered_kernels.torch_surface import head_slice, split_three

#: The sm_75 residual's own lengths first, then a ragged and a probed one.
LENGTHS = (15, 16, 7, 64)


def _first_attention(model: SwapTargetProto) -> SdpaAttentionProto:
    """The first SDPA attention module in the graph, found rather than pathed."""
    sdpa = sdpa_attention_class()
    for _, module in model.named_modules():
        if isinstance(module, sdpa):
            return module
    raise AssertionError("the probe model carries no SDPA attention module")


def _qkv(length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(31000 + length)
    shape = (1, 12, length, 64)
    return (
        torch.randn(*shape, device="cuda"),
        torch.randn(*shape, device="cuda"),
        torch.randn(*shape, device="cuda"),
    )


def _longhand_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The oracle: same order, no shared kernel."""
    batch, heads, length, dim = (int(s) for s in q.shape)
    bias = causal_bias(length, q.device)
    outs: list[torch.Tensor] = []
    for b in range(batch):
        for h in range(heads):
            q_head = head_slice(q, b, h)
            k_head = head_slice(k, b, h)
            v_head = head_slice(v, b, h)
            scores = rank1_matmul(q_head, k_head.t()) * (1.0 / math.sqrt(float(dim))) + bias
            row_max = scores.amax(dim=-1, keepdim=True)
            exps = torch.exp(scores - row_max)
            denom = torch.zeros(length, device=q.device)
            for c in range(length):
                denom = denom + exps[:, c]
            probs = exps / denom.unsqueeze(-1)
            outs.append(rank1_matmul(probs, v_head))
    return torch.stack(outs).view(batch, heads, length, dim)


class TestTheAttentionOracle:
    def test_it_is_the_longhand_fixed_order_bit_for_bit_on_every_length(self) -> None:
        for length in LENGTHS:
            q, k, v = _qkv(length)

            assert torch.equal(ordered_causal_attention(q, k, v), _longhand_attention(q, k, v)), (
                length
            )

    def test_strided_views_compute_the_same_bits_as_their_copies(self) -> None:
        # The real path hands in view-and-permute operands.
        torch.manual_seed(31999)
        packed = torch.randn(1, 16, 3 * 768, device="cuda")
        query, key, value = split_three(packed, 768, 2)
        q = query.view(1, 16, 12, 64).permute(0, 2, 1, 3)
        k = key.view(1, 16, 12, 64).permute(0, 2, 1, 3)
        v = value.view(1, 16, 12, 64).permute(0, 2, 1, 3)

        strided = ordered_causal_attention(q, k, v)
        dense = ordered_causal_attention(q.contiguous(), k.contiguous(), v.contiguous())

        assert torch.equal(strided, dense)

    def test_it_computes_the_same_function_as_the_dispatcher(self) -> None:
        # Different arithmetic, same mathematics: allclose, never equal.
        q, k, v = _qkv(16)

        ours = ordered_causal_attention(q, k, v)
        theirs = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        assert torch.allclose(ours, theirs, atol=1e-5)
        assert not torch.equal(ours, theirs)

    def test_mismatched_shapes_are_refused(self) -> None:
        q, k, _ = _qkv(7)

        with pytest.raises(ValueError, match="share one shape"):
            ordered_causal_attention(q, k, torch.randn(1, 12, 8, 64, device="cuda"))

    def test_a_3d_operand_is_refused(self) -> None:
        t = torch.randn(12, 7, 64, device="cuda")

        with pytest.raises(ValueError, match="batch, heads, length, dim"):
            ordered_causal_attention(t, t, t)


class TestTheTorchSurface:
    def test_a_split_that_is_not_three_slices_is_refused(self) -> None:
        with pytest.raises(ValueError, match="expected three slices"):
            split_three(torch.randn(1, 2, 10, device="cuda"), 3, 2)


class TestTheCausalBias:
    def test_the_future_is_minus_infinity_and_the_past_is_zero(self) -> None:
        bias = causal_bias(4, torch.device("cuda"))

        assert bias[0, 0] == 0.0
        assert bias[3, 0] == 0.0
        assert bias[0, 1] == CAUSAL_FILL
        assert bias[2, 3] == CAUSAL_FILL


class TestTheOrderedSoftmax:
    def test_rows_sum_to_one_and_masked_columns_to_zero(self) -> None:
        torch.manual_seed(41)
        scores = torch.randn(6, 6, device="cuda") + causal_bias(6, torch.device("cuda"))

        probs = ordered_softmax(scores)

        assert torch.equal(probs[0, 1:], torch.zeros(5, device="cuda"))
        assert torch.allclose(probs.sum(dim=-1), torch.ones(6, device="cuda"))


class TestTheModuleSwap:
    def test_it_replaces_every_attention_and_the_model_still_computes(self) -> None:
        model, ids = probe_model_and_input("cuda", require_probe_shape("tiny"))
        untreated = float(model.forward(input_ids=ids, labels=ids).loss.item())

        replaced = use_ordered_attention(model)
        projections = use_ordered_kernels(model)
        treated = float(model.forward(input_ids=ids, labels=ids).loss.item())

        assert replaced == 2  # the tiny rung is a two-block model
        assert projections == 9  # 4 Conv1Ds per block plus lm_head
        assert abs(treated - untreated) < 1e-4
        leftovers = [
            path
            for path, module in model.named_modules()
            if isinstance(module, sdpa_attention_class())
        ]
        assert leftovers == []

    def test_the_swapped_forward_reproduces_itself_exactly(self) -> None:
        model, ids = probe_model_and_input("cuda", require_probe_shape("tiny"))
        use_ordered_attention(model)
        use_ordered_kernels(model)

        first = model.forward(input_ids=ids, labels=ids)
        second = model.forward(input_ids=ids, labels=ids)

        assert torch.equal(first.loss, second.loss)
        assert float(first.loss.item()) == float(second.loss.item())

    def test_the_cache_rides_along_when_asked(self) -> None:
        model, _ = probe_model_and_input("cuda", require_probe_shape("tiny"))
        wrapper = OrderedSdpaAttention(_first_attention(model))
        wrapper.eval()
        hidden = torch.randn(1, 5, wrapper.embed_dim, device="cuda")

        out, present, attentions = wrapper.forward(hidden, use_cache=True)

        # The value oracle: the same path recomputed through the wrapper's
        # own projections, bit for bit.
        query, key, value = split_three(wrapper.c_attn(hidden), wrapper.split_size, 2)
        q = query.view(1, 5, wrapper.num_heads, wrapper.head_dim).permute(0, 2, 1, 3)
        k = key.view(1, 5, wrapper.num_heads, wrapper.head_dim).permute(0, 2, 1, 3)
        v = value.view(1, 5, wrapper.num_heads, wrapper.head_dim).permute(0, 2, 1, 3)
        core = ordered_causal_attention(q, k, v)
        merged = core.transpose(1, 2).contiguous().view(1, 5, wrapper.embed_dim)

        assert torch.equal(out, wrapper.c_proj(merged))
        if present is None:
            raise AssertionError("use_cache=True must return the split key and value")
        assert torch.equal(present[0], k)
        assert torch.equal(present[1], v)
        assert attentions is None

    def test_every_unowned_path_is_refused(self) -> None:
        model, _ = probe_model_and_input("cuda", require_probe_shape("tiny"))
        wrapper = OrderedSdpaAttention(_first_attention(model))
        wrapper.eval()
        hidden = torch.randn(1, 5, wrapper.embed_dim, device="cuda")
        past = (torch.zeros(1, device="cuda"),)

        with pytest.raises(ValueError, match="refused, not approximated"):
            wrapper.forward(hidden, layer_past=past)
        with pytest.raises(ValueError, match="refused, not approximated"):
            wrapper.forward(hidden, attention_mask=torch.zeros(1, 5, device="cuda"))
        with pytest.raises(ValueError, match="refused, not approximated"):
            wrapper.forward(hidden, head_mask=torch.ones(1, device="cuda"))
        with pytest.raises(ValueError, match="cross-attention"):
            wrapper.forward(hidden, encoder_hidden_states=hidden)
        with pytest.raises(ValueError, match="cross-attention"):
            wrapper.forward(hidden, output_attentions=True)
        # The flag, not .train(): nothing here trains -- the refusal keys on
        # self.training, and setting it is the narrowest way to drive it.
        wrapper.training = True
        with pytest.raises(ValueError, match="eval-only"):
            wrapper.forward(hidden)


class TestTheGradientsFlowAtAnyLength:
    """The training claim: autograd crosses the owned attention, every length.

    The forward's bits are pinned by the longhand tests above -- these hold
    the BACKWARD to the same standard: reproducible bit for bit, the same
    derivative as the vendor's numerically, and never the vendor's bits,
    because agreeing bitwise would mean passing through rather than
    computing.
    """

    def _upstream(self, length: int) -> torch.Tensor:
        # A fresh CONTIGUOUS tensor, never ``randn_like`` of either arm's
        # output: the dispatcher returns a transposed layout, and
        # ``randn_like`` preserves it, so one RNG stream would land in
        # different logical positions per arm and the comparison would hold
        # two different derivatives against each other. Measured before it
        # was a comment: gaps of ~4.0 absolute, from the layout alone.
        torch.manual_seed(41000 + length)
        return torch.randn(1, 12, length, 64, device="cuda")

    def _grads(self, length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = _qkv(length)
        q, k, v = q.requires_grad_(), k.requires_grad_(), v.requires_grad_()
        out = ordered_causal_attention(q, k, v)
        grad_q, grad_k, grad_v = torch.autograd.grad(out, (q, k, v), self._upstream(length))
        return grad_q, grad_k, grad_v

    def test_the_backward_reproduces_itself_bit_for_bit_on_every_length(self) -> None:
        for length in LENGTHS:
            first = self._grads(length)
            second = self._grads(length)
            for a, b in zip(first, second, strict=True):
                assert torch.equal(a, b), f"backward did not reproduce at L={length}"

    def test_the_gradients_agree_with_the_dispatcher_numerically(self) -> None:
        # A different ORDER of the same sums, not a different derivative --
        # and not the same bits, or the arm would be passing through.
        for length in LENGTHS:
            owned = self._grads(length)

            q, k, v = _qkv(length)
            q, k, v = q.requires_grad_(), k.requires_grad_(), v.requires_grad_()
            theirs = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            vendor = torch.autograd.grad(theirs, (q, k, v), self._upstream(length))

            for ours, refs in zip(owned, vendor, strict=True):
                assert torch.allclose(ours, refs, atol=1e-4)
            assert any(
                not torch.equal(ours, refs) for ours, refs in zip(owned, vendor, strict=True)
            )
