"""The three GEMM arms, run for real on the CPU.

WHAT THESE TESTS CANNOT COVER, and it is the whole point of the ``rank1``
arm. Its claim is that two CARDS produce the same bytes, and one CPU cannot
produce two cards. What is checkable here is every property the claim rests
on: that the arm computes the right product at all, that it reproduces
itself, that its order is the one the docstring says it is, and that it is
NOT merely a rename of the vendor call -- an arm that quietly dispatched back
to ``addmm`` would pass a cross-card comparison for the wrong reason and we
would believe we had solved something.
"""

from __future__ import annotations

import pytest
import torch

from model_trainer.core.services.model.deterministic_gemm import (
    CUBLAS_ARM,
    FP64_ARM,
    KERNEL_ARMS,
    RANK1_ARM,
    cublas_addmm,
    fp64_addmm,
    gemm_by_arm,
    rank1_addmm,
    require_kernel_arm,
)

#: Small enough to run many times on a CPU, in the orientation cuBLASLt
#: reports: ``addmm(bias[M], x[N, K], w[K, M])``.
ROWS = 8
INNER = 16
COLS = 4


def _operands() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one seeded set of operands.

    Returns:
        ``(bias[M], x[N, K], w[K, M])``.
    """
    torch.manual_seed(42)
    return (
        torch.randn(ROWS),
        torch.randn(COLS, INNER),
        torch.randn(INNER, ROWS),
    )


class TestTheArmTable:
    def test_it_names_three_arms_in_reading_order(self) -> None:
        assert KERNEL_ARMS == (CUBLAS_ARM, FP64_ARM, RANK1_ARM)

    def test_the_baseline_is_first(self) -> None:
        # A report reads baseline, cheap attempt, proof. The order is part of
        # what the table says.
        assert KERNEL_ARMS[0] == CUBLAS_ARM

    def test_every_arm_is_dispatchable(self) -> None:
        bias, x, w = _operands()
        shapes = [list(gemm_by_arm(arm, bias, x, w).shape) for arm in KERNEL_ARMS]

        assert shapes == [[COLS, ROWS], [COLS, ROWS], [COLS, ROWS]]

    def test_an_unknown_arm_is_refused(self) -> None:
        with pytest.raises(ValueError, match="kernel must be one of"):
            require_kernel_arm("triton")

    def test_the_refusal_names_the_arms_on_offer(self) -> None:
        with pytest.raises(ValueError, match="cublas, fp64, rank1"):
            require_kernel_arm("")

    def test_a_known_arm_comes_back_unchanged(self) -> None:
        assert require_kernel_arm(RANK1_ARM) == RANK1_ARM

    def test_dispatch_refuses_before_computing(self) -> None:
        bias, x, w = _operands()

        with pytest.raises(ValueError, match="kernel must be one of"):
            gemm_by_arm("splitk", bias, x, w)


class TestTheProduct:
    """Every arm has to compute the same matmul, or none of this means anything."""

    def test_cublas_is_addmm(self) -> None:
        bias, x, w = _operands()

        assert torch.equal(cublas_addmm(bias, x, w), torch.addmm(bias, x, w))

    def test_rank1_agrees_with_addmm_to_float32_rounding(self) -> None:
        # NOT bit-identical -- a different order is the point. What must hold
        # is that it is the same product, so the gap is rounding and nothing
        # else. At K=16 in float32 that is a few ulp.
        bias, x, w = _operands()
        exact = torch.addmm(bias.double(), x.double(), w.double())
        gap = (rank1_addmm(bias, x, w).double() - exact).abs().max().item()

        assert gap < 1e-5

    def test_fp64_agrees_with_addmm_to_float32_rounding(self) -> None:
        bias, x, w = _operands()
        exact = torch.addmm(bias.double(), x.double(), w.double())
        gap = (fp64_addmm(bias, x, w).double() - exact).abs().max().item()

        assert gap < 1e-5

    def test_every_arm_returns_float32(self) -> None:
        # fp64 must NARROW. Returning a wide tensor would compare a quantity
        # the forward pass never holds, and would make the arm incomparable
        # with the other two.
        bias, x, w = _operands()
        dtypes = [gemm_by_arm(arm, bias, x, w).dtype for arm in KERNEL_ARMS]

        assert dtypes == [torch.float32, torch.float32, torch.float32]

    def test_the_bias_is_actually_added_by_every_arm(self) -> None:
        bias, x, w = _operands()
        unbiased = torch.zeros(ROWS)
        biased = [gemm_by_arm(arm, bias, x, w) for arm in KERNEL_ARMS]
        plain = [gemm_by_arm(arm, unbiased, x, w) for arm in KERNEL_ARMS]

        assert [torch.equal(b, p) for b, p in zip(biased, plain, strict=True)] == [
            False,
            False,
            False,
        ]


class TestRankOneIsNotTheVendorCallInDisguise:
    """The failure mode that would make a green cross-card result meaningless.

    If ``rank1`` dispatched back to ``addmm``, two cards would agree exactly
    when cuBLAS happened to agree and we would read that as the fixed order
    working. These pin that it really is different arithmetic.
    """

    def test_it_does_not_reproduce_addmm_bit_for_bit(self) -> None:
        bias, x, w = _operands()

        assert not torch.equal(rank1_addmm(bias, x, w), cublas_addmm(bias, x, w))

    def test_it_reproduces_itself_exactly(self) -> None:
        bias, x, w = _operands()

        assert torch.equal(rank1_addmm(bias, x, w), rank1_addmm(bias, x, w))

    def test_it_adds_the_bias_last_not_first(self) -> None:
        # Both are fixed orders and they give different bits: a bias folded
        # in first participates in the rounding of all K subsequent adds.
        # Last matches how addmm is written, so the arms differ in the
        # reduction under study and not also in where the bias went.
        bias, x, w = _operands()
        accumulator = torch.zeros(COLS, ROWS)
        for k in range(INNER):
            accumulator.addr_(x[:, k], w[k, :])

        assert torch.equal(rank1_addmm(bias, x, w), bias + accumulator)

    def test_reversing_the_k_order_changes_the_bits(self) -> None:
        # The load-bearing property. If summation order did NOT change the
        # result, fixing the order would buy nothing and the whole arm would
        # be pointless -- so this failing means float addition has become
        # associative, not that the code is wrong.
        bias, x, w = _operands()
        backwards = torch.zeros(COLS, ROWS)
        for k in reversed(range(INNER)):
            backwards.addr_(x[:, k], w[k, :])

        assert not torch.equal(rank1_addmm(bias, x, w), bias + backwards)

    def test_it_sums_every_k(self) -> None:
        # An off-by-one in the loop bound would drop a term and still look
        # plausible: the result stays close to the product.
        bias, x, w = _operands()
        dropped = w.clone()
        dropped[INNER - 1, :] = 0.0

        assert not torch.equal(rank1_addmm(bias, x, w), rank1_addmm(bias, x, dropped))
