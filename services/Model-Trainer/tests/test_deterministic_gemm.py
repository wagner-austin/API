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
    BLOCK_ARMS,
    BLOCK_SIZES,
    CUBLAS_ARM,
    FP64_ARM,
    KERNEL_ARMS,
    RANK1_ARM,
    blocked_matmul,
    cublas_addmm,
    fp64_addmm,
    gemm_by_arm,
    rank1_addmm,
    require_block_size,
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
    def test_it_names_every_arm_in_reading_order(self) -> None:
        assert (CUBLAS_ARM, FP64_ARM, RANK1_ARM, *BLOCK_ARMS) == KERNEL_ARMS

    def test_the_block_arms_are_named_for_their_width(self) -> None:
        assert BLOCK_ARMS == ("block256", "block1024", "block1280")
        assert [require_block_size(a) for a in BLOCK_ARMS] == list(BLOCK_SIZES)

    def test_the_block_widths_bracket_the_measured_threshold(self) -> None:
        # The 2026-08-30 bracket put the V100's departure between K=1024
        # (agreed) and K=1152 (differed) at M=3840. The widths are chosen so
        # one sits on each side of that line, with a control far below.
        assert 1024 in BLOCK_SIZES
        assert 1280 in BLOCK_SIZES
        assert min(BLOCK_SIZES) < 1024

    def test_a_width_is_refused_for_an_arm_that_declares_none(self) -> None:
        with pytest.raises(ValueError, match="is not one of block256"):
            require_block_size(RANK1_ARM)

    def test_the_baseline_is_first(self) -> None:
        # A report reads baseline, cheap attempt, proof. The order is part of
        # what the table says.
        assert KERNEL_ARMS[0] == CUBLAS_ARM

    def test_every_arm_is_dispatchable(self) -> None:
        bias, x, w = _operands()
        shapes = [list(gemm_by_arm(arm, bias, x, w).shape) for arm in KERNEL_ARMS]

        assert shapes == [[COLS, ROWS]] * len(KERNEL_ARMS)

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

        assert dtypes == [torch.float32] * len(KERNEL_ARMS)

    def test_the_bias_is_actually_added_by_every_arm(self) -> None:
        bias, x, w = _operands()
        unbiased = torch.zeros(ROWS)
        biased = [gemm_by_arm(arm, bias, x, w) for arm in KERNEL_ARMS]
        plain = [gemm_by_arm(arm, unbiased, x, w) for arm in KERNEL_ARMS]

        assert [torch.equal(b, p) for b, p in zip(biased, plain, strict=True)] == [False] * len(
            KERNEL_ARMS
        )


class TestTheBlockedArms:
    """Chunking is the middle ground: the program cuts, the vendor reduces.

    So unlike ``rank1`` these carry no proof, and the tests say only what is
    true of them -- that they compute the right product, that the cut is where
    the width says it is, and that a K which is not a multiple of the width
    still sums every term exactly once.
    """

    def test_it_agrees_with_addmm_to_float32_rounding(self) -> None:
        bias, x, w = _operands()
        exact = torch.addmm(bias.double(), x.double(), w.double())

        for arm in BLOCK_ARMS:
            gap = (gemm_by_arm(arm, bias, x, w).double() - exact).abs().max().item()

            assert gap < 1e-5

    def test_a_width_at_or_above_k_is_one_chunk_and_adds_nothing(self) -> None:
        # With one chunk the arm IS a single addmm onto a zero accumulator, so
        # it must be bit-identical to that -- the degenerate case proving the
        # chunking machinery adds nothing of its own.
        #
        # Against addmm and NOT against torch.matmul, which is the entry point
        # this arm deliberately does not use: measured 2026-08-30, matmul does
        # not respond to CUBLASLT_WORKSPACE_SIZE at all while addmm does, so a
        # matmul-chunked arm silently ran with the split-K control off.
        _, x, w = _operands()
        zeros = torch.zeros(COLS, ROWS)

        assert torch.equal(blocked_matmul(x, w, INNER), torch.addmm(zeros, x, w))
        assert torch.equal(blocked_matmul(x, w, INNER * 4), torch.addmm(zeros, x, w))

    def test_a_ragged_k_sums_every_term_exactly_once(self) -> None:
        # K=17 against a width of 5 leaves a tail of 2. Padding with zeros
        # would change the number of terms and put a rounding difference into
        # the arm under study, so the tail is added in its own turn.
        torch.manual_seed(11)
        x = torch.randn(3, 17)
        w = torch.randn(17, 6)
        exact = x.double() @ w.double()

        gap = (blocked_matmul(x, w, 5).double() - exact).abs().max().item()

        assert gap < 1e-5

    def test_a_narrower_width_is_a_different_computation(self) -> None:
        # If every width gave identical bits the arms would be measuring
        # nothing. At a K this short they may coincide, so this asserts on a
        # K long enough for the cut to matter.
        torch.manual_seed(3)
        x = torch.randn(8, 4096)
        w = torch.randn(4096, 64)

        assert not torch.equal(blocked_matmul(x, w, 64), blocked_matmul(x, w, 4096))

    def test_it_reproduces_itself_exactly(self) -> None:
        bias, x, w = _operands()

        for arm in BLOCK_ARMS:
            assert torch.equal(gemm_by_arm(arm, bias, x, w), gemm_by_arm(arm, bias, x, w))


class TestRankOneIsNotTheVendorCallInDisguise:
    """The failure mode that would make a green cross-card result meaningless.

    If ``rank1`` dispatched back to ``addmm``, two cards would agree exactly
    when cuBLAS happened to agree and we would read that as the fixed order
    working. These pin that it really computes its own order.

    WHAT IS NOT ASSERTABLE HERE, and the assertion that used to be. This class
    opened with ``assert not torch.equal(rank1_addmm(...), cublas_addmm(...))``
    -- "it must differ from the vendor call". That passed on this laptop and
    FAILED inside the image on 2026-08-30, at M=384 K=128 N=64, because the
    container's BLAS summed that shape in ascending k as well. Two different
    implementations produced identical bits.

    That is the thesis working rather than breaking: a short K has little for
    a library to reorder, and a plain sequential dot product IS this order. So
    "differs from the baseline" was never evidence of different arithmetic --
    it is a machine-dependent coincidence, and asserting it made the suite
    depend on which BLAS ran it.

    What replaces it is positive and order-specifying: the result must equal
    an ascending-k accumulation written out longhand. Aliasing to ``addmm``
    fails that wherever the two orders differ, and where they do not differ
    the aliasing changes no value and costs nothing.
    """

    def test_it_reproduces_itself_exactly(self) -> None:
        bias, x, w = _operands()

        assert torch.equal(rank1_addmm(bias, x, w), rank1_addmm(bias, x, w))

    def test_it_is_an_ascending_k_accumulation_with_the_bias_added_last(self) -> None:
        # The anchor that replaced "must differ from addmm", and the one that
        # actually catches aliasing: an implementation that dispatched to the
        # vendor fails this wherever the vendor's order differs.
        #
        # Bias last, not first: both are fixed orders and they give different
        # bits, since a bias folded in first participates in the rounding of
        # all K subsequent adds. Last matches how addmm is written, so the
        # arms differ in the reduction under study and not also in where the
        # bias went.
        bias, x, w = _operands()
        accumulator = torch.zeros(COLS, ROWS)
        for k in range(INNER):
            accumulator.addr_(x[:, k], w[k, :])

        assert torch.equal(rank1_addmm(bias, x, w), bias + accumulator)

    def test_folding_the_bias_in_first_would_give_different_bits(self) -> None:
        # Why the previous test names an order rather than just a value. If
        # bias placement did not matter, "bias added last" would be an empty
        # claim and the anchor would be weaker than it looks.
        bias, x, w = _operands()
        first = bias.expand(COLS, ROWS).clone()
        for k in range(INNER):
            first.addr_(x[:, k], w[k, :])

        assert not torch.equal(rank1_addmm(bias, x, w), first)

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
