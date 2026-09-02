"""The kernels against their oracle: rank1's bits, at tile speed.

Every equality here is ``torch.equal`` -- bit-for-bit -- because bit-for-bit
is the product. The oracle is Model-Trainer's rank-one arithmetic, whose
records span seven GPUs; a kernel that matches it on this card inherits that
corpus as evidence.
"""

from __future__ import annotations

import pytest
import torch
from model_trainer.core.services.model.deterministic_gemm import (
    accumulate_rows,
    rank1_addmm,
    rank1_matmul,
)

from ordered_kernels.kernels import CUDA_SOURCE, K_SLICE, MICRO, THREADS, TILE, gemm, rowsum

#: Shapes chosen to exercise every tiling regime: smaller than one tile,
#: exact tile and K-slice multiples, ragged in each dimension, a K
#: straddling a slice boundary by one (the bounds-guarded tail the -0.0
#: argument is about), and rows/cols straddling the 64-wide output tile so
#: the store masks and padded stage rows are all driven.
SHAPES = (
    (4, 7, 5),
    (K_SLICE, K_SLICE, K_SLICE),
    (TILE, K_SLICE * 8, TILE),
    (7, K_SLICE + 1, 19),
    (TILE + 3, K_SLICE * 5 + 1, TILE + 17),
    (64, 1152, 384),
)


def _operands(n: int, k: int, m: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(n * 1000 + k * 10 + m)
    return (
        torch.randn(m, device="cuda"),
        torch.randn(n, k, device="cuda"),
        torch.randn(k, m, device="cuda"),
    )


class TestTheLaunchContract:
    """The Python constants and the CUDA source describe ONE geometry.

    :func:`gemm` computes its grid from ``TILE`` and its block from
    ``THREADS`` while the source hardcodes its tile literals, so a source
    edit that skips the constants would mislaunch without any oracle test
    noticing the cause. Pinning the literals to the constants makes every
    constant load-bearing and turns that drift into a named failure.
    ``ROWSUM_BLOCK`` needs no pin: the row-sum kernel reads ``blockDim.x``
    and hardcodes nothing.
    """

    def test_the_tile_is_the_thread_grid_times_the_micro_patch(self) -> None:
        assert TILE == THREADS * MICRO

    def test_the_source_hardcodes_exactly_the_constants_geometry(self) -> None:
        fragments = (
            f"__shared__ float xs[{TILE}][{K_SLICE}];",
            f"__shared__ float ws[{K_SLICE}][{TILE}];",
            f"const int row0 = blockIdx.y * {TILE};",
            f"const int col0 = blockIdx.x * {TILE};",
            f"const int tid = threadIdx.y * {THREADS} + threadIdx.x;",
            f"float acc[{MICRO}][{MICRO}];",
            f"k0 += {K_SLICE}) {{",
            f"s < {TILE} * {K_SLICE}; s += {THREADS * THREADS})",
            f"(k_dim - k0 < {K_SLICE}) ? (k_dim - k0) : {K_SLICE};",
            f"xs[threadIdx.y * {MICRO} + i][kk]",
            f"ws[kk][threadIdx.x * {MICRO} + j]",
        )
        for fragment in fragments:
            assert fragment in CUDA_SOURCE, fragment


class TestTheOracle:
    def test_the_product_is_rank1_bit_for_bit_on_every_regime(self) -> None:
        for n, k, m in SHAPES:
            _, x, w = _operands(n, k, m)

            assert torch.equal(gemm(x, w, None), rank1_matmul(x, w)), (n, k, m)

    def test_the_biased_product_is_rank1_addmm_bit_for_bit(self) -> None:
        for n, k, m in SHAPES:
            bias, x, w = _operands(n, k, m)

            assert torch.equal(gemm(x, w, bias), rank1_addmm(bias, x, w)), (n, k, m)

    def test_a_transposed_view_is_the_same_bits_after_its_copy(self) -> None:
        # The backward pass hands in .t() views; the contiguous copy must
        # not change what is computed.
        _, x, w = _operands(24, 33, 17)
        wt = w.t().contiguous().t()

        assert torch.equal(gemm(x, wt, None), rank1_matmul(x, w))

    def test_the_row_sum_is_accumulate_rows_bit_for_bit(self) -> None:
        torch.manual_seed(9)
        grad = torch.randn(37, 129, device="cuda")

        assert torch.equal(rowsum(grad), accumulate_rows(grad))

    def test_it_reproduces_itself_exactly(self) -> None:
        _, x, w = _operands(64, 1152, 384)

        assert torch.equal(gemm(x, w, None), gemm(x, w, None))


class TestTheRefusals:
    def test_a_cpu_operand_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no CPU path"):
            gemm(torch.randn(4, 4), torch.randn(4, 4, device="cuda"), None)

    def test_a_wrong_dtype_is_refused(self) -> None:
        with pytest.raises(ValueError, match="float32"):
            gemm(
                torch.randn(4, 4, device="cuda", dtype=torch.float64),
                torch.randn(4, 4, device="cuda"),
                None,
            )

    def test_a_wrong_rank_is_refused(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            gemm(torch.randn(4, device="cuda"), torch.randn(4, 4, device="cuda"), None)

    def test_mismatched_inner_dimensions_are_refused(self) -> None:
        with pytest.raises(ValueError, match="inner dimensions differ"):
            gemm(torch.randn(4, 5, device="cuda"), torch.randn(6, 4, device="cuda"), None)

    def test_a_malformed_bias_is_refused(self) -> None:
        x = torch.randn(4, 5, device="cuda")
        w = torch.randn(5, 6, device="cuda")

        with pytest.raises(ValueError, match="1-D float32 CUDA"):
            gemm(x, w, torch.randn(6))

    def test_a_wrong_length_bias_is_refused(self) -> None:
        x = torch.randn(4, 5, device="cuda")
        w = torch.randn(5, 6, device="cuda")

        with pytest.raises(ValueError, match="elements for 6 columns"):
            gemm(x, w, torch.randn(7, device="cuda"))
