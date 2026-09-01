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

from ordered_kernels.kernels import BLOCK, gemm, rowsum

#: Shapes chosen to exercise every tiling regime: smaller than one tile,
#: exact multiples, ragged in each dimension, and a K straddling a tile
#: boundary by one -- the bounds-guarded tail the -0.0 argument is about.
SHAPES = (
    (4, 7, 5),
    (BLOCK, BLOCK, BLOCK),
    (BLOCK * 4, BLOCK * 8, BLOCK * 3),
    (7, BLOCK + 1, 19),
    (64, 1152, 384),
)


def _operands(n: int, k: int, m: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(n * 1000 + k * 10 + m)
    return (
        torch.randn(m, device="cuda"),
        torch.randn(n, k, device="cuda"),
        torch.randn(k, m, device="cuda"),
    )


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
