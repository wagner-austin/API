"""The tensor fold, exercised on real tensors.

Nothing is faked. Every case here is a tensor built in the test and folded by
the production function, because the whole claim -- "this number changes when
and only when the bytes change" -- is a claim about real bytes.

WHAT THE CANCELLATION TEST IS FOR. The reason this is a digest and not a sum
is that a sum can hide a difference: move ``+d`` from one element to another
and the total is unchanged. That is not a hypothetical failure mode of some
other design, it is the failure mode of the obvious design, so it is measured
here rather than described in a comment.

WHY THE TENSORS ARE BUILT THROUGH HELPERS. ``torch.tensor([1.0, 2.0])`` types
its list literal as ``list[Any]``, which this package forbids. Passing the
literal through a parameter annotated ``list[float]`` gives it a type, and
reads better at the call sites besides.
"""

from __future__ import annotations

import hashlib
import math

import pytest
import torch

from model_trainer.core.services.model.tensor_digest import (
    CHUNK_ELEMENTS,
    DIGEST_BYTES,
    describe_tensor,
    fold_digest,
)


def floats(values: list[float], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build a floating-point tensor from a typed list."""
    return torch.tensor(values, dtype=dtype)


def ints(values: list[int], dtype: torch.dtype = torch.int64) -> torch.Tensor:
    """Build an integral tensor from a typed list."""
    return torch.tensor(values, dtype=dtype)


def bools(values: list[bool]) -> torch.Tensor:
    """Build a boolean tensor from a typed list."""
    return torch.tensor(values, dtype=torch.bool)


class TestFoldingADigest:
    def test_it_takes_the_leading_bytes_big_endian(self) -> None:
        assert fold_digest(bytes([1, 0, 0, 0, 0, 0, 255])) == float(1 << 40)

    def test_it_uses_six_bytes_so_the_float_stays_exact(self) -> None:
        assert DIGEST_BYTES == 6
        assert fold_digest(b"\xff" * 8) == float((1 << 48) - 1)
        assert (1 << 48) - 1 < 2**53

    def test_a_short_digest_is_refused_rather_than_padded(self) -> None:
        with pytest.raises(ValueError, match="at least 6 bytes, got 5"):
            fold_digest(b"\x01\x02\x03\x04\x05")

    def test_it_matches_a_hand_computed_sha256_prefix(self) -> None:
        digest = hashlib.sha256(b"corvis").digest()

        assert fold_digest(digest) == float(int.from_bytes(digest[:6], "big"))


class TestDescribingAFloatTensor:
    def test_the_same_tensor_folds_to_the_same_pair(self) -> None:
        tensor = floats([1.5, -2.25, 3.125])

        assert describe_tensor(tensor) == describe_tensor(tensor.clone())

    def test_one_changed_element_changes_the_digest(self) -> None:
        tensor = floats([1.5, -2.25, 3.125])
        nudged = tensor.clone()
        nudged[1] = torch.nextafter(nudged[1], torch.tensor(0.0))

        assert describe_tensor(tensor)[0] != describe_tensor(nudged)[0]

    def test_a_difference_the_sum_cancels_still_changes_the_digest(self) -> None:
        # The two tensors have the same total to the last bit and are not the
        # same tensor. A sum-only identity would call these equal.
        left = floats([1.0, 3.0])
        right = floats([3.0, 1.0])

        assert describe_tensor(left)[1] == describe_tensor(right)[1]
        assert describe_tensor(left)[0] != describe_tensor(right)[0]

    def test_negative_zero_is_not_the_same_tensor_as_zero(self) -> None:
        assert describe_tensor(floats([0.0]))[0] != describe_tensor(floats([-0.0]))[0]

    def test_the_sum_is_the_exactly_rounded_sum_within_one_chunk(self) -> None:
        tensor = floats([0.1, 0.2, 0.3, -0.6])
        widened: list[float] = tensor.double().tolist()

        assert describe_tensor(tensor)[1] == math.fsum(widened)

    def test_shape_does_not_enter_the_digest_only_the_bytes(self) -> None:
        flat = floats([1.0, 2.0, 3.0, 4.0])

        assert describe_tensor(flat) == describe_tensor(flat.reshape(2, 2))

    def test_a_float64_tensor_folds_by_its_own_eight_byte_rendering(self) -> None:
        as_double = floats([1.5, -2.25], dtype=torch.float64)
        as_single = floats([1.5, -2.25])

        # Same values, different width, therefore different bytes.
        assert describe_tensor(as_double)[1] == describe_tensor(as_single)[1]
        assert describe_tensor(as_double)[0] != describe_tensor(as_single)[0]

    def test_a_non_contiguous_view_folds_as_the_values_it_presents(self) -> None:
        base = floats([1.0, 2.0, 3.0, 4.0]).reshape(2, 2)

        assert describe_tensor(base.t()) == describe_tensor(floats([1.0, 3.0, 2.0, 4.0]))


class TestDescribingATensorLargerThanOneChunk:
    def test_a_change_in_the_second_chunk_changes_the_digest(self) -> None:
        # Two chunks and a remainder, so the loop runs more than once and the
        # last pass is short. A single-chunk test could not tell a working
        # loop from one that only ever folded the first CHUNK_ELEMENTS.
        tensor = torch.zeros(CHUNK_ELEMENTS * 2 + 7, dtype=torch.float32)
        tensor[CHUNK_ELEMENTS + 3] = 1.0
        moved = torch.zeros_like(tensor)
        moved[CHUNK_ELEMENTS * 2 + 3] = 1.0

        assert describe_tensor(tensor)[1] == describe_tensor(moved)[1]
        assert describe_tensor(tensor)[0] != describe_tensor(moved)[0]

    def test_the_chunked_sum_reproduces_itself(self) -> None:
        tensor = torch.full((CHUNK_ELEMENTS + 5,), 0.125, dtype=torch.float32)

        assert describe_tensor(tensor)[1] == describe_tensor(tensor.clone())[1]
        assert describe_tensor(tensor)[1] == 0.125 * (CHUNK_ELEMENTS + 5)


class TestDescribingAnIntegralTensor:
    def test_token_ids_fold_by_their_int64_rendering(self) -> None:
        token_ids = torch.arange(8, dtype=torch.long)

        assert describe_tensor(token_ids)[1] == 28.0
        assert describe_tensor(token_ids) == describe_tensor(token_ids.clone())

    def test_one_changed_id_changes_the_digest(self) -> None:
        token_ids = torch.arange(8, dtype=torch.long)
        changed = token_ids.clone()
        changed[3] = 100

        assert describe_tensor(token_ids)[0] != describe_tensor(changed)[0]

    def test_a_bool_mask_widens_to_the_same_bytes_as_its_ones_and_zeros(self) -> None:
        assert describe_tensor(bools([True, False, True])) == describe_tensor(ints([1, 0, 1]))

    def test_an_int32_tensor_folds_as_the_int64_it_widens_to(self) -> None:
        narrow = ints([7, -7], dtype=torch.int32)
        wide = ints([7, -7])

        assert describe_tensor(narrow) == describe_tensor(wide)


class TestRefusals:
    def test_a_nan_is_refused_because_struct_drops_its_payload(self) -> None:
        with pytest.raises(ValueError, match="refusing to digest a tensor holding NaN"):
            describe_tensor(floats([1.0, float("nan")]))

    def test_an_infinity_is_not_refused_because_it_round_trips_exactly(self) -> None:
        finite = floats([1.0, 2.0])
        infinite = floats([1.0, float("inf")])

        assert describe_tensor(infinite)[0] != describe_tensor(finite)[0]
        assert describe_tensor(infinite)[1] == float("inf")

    def test_a_dtype_with_no_exact_rendering_is_refused_and_names_the_ones_that_have(
        self,
    ) -> None:
        with pytest.raises(ValueError, match=r"cannot render dtype torch\.float16 exactly"):
            describe_tensor(floats([1.0], dtype=torch.float16))

    def test_the_refusal_lists_every_supported_dtype(self) -> None:
        with pytest.raises(ValueError, match=r"torch\.float32, torch\.float64, torch\.bool"):
            describe_tensor(floats([1.0], dtype=torch.bfloat16))
