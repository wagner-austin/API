"""Tests for canonical byte encoding."""

from __future__ import annotations

import math
import struct

import pytest

from navprobe.canonical import (
    MAX_ROW_LENGTH,
    CanonicalEncodingError,
    encode_float,
    encode_row,
    encode_text,
)


class TestEncodeFloat:
    """Tests for :func:`encode_float`."""

    def test_encodes_to_eight_little_endian_bytes(self) -> None:
        """One float becomes exactly the little-endian binary64 pattern."""
        assert encode_float(1.0) == struct.pack("<d", 1.0)

    def test_byte_order_is_explicit_not_native(self) -> None:
        """The high byte of 1.0 lands last, which native order would not fix."""
        assert encode_float(1.0) == b"\x00\x00\x00\x00\x00\x00\xf0\x3f"

    def test_distinguishes_positive_and_negative_zero(self) -> None:
        """Zeros that compare equal still encode differently.

        They are distinguishable in IEEE-754 and a digest that collapsed them
        would hide a real change of sign.
        """
        assert encode_float(0.0) != encode_float(-0.0)

    def test_smallest_representable_difference_changes_the_bytes(self) -> None:
        """One ulp apart encodes differently, which is the whole premise."""
        assert encode_float(1.0) != encode_float(math.nextafter(1.0, 2.0))

    def test_accepts_positive_infinity(self) -> None:
        """Infinity is ordered and equals itself, so it is encodable."""
        assert encode_float(math.inf) == struct.pack("<d", math.inf)

    def test_accepts_negative_infinity(self) -> None:
        """Negative infinity is encodable for the same reason."""
        assert encode_float(-math.inf) == struct.pack("<d", -math.inf)

    def test_rejects_nan_with_its_own_code(self) -> None:
        """NaN is refused, because it compares unequal to itself."""
        with pytest.raises(CanonicalEncodingError) as caught:
            encode_float(math.nan)
        assert caught.value.code == "NP-CANON-001"


class TestEncodeRow:
    """Tests for :func:`encode_row`."""

    def test_prefixes_the_element_count(self) -> None:
        """A two-element row starts with a little-endian count of two."""
        assert encode_row([1.0, 2.0])[:4] == struct.pack("<I", 2)

    def test_body_follows_the_prefix_in_order(self) -> None:
        """Elements appear after the prefix, in the order given."""
        assert encode_row([1.0, 2.0])[4:] == encode_float(1.0) + encode_float(2.0)

    def test_empty_row_is_just_the_prefix(self) -> None:
        """A row with no elements encodes to a zero count and nothing else."""
        assert encode_row([]) == struct.pack("<I", 0)

    def test_order_is_significant(self) -> None:
        """Reordering elements changes the encoding."""
        assert encode_row([1.0, 2.0]) != encode_row([2.0, 1.0])

    def test_length_prefix_separates_differently_grouped_rows(self) -> None:
        """Concatenated encodings of different shapes do not collide.

        Without the count prefix, one row of two elements and two rows of one
        would flatten to identical bytes.
        """
        one_row = encode_row([1.0, 2.0])
        two_rows = encode_row([1.0]) + encode_row([2.0])
        assert one_row != two_rows

    def test_propagates_nan_rejection_from_an_element(self) -> None:
        """A NaN anywhere in the row fails the whole row."""
        with pytest.raises(CanonicalEncodingError) as caught:
            encode_row([1.0, math.nan])
        assert caught.value.code == "NP-CANON-001"

    def test_rejects_a_row_longer_than_the_prefix_can_describe(self) -> None:
        """A row too long for the 32-bit count is refused by its own code.

        A :class:`range` is a real sequence that reports its length without
        materialising its elements, so the branch is reachable without
        allocating four billion floats and without a stand-in for a sequence.

        This also covers the bound for :func:`encode_text`, which shares the
        same check: a string of that length cannot be constructed to test
        directly, and duplicating the assertion would not exercise a second
        code path because there is only one.
        """
        with pytest.raises(CanonicalEncodingError) as caught:
            encode_row(range(MAX_ROW_LENGTH + 1))
        assert caught.value.code == "NP-CANON-002"


class TestEncodeText:
    """Tests for :func:`encode_text`."""

    def test_prefixes_the_byte_count(self) -> None:
        """Encoded text starts with a little-endian count of its bytes."""
        assert encode_text("abc")[:4] == struct.pack("<I", 3)

    def test_payload_follows_the_prefix(self) -> None:
        """The UTF-8 bytes follow the prefix unchanged."""
        assert encode_text("abc")[4:] == b"abc"

    def test_empty_text_is_just_the_prefix(self) -> None:
        """Empty text encodes to a zero count and nothing else."""
        assert encode_text("") == struct.pack("<I", 0)

    def test_counts_bytes_not_characters(self) -> None:
        """A multi-byte character contributes its byte length, not one.

        A character count would make the prefix disagree with the payload it
        describes, which is the same defect as having no prefix at all.
        """
        assert encode_text("é")[:4] == struct.pack("<I", 2)

    def test_a_boundary_shift_changes_the_concatenation(self) -> None:
        """Encoding is injective over a sequence, which is the whole purpose.

        ``"aab" + "b"`` and ``"aa" + "bb"`` are the same bytes concatenated
        raw. Prefixed, they are not.
        """
        shifted = encode_text("aab") + encode_text("b")
        even = encode_text("aa") + encode_text("bb")
        assert shifted != even
