"""Tests for the codec primitives.

These are the pieces every record codec is built from, so they are tested here
once rather than re-tested through each record type. The per-record modules in
``tests/codecs`` assert that a record reaches these helpers with the right field
names, not that the helpers themselves work.
"""

from __future__ import annotations

import pytest

from navprobe.wireformat import (
    FALSE_TOKEN,
    NONE_TOKEN,
    SEPARATOR,
    TRUE_TOKEN,
    WireFormatError,
    encode_bool,
    encode_float_field,
    encode_optional_int,
    header_line,
    join_document,
    require_bool_field,
    require_float_field,
    require_hexadecimal_float,
    require_int_field,
    require_no_body,
    require_non_negative_field,
    require_non_negative_float_field,
    require_optional_non_negative_field,
    require_positive_field,
    require_positive_float_field,
    require_text_field,
    split_document,
    split_header_line,
)

#: A banner used only to exercise the shared document splitter.
_BANNER = "navprobe-test/1"


class TestEncodeFloatField:
    """Tests for :func:`encode_float_field`."""

    def test_round_trips_a_value_with_no_exact_decimal_form(self) -> None:
        """The hexadecimal form recovers the original bits.

        0.055 is not exactly representable in binary, so this fails for any
        codec that goes through a rounded decimal string.
        """
        assert require_float_field(encode_float_field(0.055), "f") == 0.055

    def test_round_trips_a_very_small_value(self) -> None:
        """Subnormal-adjacent values survive, which decimal formatting need not."""
        assert require_float_field(encode_float_field(1e-300), "f") == 1e-300

    def test_round_trips_a_negative_value(self) -> None:
        """Sign survives the hexadecimal form."""
        assert require_float_field(encode_float_field(-1.5), "f") == -1.5


class TestRequireHexadecimalFloat:
    """Tests for :func:`require_hexadecimal_float`, the shared prefix check."""

    def test_accepts_zero(self) -> None:
        """Zero is a real reading, not a missing one."""
        assert require_hexadecimal_float(encode_float_field(0.0), "f") == 0.0

    def test_rejects_a_decimal_token(self) -> None:
        """Decimal text is refused rather than silently parsed.

        ``float()`` would accept it, which is exactly why the token is matched
        rather than converted: the format accepts less than the parser does.
        """
        with pytest.raises(WireFormatError) as caught:
            require_hexadecimal_float("0.055", "f")
        assert caught.value.code == "NP-WIRE-014"

    def test_accepts_not_a_number(self) -> None:
        """NaN passes the prefix check, so range checks must catch it.

        Pinned because the positive and non-negative variants rely on it: if
        this rejected NaN, their range checks would be unreachable.
        """
        parsed = require_hexadecimal_float("nan", "f")
        assert parsed != parsed


class TestRequireFloatField:
    """Tests for :func:`require_float_field`, the unconstrained variant."""

    def test_accepts_a_negative_value(self) -> None:
        """An observed value has no range; a position may be negative."""
        assert require_float_field(encode_float_field(-1.5), "f") == -1.5

    def test_accepts_zero(self) -> None:
        """Zero depth is a real reading, not a missing one."""
        assert require_float_field(encode_float_field(0.0), "f") == 0.0

    def test_rejects_a_decimal_token(self) -> None:
        """Decimal text is refused rather than silently parsed."""
        with pytest.raises(WireFormatError) as caught:
            require_float_field("0.055", "f")
        assert caught.value.code == "NP-WIRE-014"


class TestRequirePositiveFloatField:
    """Tests for :func:`require_positive_float_field`."""

    def test_accepts_a_positive_value(self) -> None:
        """A length round-trips through the exact encoding."""
        assert require_positive_float_field(encode_float_field(0.055), "f") == 0.055

    def test_rejects_a_zero_value(self) -> None:
        """A length or duration of zero describes no scene."""
        with pytest.raises(WireFormatError) as caught:
            require_positive_float_field(encode_float_field(0.0), "f")
        assert caught.value.code == "NP-WIRE-015"

    def test_rejects_a_negative_value(self) -> None:
        """Negative lengths are refused by the same check."""
        with pytest.raises(WireFormatError) as caught:
            require_positive_float_field(encode_float_field(-0.5), "f")
        assert caught.value.code == "NP-WIRE-015"

    def test_rejects_not_a_number(self) -> None:
        """NaN passes the prefix check and must be caught by the range check."""
        with pytest.raises(WireFormatError) as caught:
            require_positive_float_field("nan", "f")
        assert caught.value.code == "NP-WIRE-015"

    def test_rejects_a_decimal_token(self) -> None:
        """The prefix check fires before the range check."""
        with pytest.raises(WireFormatError) as caught:
            require_positive_float_field("0.055", "f")
        assert caught.value.code == "NP-WIRE-014"


class TestRequireNonNegativeFloatField:
    """Tests for :func:`require_non_negative_float_field`."""

    def test_accepts_zero(self) -> None:
        """A deterministic configuration has a spread of exactly zero.

        This is the most important value such a record carries, so it must
        decode rather than being rejected as out of range.
        """
        assert require_non_negative_float_field(encode_float_field(0.0), "f") == 0.0

    def test_round_trips_a_tiny_spread(self) -> None:
        """A last-bit spread survives, which is the value that matters."""
        assert require_non_negative_float_field(encode_float_field(4.47e-08), "f") == 4.47e-08

    def test_rejects_a_negative_spread(self) -> None:
        """A range cannot be below zero."""
        with pytest.raises(WireFormatError) as caught:
            require_non_negative_float_field(encode_float_field(-1.0), "f")
        assert caught.value.code == "NP-WIRE-016"

    def test_rejects_a_decimal_token(self) -> None:
        """Decimal text is refused rather than silently parsed."""
        with pytest.raises(WireFormatError) as caught:
            require_non_negative_float_field("0.5", "f")
        assert caught.value.code == "NP-WIRE-014"


class TestRequireIntField:
    """Tests for :func:`require_int_field`."""

    def test_accepts_a_positive_integer(self) -> None:
        """A digit string converts."""
        assert require_int_field("42", "f") == 42

    def test_accepts_a_negative_integer(self) -> None:
        """A leading minus is permitted at this level."""
        assert require_int_field("-3", "f") == -3

    def test_rejects_a_non_numeric_token(self) -> None:
        """Text is not an integer."""
        with pytest.raises(WireFormatError) as caught:
            require_int_field("abc", "f")
        assert caught.value.code == "NP-WIRE-001"

    def test_rejects_an_empty_token(self) -> None:
        """An empty field is not zero."""
        with pytest.raises(WireFormatError) as caught:
            require_int_field("", "f")
        assert caught.value.code == "NP-WIRE-001"

    def test_rejects_a_bare_minus(self) -> None:
        """A lone sign has no digits to convert."""
        with pytest.raises(WireFormatError) as caught:
            require_int_field("-", "f")
        assert caught.value.code == "NP-WIRE-001"

    def test_rejects_a_float_token(self) -> None:
        """A decimal point is not an integer field."""
        with pytest.raises(WireFormatError) as caught:
            require_int_field("1.5", "f")
        assert caught.value.code == "NP-WIRE-001"

    def test_names_the_field_it_failed_on(self) -> None:
        """The message identifies which field was bad, not just that one was."""
        with pytest.raises(WireFormatError) as caught:
            require_int_field("abc", "world_count")
        assert "'world_count'" in caught.value.message


class TestRequireNonNegativeField:
    """Tests for :func:`require_non_negative_field`."""

    def test_accepts_zero(self) -> None:
        """Zero is in range."""
        assert require_non_negative_field("0", "f") == 0

    def test_rejects_a_negative_value(self) -> None:
        """A negative count or index is refused."""
        with pytest.raises(WireFormatError) as caught:
            require_non_negative_field("-1", "f")
        assert caught.value.code == "NP-WIRE-002"


class TestRequirePositiveField:
    """Tests for :func:`require_positive_field`."""

    def test_accepts_one(self) -> None:
        """One is the minimum valid value."""
        assert require_positive_field("1", "f") == 1

    def test_rejects_zero(self) -> None:
        """Zero is below the bound."""
        with pytest.raises(WireFormatError) as caught:
            require_positive_field("0", "f")
        assert caught.value.code == "NP-WIRE-003"

    def test_rejects_a_negative_value(self) -> None:
        """Negative values fail the non-negative check first."""
        with pytest.raises(WireFormatError) as caught:
            require_positive_field("-1", "f")
        assert caught.value.code == "NP-WIRE-003"


class TestRequireTextField:
    """Tests for :func:`require_text_field`."""

    def test_accepts_non_empty_text(self) -> None:
        """Text passes through unchanged."""
        assert require_text_field("x", "f") == "x"

    def test_rejects_empty_text(self) -> None:
        """An empty label or digest is always a construction bug."""
        with pytest.raises(WireFormatError) as caught:
            require_text_field("", "f")
        assert caught.value.code == "NP-WIRE-004"


class TestRequireBoolField:
    """Tests for :func:`require_bool_field`."""

    def test_accepts_the_true_token(self) -> None:
        """The spelled true token decodes to ``True``."""
        assert require_bool_field(TRUE_TOKEN, "f") is True

    def test_accepts_the_false_token(self) -> None:
        """The spelled false token decodes to ``False``."""
        assert require_bool_field(FALSE_TOKEN, "f") is False

    def test_rejects_any_other_token(self) -> None:
        """An unrecognised token is refused rather than read as false.

        Reading it as false would turn a typo into a negative determinism
        verdict, which is the one result this instrument must never invent.
        """
        with pytest.raises(WireFormatError) as caught:
            require_bool_field("yes", "f")
        assert caught.value.code == "NP-WIRE-012"

    def test_rejects_a_capitalised_token(self) -> None:
        """Token matching is exact, so the format has one spelling."""
        with pytest.raises(WireFormatError) as caught:
            require_bool_field("True", "f")
        assert caught.value.code == "NP-WIRE-012"


class TestRequireOptionalNonNegativeField:
    """Tests for :func:`require_optional_non_negative_field`."""

    def test_none_token_decodes_to_absence(self) -> None:
        """The spelled none token means absent, not zero."""
        assert require_optional_non_negative_field(NONE_TOKEN, "f") is None

    def test_zero_is_a_value_not_an_absence(self) -> None:
        """Step zero is a real divergence point and must survive decoding."""
        assert require_optional_non_negative_field("0", "f") == 0

    def test_rejects_a_negative_value(self) -> None:
        """The bound still applies to the present case."""
        with pytest.raises(WireFormatError) as caught:
            require_optional_non_negative_field("-1", "f")
        assert caught.value.code == "NP-WIRE-002"

    def test_rejects_an_empty_token(self) -> None:
        """Absence is spelled, so an empty field is malformed."""
        with pytest.raises(WireFormatError) as caught:
            require_optional_non_negative_field("", "f")
        assert caught.value.code == "NP-WIRE-001"


class TestScalarEncoders:
    """Tests for the token encoders."""

    def test_encodes_true(self) -> None:
        """``True`` becomes the true token."""
        assert encode_bool(True) == TRUE_TOKEN

    def test_encodes_false(self) -> None:
        """``False`` becomes the false token."""
        assert encode_bool(False) == FALSE_TOKEN

    def test_encodes_an_absent_optional(self) -> None:
        """``None`` becomes the none token."""
        assert encode_optional_int(None) == NONE_TOKEN

    def test_encodes_a_present_optional(self) -> None:
        """A present value becomes its decimal form."""
        assert encode_optional_int(4) == "4"

    def test_encodes_zero_distinctly_from_absence(self) -> None:
        """Step zero must not encode as the none token."""
        assert encode_optional_int(0) == "0"

    def test_bool_tokens_round_trip(self) -> None:
        """Encoding then decoding a boolean is the identity."""
        assert [require_bool_field(encode_bool(value), "f") for value in (True, False)] == [
            True,
            False,
        ]

    def test_optional_tokens_round_trip(self) -> None:
        """Encoding then decoding an optional is the identity."""
        values: tuple[int | None, ...] = (None, 0, 7)
        decoded = [require_optional_non_negative_field(encode_optional_int(v), "f") for v in values]
        assert decoded == [None, 0, 7]


class TestHeaderLines:
    """Tests for :func:`header_line` and :func:`split_header_line`."""

    def test_joins_a_key_and_value(self) -> None:
        """A header line is the key, a tab, and the value."""
        assert header_line("seed", "7") == f"seed{SEPARATOR}7"

    def test_round_trips_a_value(self) -> None:
        """Splitting a joined line returns the value unchanged."""
        assert split_header_line(header_line("label", "a b"), "label") == "a b"

    def test_rejects_a_line_with_one_token(self) -> None:
        """A header line without a value is malformed."""
        with pytest.raises(WireFormatError) as caught:
            split_header_line("seed", "seed")
        assert caught.value.code == "NP-WIRE-005"

    def test_rejects_a_line_with_three_tokens(self) -> None:
        """A second separator means the value was not what it claimed."""
        with pytest.raises(WireFormatError) as caught:
            split_header_line(f"seed{SEPARATOR}7{SEPARATOR}8", "seed")
        assert caught.value.code == "NP-WIRE-005"

    def test_rejects_an_unexpected_key(self) -> None:
        """Field order is pinned by the key each position must carry."""
        with pytest.raises(WireFormatError) as caught:
            split_header_line(header_line("seed", "7"), "step_count")
        assert caught.value.code == "NP-WIRE-006"


class TestSplitDocument:
    """Tests for :func:`split_document`."""

    def test_returns_exactly_the_declared_header_count(self) -> None:
        """The header slice is the guarantee later indexing relies on."""
        text = join_document([_BANNER, "a\t1", "b\t2", "row"])
        header, _ = split_document(text, _BANNER, 2)
        assert header == ("a\t1", "b\t2")

    def test_returns_everything_after_the_header_as_body(self) -> None:
        """Body lines are whatever follows the declared header."""
        text = join_document([_BANNER, "a\t1", "row-one", "row-two"])
        _, body = split_document(text, _BANNER, 1)
        assert body == ("row-one", "row-two")

    def test_body_is_empty_when_no_rows_follow(self) -> None:
        """A record with exactly its header has an empty body, not a missing one."""
        text = join_document([_BANNER, "a\t1"])
        _, body = split_document(text, _BANNER, 1)
        assert body == ()

    def test_tolerates_a_missing_trailing_newline(self) -> None:
        """A file edited by hand may lose its final newline and still decode."""
        header, _ = split_document(f"{_BANNER}\na\t1", _BANNER, 1)
        assert header == ("a\t1",)

    def test_rejects_a_missing_banner(self) -> None:
        """Text without the banner is not a record of this type."""
        with pytest.raises(WireFormatError) as caught:
            split_document("something else\n", _BANNER, 1)
        assert caught.value.code == "NP-WIRE-009"

    def test_rejects_a_header_shorter_than_declared(self) -> None:
        """A truncated header cannot be indexed, so it is refused first."""
        with pytest.raises(WireFormatError) as caught:
            split_document(join_document([_BANNER, "a\t1"]), _BANNER, 3)
        assert caught.value.code == "NP-WIRE-010"


class TestRequireNoBody:
    """Tests for :func:`require_no_body`."""

    def test_accepts_the_empty_body_split_document_produces(self) -> None:
        """A record whose text stops at its header passes the check.

        The empty tuple is taken from :func:`split_document` rather than
        written literally, so this asserts the two agree about what "no body"
        is rather than assuming it.
        """
        _, body = split_document(join_document([_BANNER, "a\t1"]), _BANNER, 1)
        require_no_body(body, _BANNER)
        assert body == ()

    def test_rejects_trailing_lines(self) -> None:
        """Ignoring trailing content would let two documents decode alike."""
        with pytest.raises(WireFormatError) as caught:
            require_no_body(("extra",), _BANNER)
        assert caught.value.code == "NP-WIRE-013"


class TestJoinDocument:
    """Tests for :func:`join_document`."""

    def test_joins_lines_with_newlines(self) -> None:
        """Lines are joined in the order given."""
        assert join_document(["a", "b"]) == "a\nb\n"

    def test_terminates_with_a_newline(self) -> None:
        """The output is a well-formed text file."""
        assert join_document(["a"]).endswith("\n")
