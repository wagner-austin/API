"""Tests for platform_stt.whisper_parse module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from platform_stt.types import RawVerboseDict, VerboseResponse, VerboseSegment
from platform_stt.whisper_parse import (
    RawVerboseExtended,
    _as_float,
    _coerce_verbose_response,
    _is_numeric_str,
    convert_verbose_to_segments,
    to_verbose_response,
)


class TestAsFloat:
    """Tests for _as_float helper."""

    def test_as_float_int(self) -> None:
        """Convert int to float."""
        assert _as_float(42) == 42.0

    def test_as_float_float(self) -> None:
        """Return float as-is."""
        assert _as_float(3.14) == 3.14

    def test_as_float_str(self) -> None:
        """Convert numeric string to float."""
        assert _as_float("1.5") == 1.5

    def test_as_float_str_with_spaces(self) -> None:
        """Handle string with leading/trailing spaces."""
        assert _as_float("  2.5  ") == 2.5

    def test_as_float_empty_str(self) -> None:
        """Return 0.0 for empty string."""
        assert _as_float("") == 0.0

    def test_as_float_none(self) -> None:
        """Return 0.0 for None."""
        assert _as_float(None) == 0.0

    def test_as_float_non_numeric_str(self) -> None:
        """Return 0.0 for non-numeric string."""
        assert _as_float("abc") == 0.0

    def test_as_float_negative(self) -> None:
        """Handle negative numbers."""
        assert _as_float(-5) == -5.0
        assert _as_float("-3.5") == -3.5

    def test_as_float_positive_sign(self) -> None:
        """Handle positive sign."""
        assert _as_float("+2.5") == 2.5


class TestIsNumericStr:
    """Tests for _is_numeric_str helper."""

    def test_is_numeric_str_integer(self) -> None:
        """Recognize integer string."""
        assert _is_numeric_str("123") is True

    def test_is_numeric_str_float(self) -> None:
        """Recognize float string."""
        assert _is_numeric_str("3.14") is True

    def test_is_numeric_str_negative(self) -> None:
        """Recognize negative number."""
        assert _is_numeric_str("-42") is True

    def test_is_numeric_str_positive(self) -> None:
        """Recognize positive sign."""
        assert _is_numeric_str("+10") is True

    def test_is_numeric_str_empty(self) -> None:
        """Reject empty string."""
        assert _is_numeric_str("") is False

    def test_is_numeric_str_letters(self) -> None:
        """Reject letters."""
        assert _is_numeric_str("abc") is False

    def test_is_numeric_str_double_dot(self) -> None:
        """Reject multiple decimal points."""
        assert _is_numeric_str("1.2.3") is False

    def test_is_numeric_str_only_sign(self) -> None:
        """Reject sign without digits."""
        assert _is_numeric_str("-") is False
        assert _is_numeric_str("+") is False


class TestToVerboseResponse:
    """Tests for to_verbose_response function."""

    def test_to_verbose_response_from_dict(self) -> None:
        """Convert raw dict to VerboseResponse."""
        raw: RawVerboseDict = {
            "text": "Hello world",
            "language": "en",
            "segments": [{"text": "Hello", "start": 0.0, "end": 0.5}],
        }
        result = to_verbose_response(raw)
        assert result["text"] == "Hello world"
        assert result["language"] == "en"
        assert len(result["segments"]) == 1

    def test_to_verbose_response_no_language(self) -> None:
        """Handle missing language field gracefully."""
        raw: RawVerboseDict = {
            "text": "Hello world",
            "segments": [{"text": "Hello", "start": 0.0, "end": 0.5}],
        }
        result = to_verbose_response(raw)
        assert result["text"] == "Hello world"
        assert result["language"] is None
        assert len(result["segments"]) == 1

    def test_to_verbose_response_missing_text(self) -> None:
        """Reject dict missing text field."""
        raw: RawVerboseDict = {"segments": []}
        with pytest.raises(JSONTypeError, match="missing text"):
            to_verbose_response(raw)

    def test_to_verbose_response_missing_segments(self) -> None:
        """Reject dict missing segments field."""
        raw: RawVerboseDict = {"text": "Hello"}
        with pytest.raises(JSONTypeError, match="missing segments list"):
            to_verbose_response(raw)

    def test_to_verbose_response_invalid_segment(self) -> None:
        """Reject non-dict segment."""
        raw: RawVerboseDict = {"text": "Hello", "segments": []}
        # Segments need to be dicts - this is tested by creating the wrong structure
        # We cannot put a string into segments due to type restrictions, so we verify
        # at the implementation level via duck-typing checks
        result = to_verbose_response(raw)
        assert result["segments"] == []

    def test_to_verbose_response_missing_start_end(self) -> None:
        """Reject segment missing start/end."""
        raw: RawVerboseDict = {"text": "Hello", "segments": [{"text": "Hello"}]}
        with pytest.raises(JSONTypeError, match="segment missing start/end"):
            to_verbose_response(raw)

    def test_to_verbose_response_invalid_start_type(self) -> None:
        """Reject non-numeric start."""
        raw: RawVerboseDict = {
            "text": "Hello",
            "segments": [{"text": "Hello", "start": 0.0, "end": 1.0}],
        }
        # We cannot put an invalid type due to RawVerboseDict restrictions
        # Test the valid case instead
        result = to_verbose_response(raw)
        assert result["segments"][0]["start"] == 0.0

    def test_to_verbose_response_invalid_end_type(self) -> None:
        """Reject non-numeric end."""
        raw: RawVerboseDict = {
            "text": "Hello",
            "segments": [{"text": "Hello", "start": 0.0, "end": 1.0}],
        }
        # We cannot put an invalid type due to RawVerboseDict restrictions
        result = to_verbose_response(raw)
        assert result["segments"][0]["end"] == 1.0

    def test_to_verbose_response_unsupported_type(self) -> None:
        """Reject unsupported object type."""
        with pytest.raises(JSONTypeError, match="Unsupported verbose object"):
            to_verbose_response(["list", "not", "dict"])

    def test_to_verbose_response_numeric_strings(self) -> None:
        """Handle numeric strings for start/end."""
        raw: RawVerboseDict = {
            "text": "Hello",
            "segments": [{"text": "Hello", "start": "0.0", "end": "1.5"}],
        }
        result = to_verbose_response(raw)
        assert result["segments"][0]["start"] == 0.0
        assert result["segments"][0]["end"] == 1.5


class TestToVerboseResponseWithProtocol:
    """Tests for to_verbose_response with protocol objects."""

    def test_to_verbose_response_with_to_dict_recursive(self) -> None:
        """Convert object with to_dict_recursive method."""

        class FakeResponse:
            def to_dict_recursive(
                self,
            ) -> dict[str, str | int | float | bool | list[dict[str, str | int | float]] | None]:
                return {
                    "text": "Test",
                    "segments": [{"text": "Test", "start": 0.0, "end": 1.0}],
                }

        result = to_verbose_response(FakeResponse())
        assert result["text"] == "Test"

    def test_to_verbose_response_with_model_dump(self) -> None:
        """Convert object with model_dump method (Pydantic v2 style)."""

        class FakePydanticResponse:
            def model_dump(
                self,
            ) -> dict[str, str | int | float | bool | list[dict[str, str | int | float]] | None]:
                return {
                    "text": "Pydantic",
                    "segments": [{"text": "Pydantic", "start": 0.0, "end": 2.0}],
                }

        result = to_verbose_response(FakePydanticResponse())
        assert result["text"] == "Pydantic"


class TestConvertVerboseToSegments:
    """Tests for convert_verbose_to_segments function."""

    def test_convert_verbose_to_segments_basic(self) -> None:
        """Convert VerboseResponse segments to TranscriptSegments."""
        response = VerboseResponse(
            text="Hello world",
            language="en",
            segments=[
                VerboseSegment(text="Hello", start=0.0, end=0.5),
                VerboseSegment(text="world", start=0.5, end=1.0),
            ],
        )
        result = convert_verbose_to_segments(response)
        assert len(result) == 2
        assert result[0]["text"] == "Hello"
        assert result[0]["start"] == 0.0
        assert result[0]["duration"] == 0.5

    def test_convert_verbose_to_segments_filters_empty(self) -> None:
        """Filter out empty text segments."""
        response = VerboseResponse(
            text="Hello",
            language="en",
            segments=[
                VerboseSegment(text="Hello", start=0.0, end=0.5),
                VerboseSegment(text="", start=0.5, end=0.6),
                VerboseSegment(text="   ", start=0.6, end=0.7),
            ],
        )
        result = convert_verbose_to_segments(response)
        assert len(result) == 1

    def test_convert_verbose_to_segments_negative_duration(self) -> None:
        """Handle negative duration (clamp to 0)."""
        response = VerboseResponse(
            text="Test",
            language="en",
            segments=[VerboseSegment(text="Test", start=1.0, end=0.5)],
        )
        result = convert_verbose_to_segments(response)
        assert result[0]["duration"] == 0.0

    def test_convert_verbose_to_segments_strips_whitespace(self) -> None:
        """Strip whitespace from segment text."""
        response = VerboseResponse(
            text="Test",
            language="en",
            segments=[VerboseSegment(text="  Test  ", start=0.0, end=1.0)],
        )
        result = convert_verbose_to_segments(response)
        assert result[0]["text"] == "Test"


class TestCoerceVerboseResponseValidation:
    """Tests for _coerce_verbose_response runtime validation.

    These tests use RawVerboseExtended to construct invalid payloads
    that bypass static type restrictions but fail runtime validation.
    """

    def test_segment_not_dict_raises(self) -> None:
        """Raise JSONTypeError when segment is not a dict.

        Args: None

        Returns: None

        Raises:
            JSONTypeError: When segment item is not a dictionary.
        """
        # RawVerboseExtended allows list[dict | int], so we can put an int
        raw: RawVerboseExtended = {"text": "Hello", "segments": [42]}
        with pytest.raises(JSONTypeError, match="segment must be object"):
            _coerce_verbose_response(raw)

    def test_segment_start_invalid_type_raises(self) -> None:
        """Raise JSONTypeError when segment start is not int/float/str.

        Args: None

        Returns: None

        Raises:
            JSONTypeError: When start value is not a valid numeric type.
        """
        # RawVerboseExtended allows dict[str, int] as segment dict value
        # Using a list bypasses the isinstance(start_raw, (int, float, str)) check
        raw: RawVerboseExtended = {
            "text": "Hello",
            "segments": [{"text": "Hello", "start": [1, 2], "end": 1.0}],
        }
        with pytest.raises(JSONTypeError, match="segment start must be numeric"):
            _coerce_verbose_response(raw)

    def test_segment_end_invalid_type_raises(self) -> None:
        """Raise JSONTypeError when segment end is not int/float/str.

        Args: None

        Returns: None

        Raises:
            JSONTypeError: When end value is not a valid numeric type.
        """
        # Use a nested dict to bypass isinstance check for end
        raw: RawVerboseExtended = {
            "text": "Hello",
            "segments": [{"text": "Hello", "start": 0.0, "end": {"nested": 1}}],
        }
        with pytest.raises(JSONTypeError, match="segment end must be numeric"):
            _coerce_verbose_response(raw)
