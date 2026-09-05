"""Reading one named field out of a decoded JSON object."""

from __future__ import annotations

import pytest

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    optional_float,
    optional_int,
    optional_str,
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_list,
    require_str,
    require_str_list,
)


class TestRequireStr:
    """Tests for require_str."""

    def test_extracts_str(self) -> None:
        obj: JSONObject = {"name": "Alice"}
        result = require_str(obj, "name")
        assert result == "Alice"

    def test_raises_for_missing(self) -> None:
        obj: JSONObject = {}
        with pytest.raises(JSONTypeError, match="Missing required field 'name'"):
            require_str(obj, "name")

    def test_raises_for_wrong_type(self) -> None:
        obj: JSONObject = {"name": 123}
        with pytest.raises(JSONTypeError, match="Field 'name' must be a string, got int"):
            require_str(obj, "name")


class TestRequireInt:
    """Tests for require_int."""

    def test_extracts_int(self) -> None:
        obj: JSONObject = {"count": 42}
        result = require_int(obj, "count")
        assert result == 42

    def test_raises_for_missing(self) -> None:
        obj: JSONObject = {}
        with pytest.raises(JSONTypeError, match="Missing required field 'count'"):
            require_int(obj, "count")

    def test_raises_for_wrong_type(self) -> None:
        obj: JSONObject = {"count": "42"}
        with pytest.raises(JSONTypeError, match="Field 'count' must be an integer, got str"):
            require_int(obj, "count")

    def test_raises_for_bool(self) -> None:
        obj: JSONObject = {"count": True}
        with pytest.raises(JSONTypeError, match="Field 'count' must be an integer, got bool"):
            require_int(obj, "count")


class TestRequireFloat:
    """Tests for require_float."""

    def test_extracts_float(self) -> None:
        obj: JSONObject = {"rate": 3.14}
        result = require_float(obj, "rate")
        assert result == 3.14

    def test_converts_int_to_float(self) -> None:
        obj: JSONObject = {"rate": 42}
        result = require_float(obj, "rate")
        assert result == 42.0
        assert type(result) is float

    def test_raises_for_missing(self) -> None:
        obj: JSONObject = {}
        with pytest.raises(JSONTypeError, match="Missing required field 'rate'"):
            require_float(obj, "rate")

    def test_raises_for_wrong_type(self) -> None:
        obj: JSONObject = {"rate": "3.14"}
        with pytest.raises(JSONTypeError, match="Field 'rate' must be a number, got str"):
            require_float(obj, "rate")

    def test_raises_for_bool(self) -> None:
        obj: JSONObject = {"rate": True}
        with pytest.raises(JSONTypeError, match="Field 'rate' must be a number, got bool"):
            require_float(obj, "rate")


class TestRequireBool:
    """Tests for require_bool."""

    def test_extracts_true(self) -> None:
        obj: JSONObject = {"enabled": True}
        result = require_bool(obj, "enabled")
        assert result is True

    def test_extracts_false(self) -> None:
        obj: JSONObject = {"enabled": False}
        result = require_bool(obj, "enabled")
        assert result is False

    def test_raises_for_missing(self) -> None:
        obj: JSONObject = {}
        with pytest.raises(JSONTypeError, match="Missing required field 'enabled'"):
            require_bool(obj, "enabled")

    def test_raises_for_wrong_type(self) -> None:
        obj: JSONObject = {"enabled": 1}
        with pytest.raises(JSONTypeError, match="Field 'enabled' must be a boolean, got int"):
            require_bool(obj, "enabled")


class TestRequireList:
    """Tests for require_list."""

    def test_extracts_list(self) -> None:
        obj: JSONObject = {"items": [1, 2, 3]}
        result = require_list(obj, "items")
        assert result == [1, 2, 3]

    def test_raises_for_missing(self) -> None:
        obj: JSONObject = {}
        with pytest.raises(JSONTypeError, match="Missing required field 'items'"):
            require_list(obj, "items")

    def test_raises_for_wrong_type(self) -> None:
        obj: JSONObject = {"items": "not a list"}
        with pytest.raises(JSONTypeError, match="Field 'items' must be an array, got str"):
            require_list(obj, "items")


class TestRequireStrList:
    """Four packages decoded string arrays with their own copy of this loop."""

    def test_extracts_the_strings_in_order(self) -> None:
        obj: JSONObject = {"tags": ["b", "a", "c"]}

        assert require_str_list(obj, "tags") == ["b", "a", "c"]

    def test_an_empty_array_is_accepted(self) -> None:
        """A field that must not be empty is the caller's own rule; three of
        the four decoders that share this permit an empty list."""
        obj: JSONObject = {"tags": []}

        assert require_str_list(obj, "tags") == []

    def test_raises_for_missing(self) -> None:
        obj: JSONObject = {}
        with pytest.raises(JSONTypeError, match="Missing required field 'tags'"):
            require_str_list(obj, "tags")

    def test_raises_for_a_non_array(self) -> None:
        obj: JSONObject = {"tags": "a,b"}
        with pytest.raises(JSONTypeError, match="Field 'tags' must be an array, got str"):
            require_str_list(obj, "tags")

    def test_the_offending_index_is_named(self) -> None:
        """A decoder that reported only the field would leave the caller to
        find which of forty tags was the number."""
        obj: JSONObject = {"tags": ["a", "b", 3]}

        with pytest.raises(JSONTypeError, match=r"Field 'tags\[2\]' must be a string, got int"):
            require_str_list(obj, "tags")

    def test_a_nested_array_is_refused_like_any_other_non_string(self) -> None:
        obj: JSONObject = {"tags": [["a"]]}

        with pytest.raises(JSONTypeError, match=r"Field 'tags\[0\]' must be a string, got list"):
            require_str_list(obj, "tags")

    def test_the_result_is_a_new_list(self) -> None:
        """Callers wrap the result in a tuple, but a decoder that handed back
        the decoded JSON's own list would let a later mutation reach it."""
        items: list[JSONValue] = ["a", "b"]
        obj: JSONObject = {"tags": items}

        assert require_str_list(obj, "tags") is not items


class TestRequireDict:
    """Tests for require_dict."""

    def test_extracts_dict(self) -> None:
        obj: JSONObject = {"config": {"key": "value"}}
        result = require_dict(obj, "config")
        assert result == {"key": "value"}

    def test_raises_for_missing(self) -> None:
        obj: JSONObject = {}
        with pytest.raises(JSONTypeError, match="Missing required field 'config'"):
            require_dict(obj, "config")

    def test_raises_for_wrong_type(self) -> None:
        obj: JSONObject = {"config": [1, 2, 3]}
        with pytest.raises(JSONTypeError, match="Field 'config' must be an object, got list"):
            require_dict(obj, "config")


class TestOptionalStr:
    """Tests for optional_str."""

    def test_returns_none_when_key_missing(self) -> None:
        obj: JSONObject = {}
        result = optional_str(obj, "name")
        assert result is None

    def test_returns_none_when_value_is_none(self) -> None:
        obj: JSONObject = {"name": None}
        result = optional_str(obj, "name")
        assert result is None

    def test_extracts_str(self) -> None:
        obj: JSONObject = {"name": "Alice"}
        result = optional_str(obj, "name")
        assert result == "Alice"
        assert type(result) is str

    def test_raises_for_wrong_type(self) -> None:
        obj: JSONObject = {"name": 123}
        with pytest.raises(JSONTypeError, match="Field 'name' must be a string, got int"):
            optional_str(obj, "name")


class TestOptionalInt:
    """Tests for optional_int."""

    def test_returns_none_when_key_missing(self) -> None:
        obj: JSONObject = {}
        result = optional_int(obj, "count")
        assert result is None

    def test_returns_none_when_value_is_none(self) -> None:
        obj: JSONObject = {"count": None}
        result = optional_int(obj, "count")
        assert result is None

    def test_extracts_int(self) -> None:
        obj: JSONObject = {"count": 42}
        result = optional_int(obj, "count")
        assert result == 42
        assert type(result) is int

    def test_raises_for_bool(self) -> None:
        obj: JSONObject = {"count": True}
        with pytest.raises(JSONTypeError, match="Field 'count' must be an integer, got bool"):
            optional_int(obj, "count")

    def test_raises_for_wrong_type(self) -> None:
        obj: JSONObject = {"count": "42"}
        with pytest.raises(JSONTypeError, match="Field 'count' must be an integer, got str"):
            optional_int(obj, "count")


class TestOptionalFloat:
    """Tests for optional_float."""

    def test_returns_none_when_key_missing(self) -> None:
        obj: JSONObject = {}
        result = optional_float(obj, "rate")
        assert result is None

    def test_returns_none_when_value_is_none(self) -> None:
        obj: JSONObject = {"rate": None}
        result = optional_float(obj, "rate")
        assert result is None

    def test_extracts_float(self) -> None:
        obj: JSONObject = {"rate": 3.14}
        result = optional_float(obj, "rate")
        assert result == 3.14
        assert type(result) is float

    def test_converts_int_to_float(self) -> None:
        obj: JSONObject = {"rate": 42}
        result = optional_float(obj, "rate")
        assert result == 42.0
        assert type(result) is float

    def test_raises_for_bool(self) -> None:
        obj: JSONObject = {"rate": True}
        with pytest.raises(JSONTypeError, match="Field 'rate' must be a number, got bool"):
            optional_float(obj, "rate")

    def test_raises_for_wrong_type(self) -> None:
        obj: JSONObject = {"rate": "3.14"}
        with pytest.raises(JSONTypeError, match="Field 'rate' must be a number, got str"):
            optional_float(obj, "rate")
