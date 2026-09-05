"""Narrowing one JSONValue to a concrete type, or refusing it."""

from __future__ import annotations

import pytest

from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    narrow_json_to_bool,
    narrow_json_to_dict,
    narrow_json_to_float,
    narrow_json_to_int,
    narrow_json_to_list,
    narrow_json_to_str,
)


class TestNarrowJsonToDict:
    """Tests for narrow_json_to_dict."""

    def test_narrows_dict(self) -> None:
        value: JSONValue = {"a": 1, "b": "hello"}
        result = narrow_json_to_dict(value)
        assert result == {"a": 1, "b": "hello"}
        assert type(result) is dict

    def test_raises_for_list(self) -> None:
        value: JSONValue = [1, 2, 3]
        with pytest.raises(JSONTypeError, match="Expected JSON object, got list"):
            narrow_json_to_dict(value)

    def test_raises_for_str(self) -> None:
        value: JSONValue = "hello"
        with pytest.raises(JSONTypeError, match="Expected JSON object, got str"):
            narrow_json_to_dict(value)

    def test_raises_for_int(self) -> None:
        value: JSONValue = 42
        with pytest.raises(JSONTypeError, match="Expected JSON object, got int"):
            narrow_json_to_dict(value)

    def test_raises_for_none(self) -> None:
        value: JSONValue = None
        with pytest.raises(JSONTypeError, match="Expected JSON object, got NoneType"):
            narrow_json_to_dict(value)


class TestNarrowJsonToList:
    """Tests for narrow_json_to_list."""

    def test_narrows_list(self) -> None:
        value: JSONValue = [1, "two", 3.0]
        result = narrow_json_to_list(value)
        assert result == [1, "two", 3.0]
        assert type(result) is list

    def test_raises_for_dict(self) -> None:
        value: JSONValue = {"a": 1}
        with pytest.raises(JSONTypeError, match="Expected JSON array, got dict"):
            narrow_json_to_list(value)

    def test_raises_for_str(self) -> None:
        value: JSONValue = "hello"
        with pytest.raises(JSONTypeError, match="Expected JSON array, got str"):
            narrow_json_to_list(value)

    def test_raises_for_int(self) -> None:
        value: JSONValue = 42
        with pytest.raises(JSONTypeError, match="Expected JSON array, got int"):
            narrow_json_to_list(value)

    def test_raises_for_none(self) -> None:
        value: JSONValue = None
        with pytest.raises(JSONTypeError, match="Expected JSON array, got NoneType"):
            narrow_json_to_list(value)


class TestNarrowJsonToStr:
    """Tests for narrow_json_to_str."""

    def test_narrows_str(self) -> None:
        value: JSONValue = "hello world"
        result = narrow_json_to_str(value)
        assert result == "hello world"
        assert type(result) is str

    def test_raises_for_dict(self) -> None:
        value: JSONValue = {"a": 1}
        with pytest.raises(JSONTypeError, match="Expected JSON string, got dict"):
            narrow_json_to_str(value)

    def test_raises_for_list(self) -> None:
        value: JSONValue = [1, 2]
        with pytest.raises(JSONTypeError, match="Expected JSON string, got list"):
            narrow_json_to_str(value)

    def test_raises_for_int(self) -> None:
        value: JSONValue = 42
        with pytest.raises(JSONTypeError, match="Expected JSON string, got int"):
            narrow_json_to_str(value)

    def test_raises_for_none(self) -> None:
        value: JSONValue = None
        with pytest.raises(JSONTypeError, match="Expected JSON string, got NoneType"):
            narrow_json_to_str(value)


class TestNarrowJsonToInt:
    """Tests for narrow_json_to_int."""

    def test_narrows_int(self) -> None:
        value: JSONValue = 42
        result = narrow_json_to_int(value)
        assert result == 42
        assert type(result) is int

    def test_raises_for_bool(self) -> None:
        value: JSONValue = True
        with pytest.raises(JSONTypeError, match="Expected JSON integer, got bool"):
            narrow_json_to_int(value)

    def test_raises_for_float(self) -> None:
        value: JSONValue = 3.14
        with pytest.raises(JSONTypeError, match="Expected JSON integer, got float"):
            narrow_json_to_int(value)

    def test_raises_for_str(self) -> None:
        value: JSONValue = "42"
        with pytest.raises(JSONTypeError, match="Expected JSON integer, got str"):
            narrow_json_to_int(value)

    def test_raises_for_none(self) -> None:
        value: JSONValue = None
        with pytest.raises(JSONTypeError, match="Expected JSON integer, got NoneType"):
            narrow_json_to_int(value)


class TestNarrowJsonToFloat:
    """Tests for narrow_json_to_float."""

    def test_narrows_float(self) -> None:
        value: JSONValue = 3.14
        result = narrow_json_to_float(value)
        assert result == 3.14
        assert type(result) is float

    def test_converts_int_to_float(self) -> None:
        value: JSONValue = 42
        result = narrow_json_to_float(value)
        assert result == 42.0
        assert type(result) is float

    def test_raises_for_bool(self) -> None:
        value: JSONValue = True
        with pytest.raises(JSONTypeError, match="Expected JSON number, got bool"):
            narrow_json_to_float(value)

    def test_raises_for_str(self) -> None:
        value: JSONValue = "3.14"
        with pytest.raises(JSONTypeError, match="Expected JSON number, got str"):
            narrow_json_to_float(value)

    def test_raises_for_dict(self) -> None:
        value: JSONValue = {"a": 1}
        with pytest.raises(JSONTypeError, match="Expected JSON number, got dict"):
            narrow_json_to_float(value)

    def test_raises_for_none(self) -> None:
        value: JSONValue = None
        with pytest.raises(JSONTypeError, match="Expected JSON number, got NoneType"):
            narrow_json_to_float(value)


class TestNarrowJsonToBool:
    """Tests for narrow_json_to_bool."""

    def test_narrows_true(self) -> None:
        value: JSONValue = True
        result = narrow_json_to_bool(value)
        assert result is True
        assert type(result) is bool

    def test_narrows_false(self) -> None:
        value: JSONValue = False
        result = narrow_json_to_bool(value)
        assert result is False
        assert type(result) is bool

    def test_raises_for_int(self) -> None:
        value: JSONValue = 1
        with pytest.raises(JSONTypeError, match="Expected JSON boolean, got int"):
            narrow_json_to_bool(value)

    def test_raises_for_str(self) -> None:
        value: JSONValue = "true"
        with pytest.raises(JSONTypeError, match="Expected JSON boolean, got str"):
            narrow_json_to_bool(value)

    def test_raises_for_none(self) -> None:
        value: JSONValue = None
        with pytest.raises(JSONTypeError, match="Expected JSON boolean, got NoneType"):
            narrow_json_to_bool(value)
