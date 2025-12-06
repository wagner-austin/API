"""Tests for _json_bridge module."""

from __future__ import annotations

from instrument_io._json_bridge import (
    CellValue,
    JSONValue,
    _extract_row_from_dict,
    _json_col_to_float_list,
    _json_col_to_int_list,
    _json_col_to_opt_float_list,
    _json_col_to_opt_str_list,
    _json_col_to_str_list,
    _json_value_to_cell,
    _load_json_str,
)


def test_load_json_str_valid_object() -> None:
    result: JSONValue = _load_json_str('{"a": 1, "b": "test"}')
    assert type(result) is dict
    assert result["a"] == 1
    assert result["b"] == "test"


def test_load_json_str_valid_array() -> None:
    result: JSONValue = _load_json_str("[1, 2, 3]")
    assert type(result) is list
    assert result == [1, 2, 3]


def test_load_json_str_valid_string() -> None:
    result: JSONValue = _load_json_str('"hello"')
    assert result == "hello"


def test_load_json_str_valid_number() -> None:
    result: JSONValue = _load_json_str("42")
    assert result == 42


def test_load_json_str_valid_float() -> None:
    result: JSONValue = _load_json_str("3.14")
    assert result == 3.14


def test_load_json_str_valid_bool() -> None:
    result: JSONValue = _load_json_str("true")
    assert result is True


def test_load_json_str_valid_null() -> None:
    result: JSONValue = _load_json_str("null")
    assert result is None


def test_json_value_to_cell_string() -> None:
    result: CellValue = _json_value_to_cell("hello")
    assert result == "hello"


def test_json_value_to_cell_int() -> None:
    result: CellValue = _json_value_to_cell(42)
    assert result == 42


def test_json_value_to_cell_float() -> None:
    result: CellValue = _json_value_to_cell(3.14)
    assert result == 3.14


def test_json_value_to_cell_bool() -> None:
    result: CellValue = _json_value_to_cell(True)
    assert result is True


def test_json_value_to_cell_none() -> None:
    result: CellValue = _json_value_to_cell(None)
    assert result is None


def test_json_value_to_cell_list_returns_none() -> None:
    result: CellValue = _json_value_to_cell([1, 2, 3])
    assert result is None


def test_json_value_to_cell_dict_returns_none() -> None:
    result: CellValue = _json_value_to_cell({"a": 1})
    assert result is None


# Tests for _extract_row_from_dict
def test_extract_row_from_dict_basic() -> None:
    row_dict: dict[str, JSONValue] = {"a": 1, "b": "test", "c": 3.14}
    columns = ["a", "b", "c"]
    result = _extract_row_from_dict(row_dict, columns)
    assert result == [1, "test", 3.14]


def test_extract_row_from_dict_missing_column() -> None:
    row_dict: dict[str, JSONValue] = {"a": 1}
    columns = ["a", "b", "c"]
    result = _extract_row_from_dict(row_dict, columns)
    assert result == [1, None, None]


def test_extract_row_from_dict_with_nested_returns_none() -> None:
    row_dict: dict[str, JSONValue] = {"a": [1, 2], "b": {"x": 1}}
    columns = ["a", "b"]
    result = _extract_row_from_dict(row_dict, columns)
    # Nested structures should return None
    assert result == [None, None]


def test_extract_row_from_dict_with_bool() -> None:
    row_dict: dict[str, JSONValue] = {"flag": True, "other": False}
    columns = ["flag", "other"]
    result = _extract_row_from_dict(row_dict, columns)
    assert result == [True, False]


def test_extract_row_from_dict_empty_columns() -> None:
    row_dict: dict[str, JSONValue] = {"a": 1}
    columns: list[str] = []
    result = _extract_row_from_dict(row_dict, columns)
    assert result == []


# Tests for _json_col_to_str_list
def test_json_col_to_str_list_valid() -> None:
    json_str = '[{"col": "a"}, {"col": "b"}, {"col": "c"}]'
    result = _json_col_to_str_list(json_str, "col")
    assert result == ["a", "b", "c"]


def test_json_col_to_str_list_with_null() -> None:
    json_str = '[{"col": "a"}, {"col": null}, {"col": "c"}]'
    result = _json_col_to_str_list(json_str, "col")
    assert result == ["a", "", "c"]


def test_json_col_to_str_list_with_numbers() -> None:
    json_str = '[{"col": 1}, {"col": 2.5}, {"col": "text"}]'
    result = _json_col_to_str_list(json_str, "col")
    assert result == ["1", "2.5", "text"]


def test_json_col_to_str_list_not_array() -> None:
    json_str = '{"col": "value"}'
    result = _json_col_to_str_list(json_str, "col")
    assert result == []


def test_json_col_to_str_list_skips_non_dict() -> None:
    json_str = '[{"col": "a"}, "not a dict", {"col": "b"}]'
    result = _json_col_to_str_list(json_str, "col")
    assert result == ["a", "b"]


def test_json_col_to_str_list_empty_array() -> None:
    json_str = "[]"
    result = _json_col_to_str_list(json_str, "col")
    assert result == []


# Tests for _json_col_to_opt_str_list
def test_json_col_to_opt_str_list_valid() -> None:
    json_str = '[{"col": "a"}, {"col": "b"}]'
    result = _json_col_to_opt_str_list(json_str, "col")
    assert result == ["a", "b"]


def test_json_col_to_opt_str_list_with_null() -> None:
    json_str = '[{"col": "a"}, {"col": null}, {"col": "c"}]'
    result = _json_col_to_opt_str_list(json_str, "col")
    assert result == ["a", None, "c"]


def test_json_col_to_opt_str_list_with_numbers() -> None:
    json_str = '[{"col": 1}, {"col": 2.5}]'
    result = _json_col_to_opt_str_list(json_str, "col")
    assert result == ["1", "2.5"]


def test_json_col_to_opt_str_list_not_array() -> None:
    json_str = '{"col": "value"}'
    result = _json_col_to_opt_str_list(json_str, "col")
    assert result == []


def test_json_col_to_opt_str_list_skips_non_dict() -> None:
    json_str = '[{"col": "a"}, 123, {"col": "b"}]'
    result = _json_col_to_opt_str_list(json_str, "col")
    assert result == ["a", "b"]


# Tests for _json_col_to_int_list
def test_json_col_to_int_list_valid() -> None:
    json_str = '[{"col": 1}, {"col": 2}, {"col": 3}]'
    result = _json_col_to_int_list(json_str, "col")
    assert result == [1, 2, 3]


def test_json_col_to_int_list_from_float() -> None:
    json_str = '[{"col": 1.0}, {"col": 2.5}, {"col": 3.9}]'
    result = _json_col_to_int_list(json_str, "col")
    assert result == [1, 2, 3]


def test_json_col_to_int_list_skips_null() -> None:
    json_str = '[{"col": 1}, {"col": null}, {"col": 3}]'
    result = _json_col_to_int_list(json_str, "col")
    assert result == [1, 3]


def test_json_col_to_int_list_skips_string() -> None:
    json_str = '[{"col": 1}, {"col": "not int"}, {"col": 3}]'
    result = _json_col_to_int_list(json_str, "col")
    assert result == [1, 3]


def test_json_col_to_int_list_skips_bool() -> None:
    json_str = '[{"col": 1}, {"col": true}, {"col": 3}]'
    result = _json_col_to_int_list(json_str, "col")
    # bool should be skipped (not isinstance(item, bool))
    assert result == [1, 3]


def test_json_col_to_int_list_not_array() -> None:
    json_str = '{"col": 1}'
    result = _json_col_to_int_list(json_str, "col")
    assert result == []


def test_json_col_to_int_list_skips_non_dict() -> None:
    json_str = '[{"col": 1}, "string", {"col": 2}]'
    result = _json_col_to_int_list(json_str, "col")
    assert result == [1, 2]


# Tests for _json_col_to_float_list
def test_json_col_to_float_list_valid() -> None:
    json_str = '[{"col": 1.5}, {"col": 2.5}, {"col": 3.5}]'
    result = _json_col_to_float_list(json_str, "col")
    assert result == [1.5, 2.5, 3.5]


def test_json_col_to_float_list_from_int() -> None:
    json_str = '[{"col": 1}, {"col": 2}, {"col": 3}]'
    result = _json_col_to_float_list(json_str, "col")
    assert result == [1.0, 2.0, 3.0]


def test_json_col_to_float_list_skips_null() -> None:
    json_str = '[{"col": 1.5}, {"col": null}, {"col": 3.5}]'
    result = _json_col_to_float_list(json_str, "col")
    assert result == [1.5, 3.5]


def test_json_col_to_float_list_skips_string() -> None:
    json_str = '[{"col": 1.5}, {"col": "not float"}, {"col": 3.5}]'
    result = _json_col_to_float_list(json_str, "col")
    assert result == [1.5, 3.5]


def test_json_col_to_float_list_skips_bool() -> None:
    json_str = '[{"col": 1.5}, {"col": true}, {"col": 3.5}]'
    result = _json_col_to_float_list(json_str, "col")
    # bool should be skipped
    assert result == [1.5, 3.5]


def test_json_col_to_float_list_not_array() -> None:
    json_str = '{"col": 1.5}'
    result = _json_col_to_float_list(json_str, "col")
    assert result == []


def test_json_col_to_float_list_skips_non_dict() -> None:
    json_str = '[{"col": 1.5}, null, {"col": 2.5}]'
    result = _json_col_to_float_list(json_str, "col")
    assert result == [1.5, 2.5]


# Tests for _json_col_to_opt_float_list
def test_json_col_to_opt_float_list_valid() -> None:
    json_str = '[{"col": 1.5}, {"col": 2.5}]'
    result = _json_col_to_opt_float_list(json_str, "col")
    assert result == [1.5, 2.5]


def test_json_col_to_opt_float_list_with_null() -> None:
    json_str = '[{"col": 1.5}, {"col": null}, {"col": 3.5}]'
    result = _json_col_to_opt_float_list(json_str, "col")
    assert result == [1.5, None, 3.5]


def test_json_col_to_opt_float_list_from_int() -> None:
    json_str = '[{"col": 1}, {"col": 2}]'
    result = _json_col_to_opt_float_list(json_str, "col")
    assert result == [1.0, 2.0]


def test_json_col_to_opt_float_list_with_string_returns_none() -> None:
    json_str = '[{"col": 1.5}, {"col": "not float"}, {"col": 3.5}]'
    result = _json_col_to_opt_float_list(json_str, "col")
    assert result == [1.5, None, 3.5]


def test_json_col_to_opt_float_list_with_bool_returns_none() -> None:
    json_str = '[{"col": 1.5}, {"col": true}, {"col": 3.5}]'
    result = _json_col_to_opt_float_list(json_str, "col")
    # bool should be converted to None in else branch
    assert result == [1.5, None, 3.5]


def test_json_col_to_opt_float_list_not_array() -> None:
    json_str = '{"col": 1.5}'
    result = _json_col_to_opt_float_list(json_str, "col")
    assert result == []


def test_json_col_to_opt_float_list_skips_non_dict() -> None:
    json_str = '[{"col": 1.5}, [1, 2], {"col": 2.5}]'
    result = _json_col_to_opt_float_list(json_str, "col")
    assert result == [1.5, 2.5]
