"""Tests for _json_bridge module."""

from __future__ import annotations

from instrument_io._json_bridge import (
    CellValue,
    JSONValue,
    _df_get_cell_str,
    _df_get_headers_from_row,
    _df_get_row_values,
    _df_json_to_row_dicts,
    _df_slice_to_rows,
    _extract_row_from_dict,
    _get_json_opt_str_value,
    _get_json_str_value,
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


# Tests for _df_json_to_row_dicts
def test_df_json_to_row_dicts_valid() -> None:
    json_str = '[{"a": 1, "b": "test"}, {"a": 2, "b": "test2"}]'
    result = _df_json_to_row_dicts(json_str)
    assert len(result) == 2
    assert result[0]["a"] == 1
    assert result[0]["b"] == "test"
    assert result[1]["a"] == 2


def test_df_json_to_row_dicts_empty() -> None:
    json_str = "[]"
    result = _df_json_to_row_dicts(json_str)
    assert result == []


def test_df_json_to_row_dicts_not_array() -> None:
    json_str = '{"a": 1}'
    result = _df_json_to_row_dicts(json_str)
    assert result == []


def test_df_json_to_row_dicts_skips_non_dict() -> None:
    json_str = '[{"a": 1}, "string", {"a": 2}]'
    result = _df_json_to_row_dicts(json_str)
    assert len(result) == 2


# Tests for _get_json_str_value
def test_get_json_str_value_string() -> None:
    row: dict[str, JSONValue] = {"key": "value"}
    result = _get_json_str_value(row, "key")
    assert result == "value"


def test_get_json_str_value_string_with_spaces() -> None:
    row: dict[str, JSONValue] = {"key": "  value  "}
    result = _get_json_str_value(row, "key")
    assert result == "value"


def test_get_json_str_value_empty_string_returns_none() -> None:
    row: dict[str, JSONValue] = {"key": "   "}
    result = _get_json_str_value(row, "key")
    assert result is None


def test_get_json_str_value_null() -> None:
    row: dict[str, JSONValue] = {"key": None}
    result = _get_json_str_value(row, "key")
    assert result is None


def test_get_json_str_value_missing_key() -> None:
    row: dict[str, JSONValue] = {"other": "value"}
    result = _get_json_str_value(row, "key")
    assert result is None


def test_get_json_str_value_number_converts_to_string() -> None:
    row: dict[str, JSONValue] = {"key": 42}
    result = _get_json_str_value(row, "key")
    assert result == "42"


def test_get_json_str_value_float_converts_to_string() -> None:
    row: dict[str, JSONValue] = {"key": 3.14}
    result = _get_json_str_value(row, "key")
    assert result == "3.14"


# Tests for _get_json_opt_str_value
def test_get_json_opt_str_value_string() -> None:
    row: dict[str, JSONValue] = {"key": "value"}
    result = _get_json_opt_str_value(row, "key")
    assert result == "value"


def test_get_json_opt_str_value_preserves_whitespace() -> None:
    row: dict[str, JSONValue] = {"key": "  value  "}
    result = _get_json_opt_str_value(row, "key")
    assert result == "  value  "


def test_get_json_opt_str_value_null() -> None:
    row: dict[str, JSONValue] = {"key": None}
    result = _get_json_opt_str_value(row, "key")
    assert result is None


def test_get_json_opt_str_value_missing_key() -> None:
    row: dict[str, JSONValue] = {"other": "value"}
    result = _get_json_opt_str_value(row, "key")
    assert result is None


def test_get_json_opt_str_value_number_converts() -> None:
    row: dict[str, JSONValue] = {"key": 42}
    result = _get_json_opt_str_value(row, "key")
    assert result == "42"


# Tests for _df_get_row_values
def test_df_get_row_values_valid() -> None:
    json_str = '[{"a": "val1", "b": "val2"}, {"a": "val3", "b": "val4"}]'
    result = _df_get_row_values(json_str, 0)
    assert result == ["val1", "val2"]


def test_df_get_row_values_second_row() -> None:
    json_str = '[{"a": "val1"}, {"a": "val2"}]'
    result = _df_get_row_values(json_str, 1)
    assert result == ["val2"]


def test_df_get_row_values_invalid_index_negative() -> None:
    json_str = '[{"a": "val1"}]'
    result = _df_get_row_values(json_str, -1)
    assert result == []


def test_df_get_row_values_invalid_index_too_large() -> None:
    json_str = '[{"a": "val1"}]'
    result = _df_get_row_values(json_str, 5)
    assert result == []


def test_df_get_row_values_converts_null_to_empty() -> None:
    json_str = '[{"a": null, "b": "val"}]'
    result = _df_get_row_values(json_str, 0)
    assert result == ["", "val"]


def test_df_get_row_values_converts_numbers() -> None:
    json_str = '[{"a": 42, "b": 3.14}]'
    result = _df_get_row_values(json_str, 0)
    assert result == ["42", "3.14"]


# Tests for _df_get_headers_from_row
def test_df_get_headers_from_row_valid() -> None:
    json_str = '[{"col_0": "Name", "col_1": "Value"}, {"col_0": "data", "col_1": "data2"}]'
    columns = ["col_0", "col_1"]
    result = _df_get_headers_from_row(json_str, 0, columns)
    assert result == ["Name", "Value"]


def test_df_get_headers_from_row_with_null_fallback() -> None:
    json_str = '[{"col_0": null, "col_1": "Value"}]'
    columns = ["col_0", "col_1"]
    result = _df_get_headers_from_row(json_str, 0, columns)
    assert result == ["col_0", "Value"]


def test_df_get_headers_from_row_with_empty_fallback() -> None:
    json_str = '[{"col_0": "   ", "col_1": "Value"}]'
    columns = ["col_0", "col_1"]
    result = _df_get_headers_from_row(json_str, 0, columns)
    assert result == ["col_0", "Value"]


def test_df_get_headers_from_row_invalid_index() -> None:
    json_str = '[{"col_0": "Name"}]'
    columns = ["col_0", "col_1"]
    result = _df_get_headers_from_row(json_str, 5, columns)
    assert result == ["col_0", "col_1"]


def test_df_get_headers_from_row_converts_numbers() -> None:
    json_str = '[{"col_0": 123, "col_1": "Value"}]'
    columns = ["col_0", "col_1"]
    result = _df_get_headers_from_row(json_str, 0, columns)
    assert result == ["123", "Value"]


# Tests for _df_get_cell_str
def test_df_get_cell_str_valid() -> None:
    json_str = '[{"a": "val1", "b": "val2"}, {"a": "val3", "b": "val4"}]'
    result = _df_get_cell_str(json_str, 0, "a")
    assert result == "val1"


def test_df_get_cell_str_second_row() -> None:
    json_str = '[{"a": "val1"}, {"a": "val2"}]'
    result = _df_get_cell_str(json_str, 1, "a")
    assert result == "val2"


def test_df_get_cell_str_invalid_index() -> None:
    json_str = '[{"a": "val1"}]'
    result = _df_get_cell_str(json_str, 5, "a")
    assert result is None


def test_df_get_cell_str_missing_column() -> None:
    json_str = '[{"a": "val1"}]'
    result = _df_get_cell_str(json_str, 0, "b")
    assert result is None


def test_df_get_cell_str_null_value() -> None:
    json_str = '[{"a": null}]'
    result = _df_get_cell_str(json_str, 0, "a")
    assert result is None


# Tests for _df_slice_to_rows
def test_df_slice_to_rows_valid() -> None:
    json_str = '[{"a": "h1"}, {"a": "r1"}, {"a": "r2"}]'
    result = _df_slice_to_rows(json_str, 1)
    assert result == [["r1"], ["r2"]]


def test_df_slice_to_rows_from_start() -> None:
    json_str = '[{"a": "r1"}, {"a": "r2"}]'
    result = _df_slice_to_rows(json_str, 0)
    assert result == [["r1"], ["r2"]]


def test_df_slice_to_rows_beyond_length() -> None:
    json_str = '[{"a": "r1"}]'
    result = _df_slice_to_rows(json_str, 5)
    assert result == []


def test_df_slice_to_rows_converts_null() -> None:
    json_str = '[{"a": null, "b": "val"}]'
    result = _df_slice_to_rows(json_str, 0)
    assert result == [["", "val"]]


def test_df_slice_to_rows_converts_numbers() -> None:
    json_str = '[{"a": 42, "b": 3.14}]'
    result = _df_slice_to_rows(json_str, 0)
    assert result == [["42", "3.14"]]
