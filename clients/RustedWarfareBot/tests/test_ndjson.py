"""The strict NDJSON reader, including every rejection it must make."""

from __future__ import annotations

import pytest

from rw_bot.wire.ndjson import NdjsonError, parse_object


def test_parses_a_real_captured_frame_record() -> None:
    line = '{"kind":"frame","frame":1597,"clock_ms":5388,"owned":3}'
    assert parse_object(line) == {
        "kind": "frame",
        "frame": 1597,
        "clock_ms": 5388,
        "owned": 3,
    }


def test_parses_a_real_captured_entity_record() -> None:
    line = (
        '{"kind":"entity","frame":1,"index":2,'
        '"class":"com.corrodinggames.rts.game.units.h","x":-1000.0,"y":-1000.0}'
    )
    parsed = parse_object(line)
    assert parsed["class"] == "com.corrodinggames.rts.game.units.h"
    assert parsed["x"] == -1000.0
    assert parsed["y"] == -1000.0


def test_parses_an_empty_object() -> None:
    assert parse_object("{}") == {}


def test_tolerates_insignificant_whitespace() -> None:
    assert parse_object('  { "a" : 1 , "b" : 2 }  ') == {"a": 1, "b": 2}


def test_booleans_are_narrowed_not_stringified() -> None:
    parsed = parse_object('{"t":true,"f":false}')
    assert parsed["t"] is True
    assert parsed["f"] is False


def test_an_integer_stays_an_int_and_a_decimal_becomes_a_float() -> None:
    # repr distinguishes 4250 from 4250.0; == treats them as equal, so a
    # decoder that silently widened every number would still pass on ==.
    parsed = parse_object('{"i":4250,"f":4250.5,"e":1e3}')
    assert repr(parsed["i"]) == "4250"
    assert repr(parsed["f"]) == "4250.5"
    assert repr(parsed["e"]) == "1000.0"


def test_numeric_strings_are_not_coerced() -> None:
    assert parse_object('{"x":"4250"}') == {"x": "4250"}


@pytest.mark.parametrize(
    ("escape", "expected"),
    [
        (r"\"", '"'),
        (r"\\", "\\"),
        (r"\/", "/"),
        (r"\b", "\b"),
        (r"\f", "\f"),
        (r"\n", "\n"),
        (r"\r", "\r"),
        (r"\t", "\t"),
        (r"A", "A"),
    ],
)
def test_string_escapes_are_resolved(escape: str, expected: str) -> None:
    assert parse_object('{"k":"' + escape + '"}') == {"k": expected}


def test_rejects_a_line_that_is_not_an_object() -> None:
    with pytest.raises(NdjsonError) as caught:
        parse_object('["a"]')
    assert caught.value.code == "RW-NDJSON-001"


def test_rejects_a_nested_object() -> None:
    """The producer is constrained to flat records; nesting is a contract break."""
    with pytest.raises(NdjsonError) as caught:
        parse_object('{"a":{"b":1}}')
    assert caught.value.code == "RW-NDJSON-002"


def test_rejects_an_array_value() -> None:
    with pytest.raises(NdjsonError) as caught:
        parse_object('{"a":[1]}')
    assert caught.value.code == "RW-NDJSON-002"


def test_rejects_null_because_the_producer_never_emits_it() -> None:
    with pytest.raises(NdjsonError) as caught:
        parse_object('{"a":null}')
    assert caught.value.code == "RW-NDJSON-002"


def test_rejects_a_missing_colon() -> None:
    with pytest.raises(NdjsonError) as caught:
        parse_object('{"a" 1}')
    assert caught.value.code == "RW-NDJSON-002"


def test_rejects_a_missing_separator_between_pairs() -> None:
    with pytest.raises(NdjsonError) as caught:
        parse_object('{"a":1 "b":2}')
    assert caught.value.code == "RW-NDJSON-002"


def test_rejects_a_duplicate_key_rather_than_letting_the_last_win() -> None:
    with pytest.raises(NdjsonError) as caught:
        parse_object('{"a":1,"a":2}')
    assert caught.value.code == "RW-NDJSON-005"


def test_rejects_an_unterminated_string() -> None:
    with pytest.raises(NdjsonError) as caught:
        parse_object('{"a":"open}')
    assert caught.value.code == "RW-NDJSON-003"


def test_rejects_an_unquoted_key() -> None:
    with pytest.raises(NdjsonError) as caught:
        parse_object("{a:1}")
    assert caught.value.code == "RW-NDJSON-003"


def test_rejects_an_unknown_escape() -> None:
    with pytest.raises(NdjsonError) as caught:
        parse_object('{"a":"\\q"}')
    assert caught.value.code == "RW-NDJSON-003"


def test_rejects_an_escape_that_runs_past_the_end() -> None:
    with pytest.raises(NdjsonError) as caught:
        parse_object('{"a":"x\\')
    assert caught.value.code == "RW-NDJSON-003"


@pytest.mark.parametrize("bad", ['{"a":"\\u00"}', '{"a":"\\uZZZZ"}'])
def test_rejects_a_malformed_unicode_escape(bad: str) -> None:
    with pytest.raises(NdjsonError) as caught:
        parse_object(bad)
    assert caught.value.code == "RW-NDJSON-003"


@pytest.mark.parametrize("bad", ['{"a":1.2.3}', '{"a":-}', '{"a":1e}'])
def test_rejects_a_malformed_number(bad: str) -> None:
    with pytest.raises(NdjsonError) as caught:
        parse_object(bad)
    assert caught.value.code == "RW-NDJSON-004"


def test_rejects_content_after_the_object() -> None:
    """Two records on one line is the corruption newline-delimiting prevents."""
    with pytest.raises(NdjsonError) as caught:
        parse_object('{"a":1}{"b":2}')
    assert caught.value.code == "RW-NDJSON-006"


def test_rejects_content_after_an_empty_object() -> None:
    with pytest.raises(NdjsonError) as caught:
        parse_object("{} trailing")
    assert caught.value.code == "RW-NDJSON-006"


def test_rejects_a_truncated_object() -> None:
    with pytest.raises(NdjsonError) as caught:
        parse_object('{"a":1')
    assert caught.value.code == "RW-NDJSON-002"
