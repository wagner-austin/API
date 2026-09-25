"""Narrowing words to StrEnum members: found, refused, and read from JSON."""

from __future__ import annotations

from enum import StrEnum

import pytest

from platform_core.json_utils import JSONObject, JSONTypeError
from platform_core.members import as_member, find_member, require_member


class _Shade(StrEnum):
    LIGHT = "light"
    DEEP_BLUE = "deep-blue"


def test_find_member_returns_the_member_carrying_the_word() -> None:
    assert find_member("deep-blue", _Shade) is _Shade.DEEP_BLUE


@pytest.mark.parametrize("word", ["LIGHT", "", "DEEP_BLUE", "deep_blue", "dark"])
def test_find_member_returns_none_for_a_word_no_member_carries(word: str) -> None:
    assert find_member(word, _Shade) is None


def test_as_member_narrows_a_word_in_the_vocabulary() -> None:
    assert as_member("light", "shade", _Shade) is _Shade.LIGHT


def test_as_member_refuses_naming_field_word_and_every_admitted_word() -> None:
    with pytest.raises(JSONTypeError) as excinfo:
        as_member("dark", "shade", _Shade)
    assert str(excinfo.value) == "Invalid shade 'dark': must be one of 'light', 'deep-blue'"


def test_require_member_reads_the_field_then_narrows_it() -> None:
    obj: JSONObject = {"shade": "deep-blue"}
    assert require_member(obj, "shade", _Shade) is _Shade.DEEP_BLUE


def test_require_member_refuses_a_word_outside_the_vocabulary() -> None:
    obj: JSONObject = {"shade": "dark"}
    with pytest.raises(JSONTypeError) as excinfo:
        require_member(obj, "shade", _Shade)
    assert str(excinfo.value) == "Invalid shade 'dark': must be one of 'light', 'deep-blue'"


def test_require_member_refuses_a_missing_field_through_require_str() -> None:
    obj: JSONObject = {}
    with pytest.raises(JSONTypeError) as excinfo:
        require_member(obj, "shade", _Shade)
    assert str(excinfo.value) == "Missing required field 'shade'"


def test_require_member_refuses_a_non_string_field_through_require_str() -> None:
    obj: JSONObject = {"shade": 3}
    with pytest.raises(JSONTypeError) as excinfo:
        require_member(obj, "shade", _Shade)
    assert str(excinfo.value) == "Field 'shade' must be a string, got int"
