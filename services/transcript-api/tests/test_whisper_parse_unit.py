from __future__ import annotations

import logging

import pytest

import transcript_api.whisper_parse as wmod
from transcript_api.types import JsonValue, VerboseResponseTD


class _Obj1:
    def to_dict(self) -> dict[str, JsonValue]:
        return {"text": "", "segments": []}


class _Obj2:
    def to_dict_recursive(
        self,
    ) -> dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]]:
        return {"text": "all", "segments": [{"text": " a ", "start": "0.0", "end": "1.0"}]}


def test_to_verbose_dict_prefers_methods_and_raises_on_invalid() -> None:
    # Use object with to_dict_recursive
    d = wmod.to_verbose_dict(_Obj2())
    assert isinstance(d, dict) and "segments" in d
    # Accept dict passed directly
    test_dict: dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]] = {
        "text": "x",
        "segments": [],
    }
    d2 = wmod.to_verbose_dict(test_dict)
    assert d2["text"] == "x"


def test_to_verbose_dict_method_raises_and_bubbles() -> None:
    class _Bad:
        def to_dict_recursive(
            self,
        ) -> dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]]:
            raise TypeError("boom")

    import pytest

    with pytest.raises(TypeError):
        _ = wmod.to_verbose_dict(_Bad())


def test_to_verbose_dict_non_dict_then_next_method() -> None:
    class _Obj:
        def to_dict(self) -> JsonValue:
            return [1, 2, 3]  # not a dict -> ignored

        def model_dump(
            self,
        ) -> dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]]:
            return {"text": "t", "segments": []}

    d = wmod.to_verbose_dict(_Obj())
    assert d.get("text") == "t"


def test_convert_verbose_to_segments_strips_and_parses_numbers() -> None:
    payload: VerboseResponseTD = {
        "text": "",
        "segments": [
            {"text": "  hello  ", "start": 1.5, "end": 3.0},
            {"text": "   ", "start": 0, "end": 0},
        ],
    }
    out = wmod.convert_verbose_to_segments(payload)
    assert len(out) == 1 and out[0]["text"] == "hello" and out[0]["duration"] == 1.5


def test_convert_verbose_to_segments_non_list_returns_empty() -> None:
    out = wmod.convert_verbose_to_segments({"text": "", "segments": []})
    assert out == []


def test_to_verbose_dict_missing_fields_raise() -> None:
    # segment missing end field - runtime validation will catch this
    bad4: dict[str, str | bool | int | float | None | list[dict[str, str | int | float]]] = {
        "text": "x",
        "segments": [{"text": "t", "start": 1.0}],
    }
    with pytest.raises(ValueError):
        _ = wmod.to_verbose_dict(bad4)


def test_as_float_edges() -> None:
    f = wmod._as_float
    assert f(5) == 5.0
    assert f("7.25") == 7.25
    assert f("bad") == 0.0
    assert f("+4") == 4.0
    assert f(" ") == 0.0
    assert f(None) == 0.0


def test_to_verbose_dict_strict_invalid_inputs() -> None:
    bad_text: dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]] = {
        "text": 5,  # invalid type
        "segments": [],
    }
    with pytest.raises(ValueError):
        _ = wmod.to_verbose_dict(bad_text)

    missing_segments: dict[
        str, str | int | float | bool | None | list[dict[str, str | int | float]]
    ] = {"text": "x"}
    with pytest.raises(ValueError):
        _ = wmod.to_verbose_dict(missing_segments)

    bad_segment_shape: dict[
        str, str | int | float | bool | None | list[dict[str, str | int | float]]
    ] = {"text": "x", "segments": [{"text": "only"}]}
    with pytest.raises(ValueError):
        _ = wmod.to_verbose_dict(bad_segment_shape)

    seg_missing_numbers: dict[
        str, str | int | float | bool | None | list[dict[str, str | int | float]]
    ] = {
        "text": "x",
        "segments": [{"text": "a"}],
    }
    with pytest.raises(ValueError):
        _ = wmod.to_verbose_dict(seg_missing_numbers)


def test_coerce_verbose_response_rejects_invalid_segments() -> None:
    raw_int_segment: wmod.RawVerboseExtended = {"text": "x", "segments": [123]}
    with pytest.raises(ValueError):
        _ = wmod._coerce_verbose_response(raw_int_segment)

    raw_bad_numeric: wmod.RawVerboseExtended = {
        "text": "x",
        "segments": [{"text": "t", "start": {"k": 1}, "end": 1}],
    }
    with pytest.raises(ValueError):
        _ = wmod._coerce_verbose_response(raw_bad_numeric)

    raw_bad_end: wmod.RawVerboseExtended = {
        "text": "x",
        "segments": [{"text": "t", "start": 1, "end": {"k": 2}}],
    }
    with pytest.raises(ValueError):
        _ = wmod._coerce_verbose_response(raw_bad_end)


def test_to_verbose_dict_invalid_protocol_returns() -> None:
    with pytest.raises(ValueError):
        _ = wmod.to_verbose_dict(["unsupported"])


logger = logging.getLogger(__name__)
