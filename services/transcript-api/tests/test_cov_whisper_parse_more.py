from __future__ import annotations

import logging

from transcript_api.types import VerboseResponseTD
from transcript_api.whisper_parse import (
    _as_float,
    _is_numeric_str,
    convert_verbose_to_segments,
    to_verbose_dict,
)

logger = logging.getLogger(__name__)


def test_to_verbose_dict_branches() -> None:
    class _Obj1:
        def to_dict_recursive(
            self,
        ) -> dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]]:
            return {"text": "", "segments": []}

    d1 = to_verbose_dict(_Obj1())
    assert type(d1) is dict

    class _Obj2:
        def model_dump(
            self,
        ) -> dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]]:
            return {"text": "", "segments": []}

    d2 = to_verbose_dict(_Obj2())
    assert type(d2) is dict


def test_convert_verbose_to_segments_and_numeric_utils() -> None:
    data: VerboseResponseTD = {
        "text": "",
        "segments": [
            {"text": " ", "start": 0, "end": 1},
            {"text": "hello", "start": 1.5, "end": 3.0},
        ],
    }
    out = convert_verbose_to_segments(data)
    assert len(out) == 1 and abs(out[0]["duration"] - 1.5) < 1e-6
    assert _is_numeric_str("+1.23") is True
    assert _is_numeric_str("1.2.3") is False
    assert _as_float(" 2.5 ") == 2.5 and _as_float("abc") == 0.0
