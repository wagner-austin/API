from __future__ import annotations

import logging

from transcript_api.whisper_parse import _is_numeric_str, to_verbose_dict

logger = logging.getLogger(__name__)


def test_to_verbose_dict_strict_errors() -> None:
    # Test that valid Protocol implementations work correctly
    class _Obj1:
        def to_dict_recursive(
            self,
        ) -> dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]]:
            return {"text": "", "segments": []}

    result = to_verbose_dict(_Obj1())
    assert result["text"] == "" and result["segments"] == []

    class _Obj2:
        def model_dump(
            self,
        ) -> dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]]:
            return {"text": "test", "segments": []}

    result2 = to_verbose_dict(_Obj2())
    assert result2["text"] == "test"


def test_is_numeric_str_edge_blanks_and_sign_only() -> None:
    assert _is_numeric_str("") is False
    assert _is_numeric_str("+") is False
