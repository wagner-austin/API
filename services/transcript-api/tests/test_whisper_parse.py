from __future__ import annotations

import logging

from transcript_api.types import VerboseResponseTD
from transcript_api.whisper_parse import convert_verbose_to_segments, to_verbose_dict


def test_to_verbose_dict_accepts_dict_and_model_dump() -> None:
    d: dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]] = {
        "text": "hello",
        "segments": [],
    }
    out1 = to_verbose_dict(d)
    assert out1["text"] == "hello" and out1["segments"] == []

    class _Dumpable:
        def __init__(
            self,
            payload: dict[
                str, str | int | float | bool | None | list[dict[str, str | int | float]]
            ],
        ) -> None:
            self._payload = payload

        def model_dump(
            self,
        ) -> dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]]:
            return self._payload

    out2 = to_verbose_dict(_Dumpable(d))
    assert out2["text"] == "hello" and out2["segments"] == []


def test_convert_verbose_to_segments_filters_and_parses() -> None:
    payload: VerboseResponseTD = {
        "text": "",
        "segments": [
            {"text": " one ", "start": 0.0, "end": 1.5},
            {"text": "", "start": 1, "end": 2},  # filtered
            {"text": "two", "start": 2, "end": 4},
        ],
    }
    segs = convert_verbose_to_segments(payload)
    assert len(segs) == 2
    assert segs[0]["text"] == "one" and segs[0]["start"] == 0.0 and segs[0]["duration"] == 1.5
    assert segs[1]["text"] == "two" and segs[1]["duration"] == 2.0


logger = logging.getLogger(__name__)
