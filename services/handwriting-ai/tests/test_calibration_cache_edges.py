from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONTypeError

from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import LoggerInstanceProtocol
from handwriting_ai.training.calibration.cache import _decode_float, _decode_int, _read_cache

LogArg = float | int | str | Path | BaseException


ExtraDict = dict[str, str | int | float | bool | None]


class _LogStub:
    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    def info(self, msg: str, *args: LogArg, extra: ExtraDict | None = None) -> None:
        pass

    def warning(self, msg: str, *args: LogArg, extra: ExtraDict | None = None) -> None:
        pass

    def error(self, msg: str, *args: LogArg, extra: ExtraDict | None = None) -> None:
        formatted = msg % args if args else msg
        self._sink.append(formatted)

    def debug(self, msg: str, *args: LogArg, extra: ExtraDict | None = None) -> None:
        pass


def test_decode_int_logs_and_raises_on_invalid() -> None:
    called: list[str] = []
    stub = _LogStub(called)

    def _fake_get_logger(name: str) -> LoggerInstanceProtocol:
        _ = name
        return stub

    _test_hooks.get_logger = _fake_get_logger

    with pytest.raises(JSONTypeError):
        _decode_int({"bad": "x"}, "bad", 0)
    assert called, "expected logger.error to be invoked"


def test_decode_float_logs_and_raises_on_invalid() -> None:
    called: list[str] = []
    stub = _LogStub(called)

    def _fake_get_logger(name: str) -> LoggerInstanceProtocol:
        _ = name
        return stub

    _test_hooks.get_logger = _fake_get_logger

    with pytest.raises(JSONTypeError):
        _decode_float({"bad": "x"}, "bad", 0.0)
    assert called, "expected logger.error to be invoked"


def test_read_cache_missing_required_fields_raises(tmp_path: Path) -> None:
    p = tmp_path / "cache.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(JSONTypeError):
        _ = _read_cache(p)


def test_decode_obj_dict_non_dict_returns_none() -> None:
    import handwriting_ai.training.calibration.cache as c

    # Passing a non-dict JSONValue should return None
    out = c._decode_obj_dict(123)
    assert out is None


def test_decode_obj_dict_pass_through_for_dict() -> None:
    from platform_core.json_utils import JSONValue

    import handwriting_ai.training.calibration.cache as c

    src: dict[str, JSONValue] = {"a": 1, "b": 2}
    out = c._decode_obj_dict(src)
    assert out == src
