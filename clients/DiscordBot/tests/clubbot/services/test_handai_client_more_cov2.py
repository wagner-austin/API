from __future__ import annotations

from collections.abc import Mapping

import pytest
from platform_core.http_client import HttpxResponse
from platform_core.json_utils import JSONValue, dump_json_str

from clubbot.services.handai.client import (
    HandwritingAPIError,
    PredictResult,
    _decode_predict_result,
    _shape_api_error,
)


def test_predict_result_getitem_and_key_error() -> None:
    pr = PredictResult(
        digit=5, confidence=0.9, probs=(0.1, 0.9), model_id="m", uncertain=False, latency_ms=10
    )
    assert pr["digit"] == 5
    assert pr["confidence"] == 0.9
    assert pr["probs"] == (0.1, 0.9)
    assert pr["model_id"] == "m"
    assert pr["uncertain"] is False
    assert pr["latency_ms"] == 10
    with pytest.raises(KeyError):
        _ = pr["nope"]


def test_decode_predict_result_coerces_types() -> None:
    obj: dict[str, JSONValue] = {
        "digit": "7",
        "confidence": "0.75",
        "probs": [1, 2.5, "3.5"],
        "model_id": 123,  # coerced to str
        "uncertain": True,
        "latency_ms": "15",
    }
    pr = _decode_predict_result(obj)
    assert pr.digit == 7 and pr.confidence == 0.75 and pr.latency_ms == 15
    assert pr.model_id == "123" and pr.uncertain is True


class _Resp(HttpxResponse):
    def __init__(
        self,
        status: int,
        *,
        headers: Mapping[str, str] | None,
        json_obj: JSONValue | None,
        text: str,
    ) -> None:
        self.status_code = status
        self.headers: Mapping[str, str] = dict(headers or {})
        self._json = json_obj
        self.text = text
        self.content = text.encode()

    def json(self) -> JSONValue:
        if self._json is None:
            raise ValueError("no json")
        return self._json


def test_shape_api_error_extracts_fields_from_json() -> None:
    payload: JSONValue = {"code": "INVALID_INPUT", "message": "bad", "request_id": "rid"}
    resp = _Resp(400, headers={}, json_obj=payload, text=dump_json_str(payload))
    e = _shape_api_error(resp)
    assert type(e) is HandwritingAPIError
    assert e.status == 400 and e.code == "INVALID_INPUT"
    assert "bad" in str(e) and e.request_id == "rid"


def test_shape_api_error_uses_text_when_non_json() -> None:
    resp = _Resp(503, headers={"X-Request-ID": "hdr"}, json_obj=None, text="SERVICE DOWN")
    e = _shape_api_error(resp)
    assert e.status == 503 and e.request_id == "hdr" and "HTTP 503" in str(e)
