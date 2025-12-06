from __future__ import annotations

import logging
from collections.abc import Mapping

from platform_core.json_utils import JSONValue, dump_json_str

from clubbot.services.handai.client import _shape_api_error


class _FakeResponse:
    """Protocol-compliant fake response for testing."""

    def __init__(
        self,
        status: int,
        json_body: JSONValue | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status_code = int(status)
        self._json = json_body
        if json_body is not None:
            self.text = dump_json_str(json_body)
            self.content: bytes | bytearray = self.text.encode("utf-8")
        else:
            self.text = ""
            self.content = b""
        self.headers: Mapping[str, str] = dict(headers) if headers else {}

    def json(self) -> JSONValue:
        if self._json is None:
            raise ValueError("No JSON body")
        return self._json


def test_shape_api_error_numeric_code_ignored() -> None:
    # JSON has a non-string code; code should remain None, message should be taken
    payload: JSONValue = {"message": "bad req", "code": 123}
    resp = _FakeResponse(400, payload)
    err = _shape_api_error(resp)
    assert err.code is None and str(err) and err.status == 400


logger = logging.getLogger(__name__)
