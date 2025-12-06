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


def test_shape_api_error_request_id_header_precedence_when_json_non_string() -> None:
    payload: JSONValue = {"message": "bad", "request_id": 123}
    resp = _FakeResponse(502, payload, headers={"X-Request-ID": "hdr-rid"})
    err = _shape_api_error(resp)
    # JSON request_id is not a str, so header value should be retained
    assert err.request_id == "hdr-rid"


logger = logging.getLogger(__name__)
