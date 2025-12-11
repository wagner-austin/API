from __future__ import annotations

import logging
from collections.abc import Mapping

import pytest
from platform_core.http_client import HttpxResponse
from platform_core.json_utils import JSONValue, dump_json_str

from clubbot.services.handai.client import HandwritingAPIError, HandwritingClient, _shape_api_error


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


class RequestError(Exception):
    """Request error for testing retries."""


def _predict_body() -> JSONValue:
    return {
        "digit": 3,
        "confidence": 0.9,
        "probs": [0.1] * 10,
        "model_id": "m",
        "uncertain": False,
        "latency_ms": 12,
    }


class _FakeClientWithHeaders:
    """Fake client that captures headers."""

    def __init__(self) -> None:
        self.seen_headers: dict[str, str] = {}

    async def aclose(self) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JSONValue | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
    ) -> HttpxResponse:
        self.seen_headers.update(headers)
        return _FakeResponse(200, _predict_body())

    async def get(self, url: str, *, headers: Mapping[str, str]) -> HttpxResponse:
        return _FakeResponse(200, _predict_body())


class _FakeClientAlwaysFails:
    """Fake client that always raises."""

    def __init__(self) -> None:
        self.call_count = 0

    async def aclose(self) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JSONValue | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
    ) -> HttpxResponse:
        self.call_count += 1
        raise RequestError("boom")

    async def get(self, url: str, *, headers: Mapping[str, str]) -> HttpxResponse:
        raise RequestError("boom")


class _FakeClientListResponse:
    """Fake client that returns a JSON list."""

    async def aclose(self) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JSONValue | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
    ) -> HttpxResponse:
        list_body: JSONValue = [1, 2, 3]
        return _FakeResponse(200, list_body)

    async def get(self, url: str, *, headers: Mapping[str, str]) -> HttpxResponse:
        list_body: JSONValue = [1, 2, 3]
        return _FakeResponse(200, list_body)


@pytest.mark.asyncio
async def test_headers_include_api_key() -> None:
    fake_client = _FakeClientWithHeaders()
    hc = HandwritingClient(
        base_url="http://x",
        api_key="sekrit",
        timeout_seconds=1,
        max_retries=0,
        client=fake_client,
    )
    out = await hc.read_digit(
        data=b"x",
        filename="x.png",
        content_type="image/png",
        request_id="r",
    )
    assert out.digit == 3 and fake_client.seen_headers.get("X-Api-Key") == "sekrit"
    await hc.aclose()


@pytest.mark.asyncio
async def test_request_error_propagates_without_retry() -> None:
    """Test that exceptions propagate immediately when max_retries=0."""
    fake_client = _FakeClientAlwaysFails()
    hc = HandwritingClient(
        base_url="http://x",
        api_key=None,
        timeout_seconds=1,
        max_retries=0,
        client=fake_client,
    )
    with pytest.raises(RequestError):
        await hc.read_digit(
            data=b"x",
            filename="x.png",
            content_type="image/png",
            request_id="r",
        )
    assert fake_client.call_count == 1
    await hc.aclose()


@pytest.mark.asyncio
async def test_invalid_response_body_non_dict_raises() -> None:
    hc = HandwritingClient(
        base_url="http://x",
        api_key=None,
        timeout_seconds=1,
        max_retries=0,
        client=_FakeClientListResponse(),
    )
    with pytest.raises(HandwritingAPIError) as ei:
        await hc.read_digit(
            data=b"x",
            filename="x.png",
            content_type="image/png",
            request_id="r",
        )
    assert "Invalid response body" in str(ei.value)
    await hc.aclose()


def test_shape_api_error_json_list_defaults() -> None:
    list_body: JSONValue = [1, 2, 3]
    resp = _FakeResponse(503, list_body, headers={"X-Request-ID": "rid2"})
    err = _shape_api_error(resp)
    assert err.status == 503 and err.request_id == "rid2" and "HTTP 503" in str(err)


def test_shape_api_error_mixed_types_fall_back() -> None:
    # Non-string structured fields should not override defaults
    body: JSONValue = {"code": 123, "message": 456, "request_id": 789}
    resp = _FakeResponse(500, body)
    err = _shape_api_error(resp)
    assert err.status == 500
    # Message falls back to HTTP status string
    assert "HTTP 500" in str(err)


def test_shape_api_error_object_text_non_dict_body() -> None:
    class _ListResponse:
        def __init__(self, text: str, body: JSONValue, status: int) -> None:
            self.text = text
            self.status_code = status
            self.headers: Mapping[str, str] = {}
            self._body = body
            self.content: bytes | bytearray = text.encode("utf-8")

        def json(self) -> JSONValue:
            return self._body

    list_body: JSONValue = ["x"]
    fake = _ListResponse(text="{ still_not_real_json }", body=list_body, status=502)
    err = _shape_api_error(fake)
    assert err.status == 502 and "HTTP 502" in str(err)


logger = logging.getLogger(__name__)
