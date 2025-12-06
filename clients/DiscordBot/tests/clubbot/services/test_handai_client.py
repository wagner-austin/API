from __future__ import annotations

import logging
from collections.abc import Mapping

import pytest
from platform_core.http_client import HttpxResponse
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str

from clubbot.services.handai.client import (
    HandwritingAPIError,
    HandwritingClient,
    _shape_api_error,
    _top_k_indices,
)


class _FakeResponse:
    """Protocol-compliant fake response for testing."""

    def __init__(
        self,
        status: int,
        json_body: JSONValue | None = None,
        content: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status_code = int(status)
        self._json = json_body
        if content is not None:
            self.content: bytes | bytearray = content
            self.text = content.decode("utf-8", errors="replace")
        elif json_body is not None:
            self.text = dump_json_str(json_body)
            self.content = self.text.encode("utf-8")
        else:
            self.text = ""
            self.content = b""
        self.headers: Mapping[str, str] = dict(headers) if headers else {}

    def json(self) -> JSONValue:
        if self._json is None:
            return load_json_str(self.text)
        return self._json


class TimeoutError(Exception):
    """Timeout error for testing."""


class ConnectError(Exception):
    """Connection error for testing."""


class _FakeClient:
    """Protocol-compliant fake async HTTP client for testing."""

    def __init__(
        self,
        response: _FakeResponse | None = None,
        raise_timeout: bool = False,
        raise_connect: bool = False,
        timeout_count: int = 0,
    ) -> None:
        self._response = response
        self._raise_timeout = raise_timeout
        self._raise_connect = raise_connect
        self._timeout_count = timeout_count
        self._call_count = 0

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
        self._call_count += 1
        if self._raise_connect:
            raise ConnectError("boom")
        if self._raise_timeout or self._call_count <= self._timeout_count:
            raise TimeoutError("timeout")
        if self._response is None:
            raise RuntimeError("No response configured")
        return self._response

    async def get(self, url: str, *, headers: Mapping[str, str]) -> HttpxResponse:
        return await self.post(url, headers=headers, json=None, files=None)


@pytest.mark.asyncio
async def test_client_success() -> None:
    body: JSONValue = {
        "digit": 7,
        "confidence": 0.987,
        "probs": [1.0 if i == 7 else 0.0 for i in range(10)],
        "model_id": "mnist_resnet18_v1",
        "visual_png_b64": None,
        "uncertain": False,
        "latency_ms": 12,
    }
    client = HandwritingClient(
        base_url="http://example",
        api_key=None,
        client=_FakeClient(response=_FakeResponse(200, body)),
    )
    res = await client.read_digit(
        data=b"png",
        filename="x.png",
        content_type="image/png",
        request_id="req1",
    )
    assert res.digit == 7 and res.model_id == "mnist_resnet18_v1"
    await client.aclose()


@pytest.mark.asyncio
async def test_client_unsupported_media_maps_error() -> None:
    body: JSONValue = {
        "code": "unsupported_media_type",
        "message": "Only PNG and JPEG are supported",
        "request_id": "r",
    }
    client = HandwritingClient(
        base_url="http://example",
        api_key=None,
        client=_FakeClient(response=_FakeResponse(415, body)),
    )
    with pytest.raises(HandwritingAPIError) as ei:
        await client.read_digit(
            data=b"x",
            filename="x.txt",
            content_type="text/plain",
            request_id="req2",
        )
    e = ei.value
    assert e.status == 415 and (e.code == "unsupported_media_type" or e.code is None)
    await client.aclose()


@pytest.mark.asyncio
async def test_client_retry_on_timeout_propagates() -> None:
    client = HandwritingClient(
        base_url="http://example",
        api_key=None,
        timeout_seconds=1,
        max_retries=1,
        client=_FakeClient(response=_FakeResponse(200, {}), timeout_count=1),
    )
    with pytest.raises(TimeoutError):
        await client.read_digit(
            data=b"x", filename="x.png", content_type="image/png", request_id="r"
        )
    await client.aclose()


@pytest.mark.asyncio
async def test_client_timeout_exhausted_propagates() -> None:
    client = HandwritingClient(
        base_url="http://example",
        api_key=None,
        timeout_seconds=1,
        max_retries=1,
        client=_FakeClient(raise_timeout=True),
    )
    with pytest.raises(TimeoutError):
        await client.read_digit(
            data=b"x", filename="x.png", content_type="image/png", request_id="r"
        )
    await client.aclose()


@pytest.mark.asyncio
async def test_client_request_error_propagates() -> None:
    client = HandwritingClient(
        base_url="http://example",
        api_key=None,
        timeout_seconds=1,
        max_retries=0,
        client=_FakeClient(raise_connect=True),
    )
    with pytest.raises(ConnectError):
        await client.read_digit(
            data=b"x", filename="x.png", content_type="image/png", request_id="r"
        )
    await client.aclose()


@pytest.mark.asyncio
async def test_client_invalid_json_maps_invalid_body() -> None:
    client = HandwritingClient(
        base_url="http://example",
        api_key=None,
        timeout_seconds=1,
        max_retries=0,
        client=_FakeClient(response=_FakeResponse(200, content=b"not-json")),
    )
    from platform_core.json_utils import InvalidJsonError

    with pytest.raises(InvalidJsonError):
        await client.read_digit(
            data=b"x", filename="x.png", content_type="image/png", request_id="r"
        )
    await client.aclose()


def test_shape_api_error_no_json_defaults_message() -> None:
    resp = _FakeResponse(503, content=b"bad", headers={"X-Request-ID": "rid"})
    err = _shape_api_error(resp)
    assert err.status == 503 and err.request_id == "rid" and "HTTP 503" in str(err)


def test_top_k_indices_basic() -> None:
    assert _top_k_indices([0.1, 0.3, 0.2], k=2) == [1, 2]
    assert _top_k_indices([], k=3) == []


logger = logging.getLogger(__name__)
