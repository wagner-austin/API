"""Tests for HTTP fake implementations in testing.py."""

from __future__ import annotations

import pytest

from platform_core.testing import (
    FakeHttpxAsyncClient,
    FakeHttpxAsyncClientRaises,
    FakeHttpxClient,
    FakeHttpxClientRaises,
    FakeHttpxModule,
    FakeHttpxModuleSyncOnly,
    FakeHttpxResponse,
    FakeTimeout,
    make_async_client_ctor,
    make_client_ctor,
    make_timeout_ctor,
)


def test_fake_timeout_repr() -> None:
    t = FakeTimeout(30.0)
    assert repr(t) == "Timeout(30.0)"


def test_fake_response_from_json_body() -> None:
    resp = FakeHttpxResponse(200, {"key": "value"})
    assert resp.status_code == 200
    assert resp.json() == {"key": "value"}
    assert "key" in resp.text
    assert b"key" in resp.content
    assert resp.headers == {}


def test_fake_response_from_content() -> None:
    resp = FakeHttpxResponse(201, content=b"raw bytes")
    assert resp.status_code == 201
    assert resp.content == b"raw bytes"
    assert resp.text == "raw bytes"


def test_fake_response_from_content_with_text() -> None:
    resp = FakeHttpxResponse(200, content=b"raw", text="override")
    assert resp.content == b"raw"
    assert resp.text == "override"


def test_fake_response_from_text() -> None:
    resp = FakeHttpxResponse(200, text="text only")
    assert resp.text == "text only"
    assert resp.content == b"text only"


def test_fake_response_empty() -> None:
    resp = FakeHttpxResponse(204)
    assert resp.text == ""
    assert resp.content == b""


def test_fake_response_json_parse_from_text() -> None:
    resp = FakeHttpxResponse(200, text='{"parsed": true}')
    assert resp.json() == {"parsed": True}


def test_fake_response_with_headers() -> None:
    resp = FakeHttpxResponse(200, headers={"X-Custom": "value"})
    assert resp.headers["X-Custom"] == "value"


@pytest.mark.asyncio
async def test_fake_async_client_post() -> None:
    resp = FakeHttpxResponse(200, {"ok": True})
    client = FakeHttpxAsyncClient(resp)
    result = await client.post("http://test/path", headers={"Auth": "key"})
    assert result.status_code == 200
    assert client.seen_urls == ["http://test/path"]
    assert client.seen_headers["Auth"] == "key"
    assert client.call_count == 1


@pytest.mark.asyncio
async def test_fake_async_client_get() -> None:
    resp = FakeHttpxResponse(200, {"ok": True})
    client = FakeHttpxAsyncClient(resp)
    result = await client.get("http://test/get", headers={"X": "Y"})
    assert result.status_code == 200
    assert client.seen_urls == ["http://test/get"]


@pytest.mark.asyncio
async def test_fake_async_client_aclose() -> None:
    client = FakeHttpxAsyncClient(FakeHttpxResponse(200))
    await client.aclose()


@pytest.mark.asyncio
async def test_fake_async_client_no_response_raises() -> None:
    client = FakeHttpxAsyncClient(None)
    with pytest.raises(RuntimeError, match="No response configured"):
        await client.post("http://test", headers={})


@pytest.mark.asyncio
async def test_fake_async_client_exception_always() -> None:
    exc = ValueError("test error")
    client = FakeHttpxAsyncClient(None, exception_to_raise=exc, exception_count=0)
    with pytest.raises(ValueError, match="test error"):
        await client.post("http://test", headers={})


@pytest.mark.asyncio
async def test_fake_async_client_exception_count() -> None:
    resp = FakeHttpxResponse(200)
    exc = ValueError("transient")
    client = FakeHttpxAsyncClient(resp, exception_to_raise=exc, exception_count=2)

    with pytest.raises(ValueError):
        await client.post("http://test", headers={})
    with pytest.raises(ValueError):
        await client.post("http://test", headers={})
    result = await client.post("http://test", headers={})
    assert result.status_code == 200


@pytest.mark.asyncio
async def test_fake_async_client_raises_post() -> None:
    exc = ConnectionError("fail")
    client = FakeHttpxAsyncClientRaises(exc)
    await client.aclose()
    with pytest.raises(ConnectionError, match="fail"):
        await client.post("http://x", headers={})


@pytest.mark.asyncio
async def test_fake_async_client_raises_get() -> None:
    exc = TimeoutError("timeout")
    client = FakeHttpxAsyncClientRaises(exc)
    with pytest.raises(TimeoutError, match="timeout"):
        await client.get("http://x", headers={})


def test_fake_sync_client_post() -> None:
    resp = FakeHttpxResponse(201, {"created": True})
    client = FakeHttpxClient(resp)
    result = client.post("http://api/create", headers={"Token": "abc"})
    assert result.status_code == 201
    assert client.seen_urls == ["http://api/create"]
    assert client.seen_headers["Token"] == "abc"


def test_fake_sync_client_close() -> None:
    client = FakeHttpxClient(FakeHttpxResponse(200))
    client.close()


def test_fake_sync_client_no_response_raises() -> None:
    client = FakeHttpxClient(None)
    with pytest.raises(RuntimeError, match="No response configured"):
        client.post("http://test", headers={})


def test_fake_sync_client_raises_post() -> None:
    exc = OSError("network")
    client = FakeHttpxClientRaises(exc)
    client.close()
    with pytest.raises(OSError, match="network"):
        client.post("http://x", headers={})


def test_make_timeout_ctor() -> None:
    ctor = make_timeout_ctor()
    t = ctor(15.0)
    assert repr(t) == "Timeout(15.0)"


def test_make_async_client_ctor() -> None:
    resp = FakeHttpxResponse(200)
    ctor = make_async_client_ctor(resp)
    timeout = FakeTimeout(10.0)
    client = ctor(timeout=timeout)
    assert client.__class__.__name__ == "FakeHttpxAsyncClient"


def test_make_client_ctor() -> None:
    resp = FakeHttpxResponse(200)
    ctor = make_client_ctor(resp)
    timeout = FakeTimeout(10.0)
    client = ctor(timeout=timeout)
    assert client.__class__.__name__ == "FakeHttpxClient"


def test_fake_httpx_module_sync() -> None:
    resp = FakeHttpxResponse(200)
    _ = FakeHttpxModule(resp, async_client=False)
    # Verify factory helpers can be used to create clients
    timeout = make_timeout_ctor()(5.0)
    client = make_client_ctor(resp)(timeout=timeout)
    assert client.post("http://test", headers={}).status_code == 200


def test_fake_httpx_module_async() -> None:
    resp = FakeHttpxResponse(200)
    _ = FakeHttpxModule(resp, async_client=True)


def test_fake_httpx_module_sync_only() -> None:
    resp = FakeHttpxResponse(200)
    _ = FakeHttpxModuleSyncOnly(resp)
