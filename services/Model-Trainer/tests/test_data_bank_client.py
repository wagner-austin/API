from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Final

import httpx
import pytest
from platform_core.data_bank_client import (
    AuthorizationError,
    BadRequestError,
    ConflictError,
    DataBankClient,
    DataBankClientError,
    ForbiddenError,
    HeadInfo,
    InsufficientStorageClientError,
    NotFoundError,
    RangeNotSatisfiableError,
)


def _sha256(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def test_client_head_and_download_resume_and_verify(tmp_path: Path) -> None:
    payload: Final[bytes] = b"hello world" * 10
    etag: Final[str] = _sha256(payload)
    base = "http://db"

    # Simulate HEAD and GET with Range support based on request headers
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD" and request.url.path.startswith("/files/"):
            return httpx.Response(
                200,
                headers={
                    "Content-Length": str(len(payload)),
                    "Content-Type": "application/octet-stream",
                    "ETag": etag,
                },
            )
        if request.method == "GET" and request.url.path.startswith("/files/"):
            start = 0
            hdrs: dict[str, str] = {k.lower(): v for (k, v) in request.headers.items()}
            rng_header = hdrs.get("range")
            if rng_header is not None and rng_header.startswith("bytes="):
                start = int(rng_header.split("=", 1)[1].split("-", 1)[0])
            part = payload[start:]
            return httpx.Response(200, content=part)
        return httpx.Response(405)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    c = DataBankClient(base_url=base, api_key="k", client=client)

    # HEAD
    h = c.head("fid", request_id="rid")
    assert h["size"] == len(payload) and h["etag"] == etag

    # Download fresh
    dest = tmp_path / "file.bin"
    c.download_to_path("fid", dest, request_id="rid", verify_etag=True)
    assert dest.read_bytes() == payload

    # Resume no-op (already complete) exercises verify branch
    c.download_to_path("fid", dest, request_id="rid", verify_etag=True)

    # Truncate and resume appending
    part = dest.stat().st_size - 5
    dest.write_bytes(dest.read_bytes()[:part])
    c.download_to_path("fid", dest, request_id="rid", verify_etag=True)
    assert dest.read_bytes() == payload


def test_client_error_mappings_and_retry(tmp_path: Path) -> None:
    # Sequence: first HEAD 500 then 200 to trigger retry; then GET 416 mapping
    calls = {"head": 0}
    payload = b"abc"
    etag = _sha256(payload)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            calls["head"] += 1
            if calls["head"] == 1:
                return httpx.Response(500)
            return httpx.Response(200, headers={"Content-Length": str(len(payload)), "ETag": etag})
        if request.method == "GET":
            # Simulate 416 by returning explicit status
            return httpx.Response(416)
        return httpx.Response(405)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    c = DataBankClient(
        base_url="http://db", api_key="x", client=client, retries=1, backoff_seconds=0
    )
    # Retry occurs and head succeeds
    _ = c.head("fid")
    # GET path raises 416
    with pytest.raises(RangeNotSatisfiableError):
        c.download_to_path("fid", tmp_path / "x.bin")


def test_client_auth_forbidden_notfound_and_badrequest(tmp_path: Path) -> None:
    # Map 401/403/404/400
    def handler(request: httpx.Request) -> httpx.Response:
        hdrs: dict[str, str] = dict(request.headers.items())
        code_map: dict[str, int] = {"401": 401, "403": 403, "404": 404, "400": 400}
        code = code_map.get(hdrs.get("X-Test", ""), 200)
        return httpx.Response(code)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    c = DataBankClient(base_url="http://db", api_key="x", client=client)
    with pytest.raises(AuthorizationError):
        c._raise_for_error(httpx.Response(401))
    with pytest.raises(ForbiddenError):
        c._raise_for_error(httpx.Response(403))
    with pytest.raises(NotFoundError):
        c._raise_for_error(httpx.Response(404))
    with pytest.raises(BadRequestError):
        c._raise_for_error(httpx.Response(400))
    with pytest.raises(ConflictError):
        c._raise_for_error(httpx.Response(409))


def test_client_upload_success_and_missing_file_id(tmp_path: Path) -> None:
    # Upload returns JSON with file_id, and then missing file_id branch
    body_ok = (
        b'{"file_id":"deadbeef","size":1,"sha256":"deadbeef","content_type":"application/x-tar"}'
    )
    body_bad = b"{}"
    calls = {"post": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/files":
            calls["post"] += 1
            return httpx.Response(201, content=body_ok if calls["post"] == 1 else body_bad)
        if request.method == "HEAD":
            # Upload() performs separate HEAD after POST in some clients; this client does not
            return httpx.Response(200, headers={"Content-Length": "0", "ETag": ""})
        return httpx.Response(405)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    c = DataBankClient(base_url="http://db", api_key="x", client=client)

    # Success
    import io

    fid = c.upload(
        "m.tar",
        io.BytesIO(b"tar"),
        content_type="application/x-tar",
        request_id="rid",
    )["file_id"]
    assert fid == "deadbeef"
    # Missing file_id
    with pytest.raises(DataBankClientError):
        _ = c.upload(
            "m2.tar", io.BytesIO(b"tar2"), content_type="application/x-tar", request_id="rid"
        )


def test_client_upload_blank_file_id_raises(tmp_path: Path) -> None:
    body_blank = b'{"file_id":"  "}'

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/files":
            return httpx.Response(201, content=body_blank)
        return httpx.Response(405)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    c = DataBankClient(base_url="http://db", api_key="x", client=client)

    with pytest.raises(DataBankClientError):
        _ = c.upload(
            "m3.tar", io.BytesIO(b"tar3"), content_type="application/x-tar", request_id="rid"
        )


def test_client_upload_http_error_propagates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/files":
            return httpx.Response(507, text="no space")
        return httpx.Response(405)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    c = DataBankClient(base_url="http://db", api_key="x", client=client)
    with pytest.raises(InsufficientStorageClientError):
        _ = c.upload(
            "m4.tar", io.BytesIO(b"tar4"), content_type="application/x-tar", request_id="rid"
        )


def test_client_upload_invalid_json_response_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/files":
            return httpx.Response(201, content=b"[]")  # not an object
        return httpx.Response(405)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    c = DataBankClient(base_url="http://db", api_key="x", client=client)
    with pytest.raises(DataBankClientError):
        _ = c.upload(
            "m5.tar", io.BytesIO(b"tar5"), content_type="application/x-tar", request_id="rid"
        )


def test_client_request_transport_error_retry_then_success() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.TransportError("network-down")
        return httpx.Response(200)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    c = DataBankClient(
        base_url="http://db", api_key="x", client=client, retries=1, backoff_seconds=0
    )
    info = c.head("fid")
    assert info["size"] == 0


def test_client_request_transport_error_no_retry() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TransportError("network-down")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    c = DataBankClient(base_url="http://db", api_key="x", client=client, retries=0)
    with pytest.raises(DataBankClientError):
        _ = c.head("fid")


def test_client_error_mapping_insufficient_storage_and_generic() -> None:
    c = DataBankClient(base_url="http://db", api_key="x", client=httpx.Client())
    with pytest.raises(InsufficientStorageClientError):
        c._raise_for_error(httpx.Response(507, text="no space"))
    with pytest.raises(DataBankClientError):
        c._raise_for_error(httpx.Response(500, text="server error"))
    # Successful responses should be ignored by the helper
    c._raise_for_error(httpx.Response(200, text="ok"))


def test_client_head_error_uses_raise_for_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(404)
        return httpx.Response(405)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    c = DataBankClient(base_url="http://db", api_key="x", client=client)
    with pytest.raises(NotFoundError):
        c.head("missing-id")


def test_client_download_local_larger_than_remote(tmp_path: Path) -> None:
    class _Client(DataBankClient):
        def __init__(self: _Client) -> None:
            super().__init__(base_url="http://db", api_key="x", client=httpx.Client())

        def head(self: _Client, file_id: str, *, request_id: str | None = None) -> HeadInfo:
            return {"size": 3, "etag": _sha256(b"abc"), "content_type": "application/octet-stream"}

    dest = tmp_path / "file.bin"
    dest.write_bytes(b"abcdef")  # six bytes, larger than remote
    c = _Client()
    with pytest.raises(RangeNotSatisfiableError):
        c.download_to_path("fid", dest, request_id="rid")


def test_client_download_already_complete_no_verify(tmp_path: Path) -> None:
    class _Client(DataBankClient):
        def __init__(self: _Client) -> None:
            super().__init__(base_url="http://db", api_key="x", client=httpx.Client())

        def head(self: _Client, file_id: str, *, request_id: str | None = None) -> HeadInfo:
            return {"size": 3, "etag": _sha256(b"abc"), "content_type": "application/octet-stream"}

    dest = tmp_path / "file2.bin"
    dest.write_bytes(b"abc")
    c = _Client()
    # When verify_etag is False, this should return without raising
    info = c.download_to_path("fid", dest, request_id="rid", verify_etag=False)
    assert info["size"] == 3


def test_client_download_stream_no_verify(tmp_path: Path) -> None:
    payload: Final[bytes] = b"hello" * 5
    etag: Final[str] = _sha256(payload)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(
                200,
                headers={
                    "Content-Length": str(len(payload)),
                    "Content-Type": "application/octet-stream",
                    "ETag": etag,
                },
            )
        if request.method == "GET":
            return httpx.Response(200, content=payload)
        return httpx.Response(405)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    c = DataBankClient(base_url="http://db", api_key="x", client=client)

    dest = tmp_path / "file-stream.bin"
    info = c.download_to_path("fid", dest, request_id="rid", verify_etag=False)
    assert info["size"] == len(payload)
    assert dest.read_bytes() == payload


def test_verify_file_etag_mismatch_raises(tmp_path: Path) -> None:
    dest = tmp_path / "file3.bin"
    dest.write_bytes(b"xyz")
    with pytest.raises(DataBankClientError):
        DataBankClient._verify_file_etag(dest, expected_etag=_sha256(b"abc"))
