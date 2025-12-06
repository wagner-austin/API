from __future__ import annotations

import io
import os
from hashlib import sha256
from pathlib import Path

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
    _decode_upload_response,
)
from platform_core.json_utils import JSONValue, dump_json_str


class _MemStore:
    def __init__(self) -> None:
        self._files: dict[str, bytes] = {}
        self._ctype: dict[str, str] = {}

    def put(self, fid: str, data: bytes, ctype: str) -> None:
        self._files[fid] = data
        self._ctype[fid] = ctype

    def get(self, fid: str) -> tuple[bytes, str]:
        if fid not in self._files:
            raise KeyError(fid)
        return self._files[fid], self._ctype.get(fid, "application/octet-stream")

    def delete(self, fid: str) -> bool:
        if fid in self._files:
            del self._files[fid]
            self._ctype.pop(fid, None)
            return True
        return False

    def exists(self, fid: str) -> bool:
        return fid in self._files


class _MockServer:
    def __init__(self, store: _MemStore, expect_key: str) -> None:
        self._store = store
        self._key = expect_key
        self._counts: dict[str, int] = {}

    @staticmethod
    def _unauth() -> httpx.Response:
        body = {"code": "UNAUTHORIZED", "message": "missing/invalid", "request_id": None}
        return httpx.Response(401, text=dump_json_str(body))

    @staticmethod
    def _not_found() -> httpx.Response:
        body = {"code": "NOT_FOUND", "message": "not found", "request_id": None}
        return httpx.Response(404, text=dump_json_str(body))

    def _post_files(self, request: httpx.Request) -> httpx.Response:
        content = request.content
        hdrs = {k.lower(): v for (k, v) in request.headers.items()}
        ct_header = hdrs.get("content-type", "")
        boundary_key = "boundary="
        bpos = ct_header.find(boundary_key)
        assert bpos != -1
        boundary = ct_header[bpos + len(boundary_key) :].strip('"')
        bbytes = ("--" + boundary).encode("latin-1")
        txt = content.decode("latin-1", errors="ignore")
        idx = txt.find("filename=")
        assert idx != -1
        q1 = txt.find('"', idx)
        q2 = txt.find('"', q1 + 1)
        fid = txt[q1 + 1 : q2]
        bpos0 = content.find(bbytes)
        assert bpos0 != -1
        start = content.find(b"\r\n\r\n", bpos0)
        assert start != -1
        start += 4
        end = content.find(b"\r\n" + bbytes, start)
        assert end != -1
        data = content[start:end]
        ctype_line_start = txt.find("Content-Type:", q2)
        ctype = "application/octet-stream"
        if ctype_line_start != -1:
            ctype_line_end = txt.find("\r\n", ctype_line_start)
            ctype = txt[ctype_line_start:ctype_line_end].split(":", 1)[1].strip()
        if len(data) > 0 or not self._store.exists(fid):
            self._store.put(fid, data, ctype)
        body = {
            "file_id": fid,
            "size": len(data),
            "sha256": sha256(data).hexdigest(),
            "content_type": ctype,
            "created_at": None,
        }
        return httpx.Response(201, text=dump_json_str(body))

    def _head_file(self, fid: str, headers: dict[str, str]) -> httpx.Response:
        if fid == "conflict":
            return httpx.Response(409, text="conflict")
        if fid == "bad400":
            body = {"code": "BAD_REQUEST", "message": "bad", "request_id": None}
            return httpx.Response(400, text=dump_json_str(body))
        if fid == "bad403":
            body = {"code": "FORBIDDEN", "message": "no", "request_id": None}
            return httpx.Response(403, text=dump_json_str(body))
        if fid == "bad507":
            body = {"code": "INSUFFICIENT_STORAGE", "message": "low", "request_id": None}
            return httpx.Response(507, text=dump_json_str(body))
        if fid == "err502":
            body = {"code": "ERROR", "message": "bad gateway", "request_id": None}
            return httpx.Response(502, text=dump_json_str(body))
        if fid == "checkrid" and "x-request-id" not in headers:
            body = {"code": "E", "message": "missing rid", "request_id": None}
            return httpx.Response(500, text=dump_json_str(body))
        if fid == "retryme":
            count = self._counts.get(fid, 0) + 1
            self._counts[fid] = count
            if count <= 2:
                body = {"code": "E", "message": "retry", "request_id": None}
                return httpx.Response(500, text=dump_json_str(body))
        if fid not in self._store._files:
            return self._not_found()
        data, ctype = self._store.get(fid)
        headers_out = {
            "Content-Length": str(len(data)),
            "ETag": sha256(data).hexdigest(),
            "Content-Type": ctype,
        }
        return httpx.Response(200, headers=headers_out)

    def _get_info(self, fid: str) -> httpx.Response:
        if fid not in self._store._files:
            return self._not_found()
        data, ctype = self._store.get(fid)
        body = {
            "file_id": fid,
            "size": len(data),
            "sha256": sha256(data).hexdigest(),
            "content_type": ctype,
        }
        return httpx.Response(200, text=dump_json_str(body))

    def _get_file(self, fid: str, rng: str | None) -> httpx.Response:
        if fid == "err416":
            body = {"code": "RANGE_NOT_SATISFIABLE", "message": "bad", "request_id": None}
            headers = {"Content-Range": "bytes */10"}
            return httpx.Response(416, text=dump_json_str(body), headers=headers)
        if fid not in self._store._files:
            return self._not_found()
        data, ctype = self._store.get(fid)
        if rng is None:
            headers = {"Content-Length": str(len(data)), "Content-Type": ctype}
            return httpx.Response(200, content=data, headers=headers)
        if not rng.startswith("bytes="):
            body = {"code": "INVALID_RANGE", "message": "invalid range", "request_id": None}
            return httpx.Response(416, text=dump_json_str(body))
        start_s = rng[len("bytes=") :].split("-")[0]
        try:
            start = int(start_s) if start_s != "" else 0
        except ValueError:
            body = {"code": "INVALID_RANGE", "message": "invalid range", "request_id": None}
            return httpx.Response(416, text=dump_json_str(body))
        if start >= len(data):
            headers = {"Content-Range": f"bytes */{len(data)}"}
            body = {"code": "RANGE_NOT_SATISFIABLE", "message": "unsat", "request_id": None}
            return httpx.Response(416, text=dump_json_str(body), headers=headers)
        part = data[start:]
        headers = {
            "Content-Length": str(len(part)),
            "Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}",
        }
        return httpx.Response(206, content=part, headers=headers)

    def _delete(self, fid: str) -> httpx.Response:
        if not self._store.delete(fid):
            return self._not_found()
        return httpx.Response(204)

    def handle(self, request: httpx.Request) -> httpx.Response:
        hdrs: dict[str, str] = {k.lower(): v for (k, v) in request.headers.items()}
        hdr = hdrs.get("x-api-key")
        if hdr != self._key:
            return self._unauth()
        path = request.url.path
        if path == "/files" and request.method == "POST":
            return self._post_files(request)
        if request.method == "HEAD" and path.startswith("/files/"):
            fid = path.split("/")[-1]
            return self._head_file(fid, hdrs)
        if request.method == "GET" and path.endswith("/info"):
            return self._get_info(path.split("/")[-2])
        if request.method == "GET" and path.startswith("/files/"):
            rng_s = hdrs.get("range")
            return self._get_file(path.split("/")[-1], rng_s)
        if request.method == "DELETE" and path.startswith("/files/"):
            return self._delete(path.split("/")[-1])
        body = {"code": "ERROR", "message": "unhandled", "request_id": None}
        return httpx.Response(500, text=dump_json_str(body))


def _mock_transport(store: _MemStore, expect_key: str) -> httpx.MockTransport:
    server = _MockServer(store, expect_key)
    return httpx.MockTransport(server.handle)


def _client_with_transport(transport: httpx.BaseTransport, *, api_key: str = "k") -> DataBankClient:
    client = httpx.Client(transport=transport)
    return DataBankClient("http://testserver", api_key=api_key, client=client)


def test_client_upload_head_download(tmp_path: Path) -> None:
    store = _MemStore()
    client = _client_with_transport(_mock_transport(store, expect_key="k"))

    payload = b"hello" * 1000
    store.put("abcd1234", payload, "text/plain")
    up = client.upload("abcd1234", io.BytesIO(payload), content_type="text/plain")
    assert up["size"] == len(payload)

    head = client.head("abcd1234")
    assert head["size"] == len(payload)
    assert head["etag"] == sha256(payload).hexdigest()

    dest = tmp_path / "file.bin"
    client.download_to_path("abcd1234", dest)
    assert dest.read_bytes() == payload


def test_client_resume_and_verify(tmp_path: Path) -> None:
    store = _MemStore()
    data = os.urandom(128 * 1024)
    store.put("deadbeef", data, "application/octet-stream")
    client = _client_with_transport(_mock_transport(store, expect_key="k"))

    dest = tmp_path / "part.bin"
    dest.write_bytes(data[:10_000])
    head = client.download_to_path("deadbeef", dest, resume=True)
    assert head["size"] == len(data)
    assert dest.read_bytes() == data


def test_client_416_and_404_errors(tmp_path: Path) -> None:
    store = _MemStore()
    store.put("aa11bb22", b"x" * 10, "application/octet-stream")
    client = _client_with_transport(_mock_transport(store, expect_key="k"))

    dest = tmp_path / "d.bin"
    dest.write_bytes(b"z" * 100)
    with pytest.raises(RangeNotSatisfiableError):
        client.download_to_path("aa11bb22", dest, resume=True)

    with pytest.raises(NotFoundError):
        client.head("missing")


def test_client_auth_and_conflict_errors() -> None:
    store = _MemStore()
    client = _client_with_transport(_mock_transport(store, expect_key="correct"))
    with pytest.raises(AuthorizationError):
        client.head("anything")

    store.put("conflict", b"x", "text/plain")
    conflict_client = _client_with_transport(
        _mock_transport(store, expect_key="correct"), api_key="correct"
    )
    with pytest.raises(ConflictError):
        conflict_client.head("conflict")


def test_client_retry_and_error_mappings(tmp_path: Path) -> None:
    store = _MemStore()
    store.put("retryme", b"xx", "text/plain")
    store.put("checkrid", b"yy", "text/plain")
    client = _client_with_transport(_mock_transport(store, expect_key="k"))
    head = client.head("retryme")
    assert head["size"] == 2

    store.put("bad400", b"x", "text/plain")
    with pytest.raises(BadRequestError):
        client.head("bad400")

    store.put("bad403", b"x", "text/plain")
    with pytest.raises(ForbiddenError):
        client.head("bad403")

    store.put("bad507", b"x", "text/plain")
    with pytest.raises(InsufficientStorageClientError):
        client.head("bad507")

    store.put("err502", b"x", "text/plain")
    with pytest.raises(DataBankClientError):
        client.head("err502")

    head_with_rid = client.head("checkrid", request_id="RID-1")
    assert head_with_rid["etag"] == sha256(b"yy").hexdigest()


def test_download_stream_error_branch(tmp_path: Path) -> None:
    store = _MemStore()
    store.put("err416", b"abcd", "text/plain")
    client = _client_with_transport(_mock_transport(store, expect_key="k"))
    with pytest.raises(RangeNotSatisfiableError):
        client.download_to_path("err416", tmp_path / "x.bin", resume=False)


def test_download_no_verify_branch(tmp_path: Path) -> None:
    store = _MemStore()
    data = b"m" * 64
    store.put("nv", data, "application/octet-stream")
    client = _client_with_transport(_mock_transport(store, expect_key="k"))
    dest = tmp_path / "nv.bin"
    client.download_to_path("nv", dest, resume=False, verify_etag=False)
    assert dest.read_bytes() == data


def test_client_delete_and_info(tmp_path: Path) -> None:
    store = _MemStore()
    data = b"abc" * 10
    store.put("ff00aa11", data, "text/plain")
    client = _client_with_transport(_mock_transport(store, expect_key="k"))

    info = client.info("ff00aa11")
    assert info["size"] == len(data)
    assert info["sha256"] == sha256(data).hexdigest()

    client.delete("ff00aa11")
    with pytest.raises(NotFoundError):
        client.delete("ff00aa11")

    store.put("11223344", b"xyz", "application/octet-stream")
    dest = tmp_path / "mismatch.bin"
    client.download_to_path("11223344", dest)
    dest.write_bytes(b"zzz")
    with pytest.raises(DataBankClientError):
        client.download_to_path("11223344", dest, resume=True, verify_etag=True)


def test_client_transport_retry_then_fail() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    client = DataBankClient(
        "http://x",
        api_key="k",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retries=1,
        backoff_seconds=0.0,
    )
    with pytest.raises(DataBankClientError):
        client.head("any")


def test_client_upload_unauthorized() -> None:
    store = _MemStore()
    client = _client_with_transport(_mock_transport(store, expect_key="other"))
    with pytest.raises(AuthorizationError):
        client.upload("fid", io.BytesIO(b"data"), content_type="text/plain")


def test_client_resume_already_complete(tmp_path: Path) -> None:
    store = _MemStore()
    data = b"z" * 1024
    store.put("c1", data, "application/octet-stream")
    client = _client_with_transport(_mock_transport(store, expect_key="k"))
    dest = tmp_path / "c1.bin"
    client.download_to_path("c1", dest)
    head2: HeadInfo = client.download_to_path("c1", dest, resume=True, verify_etag=True)
    assert head2["size"] == len(data)


def test_client_already_complete_no_verify(tmp_path: Path) -> None:
    store = _MemStore()
    data = b"q" * 256
    store.put("c2", data, "application/octet-stream")
    client = _client_with_transport(_mock_transport(store, expect_key="k"))
    dest = tmp_path / "c2.bin"
    client.download_to_path("c2", dest)
    head = client.download_to_path("c2", dest, resume=True, verify_etag=False)
    assert head["size"] == len(data)


def test_raise_for_error_return_branch() -> None:
    dummy = httpx.Response(200, text="ok")
    DataBankClient("http://x", api_key="k")._raise_for_error(dummy)


def test_client_upload_invalid_json_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/files":
            return httpx.Response(201, text="not-json")
        return httpx.Response(500, text="unhandled")

    transport = httpx.MockTransport(handler)
    client = DataBankClient("http://x", api_key="k", client=httpx.Client(transport=transport))
    with pytest.raises(DataBankClientError):
        client.upload("fid", io.BytesIO(b"data"), content_type="text/plain")


def test_client_upload_missing_fields_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/files":
            body = {"file_id": "fid"}
            return httpx.Response(201, text=dump_json_str(body))
        return httpx.Response(500, text="unhandled")

    transport = httpx.MockTransport(handler)
    client = DataBankClient("http://x", api_key="k", client=httpx.Client(transport=transport))
    with pytest.raises(DataBankClientError):
        client.upload("fid", io.BytesIO(b"data"), content_type="text/plain")


def test_client_upload_missing_file_id_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/files":
            body = {
                "size": 1,
                "sha256": "a" * 64,
                "content_type": "text/plain",
                "created_at": None,
            }
            return httpx.Response(201, text=dump_json_str(body))
        return httpx.Response(500, text="unhandled")

    transport = httpx.MockTransport(handler)
    client = DataBankClient("http://x", api_key="k", client=httpx.Client(transport=transport))
    with pytest.raises(DataBankClientError):
        client.upload("fid", io.BytesIO(b"data"), content_type="text/plain")


def test_client_upload_non_object_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/files":
            return httpx.Response(201, text="[]")
        return httpx.Response(500, text="unhandled")

    transport = httpx.MockTransport(handler)
    client = DataBankClient("http://x", api_key="k", client=httpx.Client(transport=transport))
    with pytest.raises(DataBankClientError):
        client.upload("fid", io.BytesIO(b"data"), content_type="text/plain")


def test_decode_upload_response_non_dict_path() -> None:
    raw: JSONValue = ["not-a-dict"]
    with pytest.raises(DataBankClientError):
        _decode_upload_response(raw)


def test_client_upload_missing_sha256_or_content_type_or_created_at() -> None:
    def handler_missing_sha(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/files":
            body = {
                "file_id": "fid",
                "size": 1,
                "content_type": "text/plain",
                "created_at": None,
            }
            return httpx.Response(201, text=dump_json_str(body))
        return httpx.Response(500, text="unhandled")

    client1 = DataBankClient(
        "http://x",
        api_key="k",
        client=httpx.Client(transport=httpx.MockTransport(handler_missing_sha)),
    )
    with pytest.raises(DataBankClientError):
        client1.upload("fid", io.BytesIO(b"data"), content_type="text/plain")

    def handler_missing_ct(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/files":
            body = {"file_id": "fid", "size": 1, "sha256": "a" * 64, "created_at": None}
            return httpx.Response(201, text=dump_json_str(body))
        return httpx.Response(500, text="unhandled")

    client2 = DataBankClient(
        "http://x",
        api_key="k",
        client=httpx.Client(transport=httpx.MockTransport(handler_missing_ct)),
    )
    with pytest.raises(DataBankClientError):
        client2.upload("fid", io.BytesIO(b"data"), content_type="text/plain")

    def handler_invalid_created_at(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/files":
            body = {
                "file_id": "fid",
                "size": 1,
                "sha256": "a" * 64,
                "content_type": "text/plain",
                "created_at": 123,
            }
            return httpx.Response(201, text=dump_json_str(body))
        return httpx.Response(500, text="unhandled")

    client3 = DataBankClient(
        "http://x",
        api_key="k",
        client=httpx.Client(transport=httpx.MockTransport(handler_invalid_created_at)),
    )
    with pytest.raises(DataBankClientError):
        client3.upload("fid", io.BytesIO(b"data"), content_type="text/plain")


def test_stream_download() -> None:
    """Test stream_download returns response that can be iterated."""
    content = b"stream test content"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/files/stream-file":
            return httpx.Response(200, content=content)
        return httpx.Response(404, text="not found")

    client = DataBankClient(
        "http://x",
        api_key="k",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    resp = client.stream_download("stream-file", request_id="req-123")
    assert resp.status_code == 200
    chunks = list(resp.iter_bytes())
    assert b"".join(chunks) == content
    resp.close()
