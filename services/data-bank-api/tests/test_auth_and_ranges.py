from __future__ import annotations

import io
from pathlib import Path
from typing import TypedDict

from fastapi.testclient import TestClient
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import load_json_str

from data_bank_api import _test_hooks
from data_bank_api.api.main import create_app
from data_bank_api.config import Settings
from data_bank_api.testing import FakeStorage, make_fake_storage_factory

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


class ErrorBody(TypedDict):
    code: str
    message: str
    request_id: str | None


class UploadResponseBody(TypedDict):
    file_id: str
    size: int
    sha256: str
    content_type: str
    created_at: str | None


def _decode_json(text: str) -> UnknownJson:
    return load_json_str(text)


def _decode_error_body(data: UnknownJson) -> ErrorBody:
    if not isinstance(data, dict):
        raise ValueError("expected dict")
    code = data.get("code")
    message = data.get("message")
    request_id = data.get("request_id")
    if not isinstance(code, str) or not isinstance(message, str):
        raise ValueError("invalid error body")
    if request_id is not None and not isinstance(request_id, str):
        raise ValueError("invalid request_id")
    return {"code": code, "message": message, "request_id": request_id}


def _decode_upload_response(data: UnknownJson) -> UploadResponseBody:
    if not isinstance(data, dict):
        raise ValueError("expected dict")
    file_id = data.get("file_id")
    size = data.get("size")
    sha256 = data.get("sha256")
    content_type = data.get("content_type")
    created_at = data.get("created_at")
    if not isinstance(file_id, str) or not isinstance(size, int) or not isinstance(sha256, str):
        raise ValueError("invalid upload response")
    if not isinstance(content_type, str):
        raise ValueError("invalid content_type")
    if created_at is not None and not isinstance(created_at, str):
        raise ValueError("invalid created_at")
    return {
        "file_id": file_id,
        "size": size,
        "sha256": sha256,
        "content_type": content_type,
        "created_at": created_at,
    }


def _client(tmp_path: Path, settings: Settings | None = None) -> TestClient:
    root = tmp_path / "files"
    s: Settings = settings or {
        "redis_url": "redis://ignored",
        "data_root": str(root),
        "min_free_gb": 0,
        "delete_strict_404": False,
        "max_file_bytes": 0,
        "api_upload_keys": frozenset(),
        "api_read_keys": frozenset(),
        "api_delete_keys": frozenset(),
    }
    return TestClient(create_app(s))


def test_auth_enforced_upload_401_403_200(tmp_path: Path) -> None:
    s: Settings = {
        "redis_url": "redis://ignored",
        "data_root": str(tmp_path / "files"),
        "min_free_gb": 0,
        "delete_strict_404": False,
        "max_file_bytes": 0,
        "api_upload_keys": frozenset({"k1"}),
        "api_read_keys": frozenset(),
        "api_delete_keys": frozenset(),
    }
    client = _client(tmp_path, s)

    # Missing key -> 401
    r1 = client.post(
        "/files",
        files={"file": ("abcd1234", io.BytesIO(b"hi"), "text/plain")},
    )
    assert r1.status_code == 401
    b1 = _decode_error_body(_decode_json(r1.text))
    assert b1["code"] == "UNAUTHORIZED"

    # Wrong key -> 403
    r2 = client.post(
        "/files",
        files={"file": ("abcd1234", io.BytesIO(b"hi"), "text/plain")},
        headers={"X-API-Key": "wrong"},
    )
    assert r2.status_code == 403

    # Correct key -> 201
    r3 = client.post(
        "/files",
        files={"file": ("abcd1234", io.BytesIO(b"hi"), "text/plain")},
        headers={"X-API-Key": "k1"},
    )
    assert r3.status_code == 201
    # also verify HEAD works with read key when configured
    s2: Settings = {
        "redis_url": "redis://ignored",
        "data_root": str(tmp_path / "files2"),
        "min_free_gb": 0,
        "delete_strict_404": False,
        "max_file_bytes": 0,
        "api_upload_keys": frozenset(),
        "api_read_keys": frozenset({"r1"}),
        "api_delete_keys": frozenset(),
    }
    client2 = _client(tmp_path, s2)
    # HEAD without key -> 401
    assert client2.head("/files/deadbeef").status_code == 401
    # with wrong key -> 403
    assert client2.head("/files/deadbeef", headers={"X-API-Key": "bad"}).status_code == 403
    # with correct key -> 404 (resource missing), but auth passed
    assert client2.head("/files/deadbeef", headers={"X-API-Key": "r1"}).status_code == 404


def test_range_errors_and_headers(tmp_path: Path) -> None:
    client = _client(tmp_path)
    payload = b"hello world" * 3
    r0 = client.post(
        "/files",
        files={"file": ("anyname.txt", io.BytesIO(payload), "application/octet-stream")},
    )
    assert r0.status_code in (200, 201)
    body0 = _decode_upload_response(_decode_json(r0.text))
    fid = body0["file_id"]
    assert fid != ""

    # invalid prefix
    r1 = client.get(f"/files/{fid}", headers={"Range": "bad=0-10"})
    assert r1.status_code == 416
    j1 = _decode_error_body(_decode_json(r1.text))
    assert j1["code"] == "RANGE_NOT_SATISFIABLE"

    # multiple ranges
    r2 = client.get(f"/files/{fid}", headers={"Range": "bytes=0-1,2-3"})
    assert r2.status_code == 416

    # non-numeric
    r3 = client.get(f"/files/{fid}", headers={"Range": "bytes=abc-"})
    assert r3.status_code == 416

    # unsatisfiable
    r4 = client.get(f"/files/{fid}", headers={"Range": "bytes=999999-"})
    assert r4.status_code == 416
    assert r4.headers["Content-Range"].startswith("bytes */")

    # ETag on HEAD and GET
    h = client.head(f"/files/{fid}")
    assert h.status_code == 200
    etag: str = h.headers["ETag"]
    g = client.get(f"/files/{fid}")
    assert g.status_code == 200
    assert g.headers["ETag"] == etag


def test_upload_507_from_guard(tmp_path: Path) -> None:
    # Force guard to raise at upload path
    root = tmp_path / "files"
    fake_storage = FakeStorage(root, 0, max_file_bytes=0)

    def _boom() -> None:
        raise AppError(ErrorCode.INSUFFICIENT_STORAGE, "x", 507)

    fake_storage.ensure_free_space_override = _boom
    _test_hooks.storage_factory = make_fake_storage_factory(fake_storage)

    s: Settings = {
        "redis_url": "redis://ignored",
        "data_root": str(root),
        "min_free_gb": 0,
        "delete_strict_404": False,
        "max_file_bytes": 0,
        "api_upload_keys": frozenset(),
        "api_read_keys": frozenset(),
        "api_delete_keys": frozenset(),
    }
    client = TestClient(create_app(s))
    r = client.post(
        "/files",
        files={"file": ("abcd1234", io.BytesIO(b"data"), "text/plain")},
    )
    assert r.status_code == 507


def test_upload_413_payload_too_large(tmp_path: Path) -> None:
    # Configure max size to 1 byte and upload 2 bytes to trigger 413
    s: Settings = {
        "redis_url": "redis://ignored",
        "data_root": str(tmp_path / "files"),
        "min_free_gb": 0,
        "delete_strict_404": False,
        "max_file_bytes": 1,
        "api_upload_keys": frozenset(),
        "api_read_keys": frozenset(),
        "api_delete_keys": frozenset(),
    }
    client = _client(tmp_path, s)
    resp = client.post(
        "/files",
        files={"file": ("x.txt", io.BytesIO(b"dd"), "text/plain")},
    )
    assert resp.status_code == 413


def test_delete_strict_404(tmp_path: Path) -> None:
    s: Settings = {
        "redis_url": "redis://ignored",
        "data_root": str(tmp_path / "files"),
        "min_free_gb": 0,
        "delete_strict_404": True,
        "max_file_bytes": 0,
        "api_upload_keys": frozenset(),
        "api_read_keys": frozenset(),
        "api_delete_keys": frozenset(),
    }
    client = _client(tmp_path, s)
    r = client.delete("/files/deadbeef")
    assert r.status_code == 404


def test_download_missing_file_full_and_range(tmp_path: Path) -> None:
    client = _client(tmp_path)
    # full GET missing
    r1 = client.get("/files/deadbeef")
    assert r1.status_code == 404
    # range GET missing
    r2 = client.get("/files/deadbeef", headers={"Range": "bytes=0-10"})
    assert r2.status_code == 404


def test_unsatisfiable_range_with_disappearing_file(tmp_path: Path) -> None:
    root = tmp_path / "files"
    fake_storage = FakeStorage(root, 0, max_file_bytes=0)
    _test_hooks.storage_factory = make_fake_storage_factory(fake_storage)

    s: Settings = {
        "redis_url": "redis://ignored",
        "data_root": str(root),
        "min_free_gb": 0,
        "delete_strict_404": False,
        "max_file_bytes": 0,
        "api_upload_keys": frozenset(),
        "api_read_keys": frozenset(),
        "api_delete_keys": frozenset(),
    }
    client = TestClient(create_app(s))
    fid = "a1b2c3d4"
    # create a small file
    _ = client.post(
        "/files",
        files={"file": (fid, io.BytesIO(b"hello"), "application/octet-stream")},
    )

    # Simulate file disappearing when computing size after unsatisfiable detection
    def _raise_get_size(_file_id: str) -> int:
        raise AppError(ErrorCode.NOT_FOUND, "gone", 404)

    fake_storage.get_size_override = _raise_get_size
    r = client.get(f"/files/{fid}", headers={"Range": "bytes=999999-"})
    assert r.status_code == 404


def test_read_auth_enforced_for_head_get_info_delete(tmp_path: Path) -> None:
    s: Settings = {
        "redis_url": "redis://ignored",
        "data_root": str(tmp_path / "files"),
        "min_free_gb": 0,
        "delete_strict_404": False,
        "max_file_bytes": 0,
        "api_upload_keys": frozenset(),
        "api_read_keys": frozenset({"rk"}),
        "api_delete_keys": frozenset({"dk"}),
    }
    client = _client(tmp_path, s)

    assert client.head("/files/deadbeef").status_code == 401
    assert client.get("/files/deadbeef").status_code == 401
    assert client.get("/files/deadbeef/info").status_code == 401
    assert client.delete("/files/deadbeef").status_code == 401


def test_info_404_on_missing(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/files/deadbeef/info").status_code == 404


def test_full_download_metadata_error(tmp_path: Path) -> None:
    root = tmp_path / "files"
    fake_storage = FakeStorage(root, 0, max_file_bytes=0)
    _test_hooks.storage_factory = make_fake_storage_factory(fake_storage)

    s: Settings = {
        "redis_url": "redis://ignored",
        "data_root": str(root),
        "min_free_gb": 0,
        "delete_strict_404": False,
        "max_file_bytes": 0,
        "api_upload_keys": frozenset(),
        "api_read_keys": frozenset(),
        "api_delete_keys": frozenset(),
    }
    client = TestClient(create_app(s))
    fid = "abcd1234"
    _ = client.post(
        "/files",
        files={"file": (fid, io.BytesIO(b"hi"), "application/octet-stream")},
    )

    from data_bank_api.storage import FileMetadata

    def _raise_meta(_file_id: str) -> FileMetadata:
        raise AppError(ErrorCode.INVALID_INPUT, "meta bad", 400)

    fake_storage.head_override = _raise_meta
    r = client.get(f"/files/{fid}")
    assert r.status_code == 400
    body = _decode_error_body(_decode_json(r.text))
    assert body["code"] == "INVALID_INPUT"


def test_range_metadata_error(tmp_path: Path) -> None:
    root = tmp_path / "files"
    fake_storage = FakeStorage(root, 0, max_file_bytes=0)
    _test_hooks.storage_factory = make_fake_storage_factory(fake_storage)

    s: Settings = {
        "redis_url": "redis://ignored",
        "data_root": str(root),
        "min_free_gb": 0,
        "delete_strict_404": False,
        "max_file_bytes": 0,
        "api_upload_keys": frozenset(),
        "api_read_keys": frozenset(),
        "api_delete_keys": frozenset(),
    }
    client = TestClient(create_app(s))
    fid = "abcd5678"
    _ = client.post(
        "/files",
        files={"file": (fid, io.BytesIO(b"hi"), "application/octet-stream")},
    )

    from data_bank_api.storage import FileMetadata

    def _raise_meta(_file_id: str) -> FileMetadata:
        raise AppError(ErrorCode.INVALID_INPUT, "meta bad", 400)

    fake_storage.head_override = _raise_meta
    r = client.get(f"/files/{fid}", headers={"Range": "bytes=0-1"})
    assert r.status_code == 400
    body = _decode_error_body(_decode_json(r.text))
    assert body["code"] == "INVALID_INPUT"


def test_head_metadata_error(tmp_path: Path) -> None:
    root = tmp_path / "files"
    fake_storage = FakeStorage(root, 0, max_file_bytes=0)
    _test_hooks.storage_factory = make_fake_storage_factory(fake_storage)

    s: Settings = {
        "redis_url": "redis://ignored",
        "data_root": str(root),
        "min_free_gb": 0,
        "delete_strict_404": False,
        "max_file_bytes": 0,
        "api_upload_keys": frozenset(),
        "api_read_keys": frozenset(),
        "api_delete_keys": frozenset(),
    }
    client = TestClient(create_app(s))
    fid = "abcdefff"
    _ = client.post(
        "/files",
        files={"file": (fid, io.BytesIO(b"hi"), "application/octet-stream")},
    )

    from data_bank_api.storage import FileMetadata

    def _raise_meta(_file_id: str) -> FileMetadata:
        raise AppError(ErrorCode.INVALID_INPUT, "meta bad", 400)

    fake_storage.head_override = _raise_meta
    r = client.head(f"/files/{fid}")
    assert r.status_code == 400


def test_info_metadata_error(tmp_path: Path) -> None:
    root = tmp_path / "files"
    fake_storage = FakeStorage(root, 0, max_file_bytes=0)
    _test_hooks.storage_factory = make_fake_storage_factory(fake_storage)

    s: Settings = {
        "redis_url": "redis://ignored",
        "data_root": str(root),
        "min_free_gb": 0,
        "delete_strict_404": False,
        "max_file_bytes": 0,
        "api_upload_keys": frozenset(),
        "api_read_keys": frozenset(),
        "api_delete_keys": frozenset(),
    }
    client = TestClient(create_app(s))
    fid = "abcdfff0"
    _ = client.post(
        "/files",
        files={"file": (fid, io.BytesIO(b"hi"), "application/octet-stream")},
    )

    from data_bank_api.storage import FileMetadata

    def _raise_meta(_file_id: str) -> FileMetadata:
        raise AppError(ErrorCode.INVALID_INPUT, "meta bad", 400)

    fake_storage.head_override = _raise_meta
    r = client.get(f"/files/{fid}/info")
    assert r.status_code == 400


def test_range_storage_error_then_get_size_not_found(tmp_path: Path) -> None:
    # Cover app.py range error handling where get_size raises NotFound
    root = tmp_path / "files"
    fake_storage = FakeStorage(root, 0, max_file_bytes=0)
    _test_hooks.storage_factory = make_fake_storage_factory(fake_storage)

    s: Settings = {
        "redis_url": "redis://ignored",
        "data_root": str(root),
        "min_free_gb": 0,
        "delete_strict_404": False,
        "max_file_bytes": 0,
        "api_upload_keys": frozenset(),
        "api_read_keys": frozenset(),
        "api_delete_keys": frozenset(),
    }
    client = TestClient(create_app(s))
    # Upload a file first and get its actual file_id
    resp = client.post(
        "/files",
        files={"file": ("testfile", io.BytesIO(b"hello"), "application/octet-stream")},
    )
    body = _decode_upload_response(_decode_json(resp.text))
    fid = body["file_id"]

    from collections.abc import Generator

    # Patch open_range to raise AppError with RANGE_NOT_SATISFIABLE
    def _raise_range_error(
        file_id: str, start: int, end: int | None
    ) -> tuple[Generator[bytes, None, None], int, int]:
        raise AppError(ErrorCode.RANGE_NOT_SATISFIABLE, "simulated unsatisfiable", 416)

    # Patch get_size to raise AppError NOT_FOUND
    def _raise_not_found(file_id: str) -> int:
        raise AppError(ErrorCode.NOT_FOUND, "file vanished", 404)

    fake_storage.open_range_override = _raise_range_error
    fake_storage.get_size_override = _raise_not_found

    # Make a range request that will trigger the range error path
    r = client.get(f"/files/{fid}", headers={"Range": "bytes=0-10"})
    # Should return 404 because get_size raised NotFound
    assert r.status_code == 404


def test_range_open_range_non_416_error_reraises(tmp_path: Path) -> None:
    # Cover app.py line 125: when open_range raises AppError with status != 416
    root = tmp_path / "files"
    fake_storage = FakeStorage(root, 0, max_file_bytes=0)
    _test_hooks.storage_factory = make_fake_storage_factory(fake_storage)

    s: Settings = {
        "redis_url": "redis://ignored",
        "data_root": str(root),
        "min_free_gb": 0,
        "delete_strict_404": False,
        "max_file_bytes": 0,
        "api_upload_keys": frozenset(),
        "api_read_keys": frozenset(),
        "api_delete_keys": frozenset(),
    }
    client = TestClient(create_app(s))
    # Upload a file first and get its actual file_id
    resp = client.post(
        "/files",
        files={"file": ("testfile", io.BytesIO(b"hello"), "application/octet-stream")},
    )
    body = _decode_upload_response(_decode_json(resp.text))
    fid = body["file_id"]

    from collections.abc import Generator

    # Patch open_range to raise AppError with NOT_FOUND (404), not 416
    def _raise_not_found(
        file_id: str, start: int, end: int | None
    ) -> tuple[Generator[bytes, None, None], int, int]:
        raise AppError(ErrorCode.NOT_FOUND, "file vanished mid-request", 404)

    fake_storage.open_range_override = _raise_not_found

    # Make a range request - should re-raise the non-416 error
    r = client.get(f"/files/{fid}", headers={"Range": "bytes=0-10"})
    # Should return 404 because open_range raised NOT_FOUND
    assert r.status_code == 404
    body2 = _decode_error_body(_decode_json(r.text))
    assert body2["code"] == "NOT_FOUND"
