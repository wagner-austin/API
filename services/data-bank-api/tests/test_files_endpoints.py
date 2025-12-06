from __future__ import annotations

import io
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO, TypedDict

import pytest
from fastapi.testclient import TestClient
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import load_json_str

from data_bank_api.app import create_app
from data_bank_api.config import Settings
from data_bank_api.storage import Storage

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


class InfoResponseBody(TypedDict):
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
    sha256_val = data.get("sha256")
    content_type = data.get("content_type")
    created_at = data.get("created_at")
    if not isinstance(file_id, str) or not isinstance(size, int) or not isinstance(sha256_val, str):
        raise ValueError("invalid upload response")
    if not isinstance(content_type, str):
        raise ValueError("invalid content_type")
    if created_at is not None and not isinstance(created_at, str):
        raise ValueError("invalid created_at")
    return {
        "file_id": file_id,
        "size": size,
        "sha256": sha256_val,
        "content_type": content_type,
        "created_at": created_at,
    }


def _decode_info_response(data: UnknownJson) -> InfoResponseBody:
    if not isinstance(data, dict):
        raise ValueError("expected dict")
    file_id = data.get("file_id")
    size = data.get("size")
    sha256_val = data.get("sha256")
    content_type = data.get("content_type")
    created_at = data.get("created_at")
    if not isinstance(file_id, str) or not isinstance(size, int) or not isinstance(sha256_val, str):
        raise ValueError("invalid info response")
    if not isinstance(content_type, str):
        raise ValueError("invalid content_type")
    if created_at is not None and not isinstance(created_at, str):
        raise ValueError("invalid created_at")
    return {
        "file_id": file_id,
        "size": size,
        "sha256": sha256_val,
        "content_type": content_type,
        "created_at": created_at,
    }


def _client(tmp_path: Path) -> TestClient:
    root = tmp_path / "files"
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
    return TestClient(create_app(s))


def test_upload_head_get_delete_roundtrip(tmp_path: Path) -> None:
    client = _client(tmp_path)
    payload = b"hello world" * 1000
    _ = sha256(payload).hexdigest()

    # upload
    resp = client.post(
        "/files",
        files={"file": ("abcd1234", io.BytesIO(payload), "text/plain")},
    )
    # fastapi may return 200 or 201 depending on model; accept either
    assert resp.status_code in (200, 201)
    body = _decode_upload_response(_decode_json(resp.text))
    fid = body["file_id"]

    # head
    r2 = client.head(f"/files/{fid}")
    assert r2.status_code == 200
    assert r2.headers["Content-Length"] == str(len(payload))

    # get full
    r3 = client.get(f"/files/{fid}")
    assert r3.status_code == 200
    assert r3.content == payload

    # get range
    r4 = client.get(f"/files/{fid}", headers={"Range": "bytes=5-15"})
    assert r4.status_code == 206
    assert r4.content == payload[5:16]
    # headers include ETag and Content-Type on partial content
    headers_map: dict[str, str] = {str(k).lower(): str(v) for (k, v) in r4.headers.items()}
    assert "etag" in headers_map
    ctype = headers_map.get("content-type", "")
    assert ctype.startswith("text/plain")

    # info
    r5 = client.get(f"/files/{fid}/info")
    assert r5.status_code == 200
    b5 = _decode_info_response(_decode_json(r5.text))
    size_val = b5["size"]
    assert type(size_val) is int
    assert size_val == len(payload)
    # info includes created_at
    assert "created_at" in b5

    # delete
    r6 = client.delete(f"/files/{fid}")
    assert r6.status_code == 204
    # idempotent
    r7 = client.delete(f"/files/{fid}")
    assert r7.status_code == 204


def test_upload_400_bad_request_on_storage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Monkeypatch Storage.save_stream to raise AppError with INVALID_INPUT
    client = _client(tmp_path)

    def _boom(self: Storage, stream: BinaryIO, content_type: str) -> None:
        raise AppError(ErrorCode.INVALID_INPUT, "boom", 400)

    monkeypatch.setattr(Storage, "save_stream", _boom)
    resp = client.post(
        "/files",
        files={"file": ("x.txt", io.BytesIO(b"x"), "text/plain")},
    )
    assert resp.status_code == 400
    body = _decode_error_body(_decode_json(resp.text))
    assert body["code"] == "INVALID_INPUT"
