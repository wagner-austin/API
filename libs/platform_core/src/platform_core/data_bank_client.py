from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import BinaryIO, Final, TypedDict

import httpx

from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.http_utils import add_correlation_header
from platform_core.json_utils import JSONValue, load_json_str


class FileInfoDict(TypedDict):
    file_id: str
    size: int
    sha256: str
    content_type: str


class HeadInfo(TypedDict):
    size: int
    etag: str
    content_type: str


class DataBankClientError(Exception):
    """Base client error for data-bank operations."""


class AuthorizationError(DataBankClientError):
    pass


class ForbiddenError(DataBankClientError):
    pass


class NotFoundError(DataBankClientError):
    pass


class ConflictError(DataBankClientError):
    pass


class RangeNotSatisfiableError(DataBankClientError):
    pass


class BadRequestError(DataBankClientError):
    pass


class InsufficientStorageClientError(DataBankClientError):
    pass


def _decode_upload_response(raw: JSONValue) -> FileUploadResponse:
    if not isinstance(raw, dict):
        raise DataBankClientError("upload response is not an object")
    file_id = raw.get("file_id")
    size = raw.get("size")
    sha256_val = raw.get("sha256")
    content_type = raw.get("content_type")
    created_at = raw.get("created_at")
    if not isinstance(file_id, str) or file_id.strip() == "":
        raise DataBankClientError("upload response missing file_id")
    if not isinstance(size, int):
        raise DataBankClientError("upload response missing size")
    if not isinstance(sha256_val, str) or sha256_val.strip() == "":
        raise DataBankClientError("upload response missing sha256")
    if not isinstance(content_type, str) or content_type.strip() == "":
        raise DataBankClientError("upload response missing content_type")
    if created_at is not None and not isinstance(created_at, str):
        raise DataBankClientError("upload response has invalid created_at")
    return {
        "file_id": file_id,
        "size": size,
        "sha256": sha256_val,
        "content_type": content_type,
        "created_at": created_at,
    }


class DataBankClient:
    """Strict HTTP client for data-bank-api."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_seconds: float = 60.0,
        retries: int = 3,
        backoff_seconds: float = 0.5,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url: Final[str] = base_url.rstrip("/")
        self._api_key: Final[str] = api_key
        self._timeout: Final[float] = timeout_seconds
        self._retries: Final[int] = max(0, retries)
        self._backoff: Final[float] = max(0.0, backoff_seconds)
        self._client: Final[httpx.Client] = client or httpx.Client(timeout=self._timeout)

    def _headers(self, request_id: str | None) -> dict[str, str]:
        base = {"X-API-Key": self._api_key}
        if request_id is None or request_id.strip() == "":
            return base
        return add_correlation_header(base, request_id)

    def _request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        files: dict[str, tuple[str, BinaryIO, str]] | None = None,
        idempotent: bool = True,
    ) -> httpx.Response:
        url = f"{self._base_url}{path}"
        attempt = 0
        while True:
            try:
                resp = self._client.request(
                    method,
                    url,
                    headers=headers,
                    files=files,
                    timeout=self._timeout,
                )
                if idempotent and 500 <= resp.status_code < 600 and attempt < self._retries:
                    attempt += 1
                    time.sleep(self._backoff * attempt)
                    continue
                return resp
            except httpx.TransportError as exc:
                if not idempotent or attempt >= self._retries:
                    raise DataBankClientError(f"transport error: {exc}") from exc
                attempt += 1
                time.sleep(self._backoff * attempt)

    def _raise_for_error(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        msg = resp.text
        if resp.status_code == 401:
            raise AuthorizationError(msg)
        if resp.status_code == 403:
            raise ForbiddenError(msg)
        if resp.status_code == 404:
            raise NotFoundError(msg)
        if resp.status_code == 409:
            raise ConflictError(msg)
        if resp.status_code == 416:
            raise RangeNotSatisfiableError(msg)
        if resp.status_code == 400:
            raise BadRequestError(msg)
        if resp.status_code == 507:
            raise InsufficientStorageClientError(msg)
        raise DataBankClientError(f"HTTP {resp.status_code}: {msg}")

    def head(self, file_id: str, *, request_id: str | None = None) -> HeadInfo:
        resp = self._request("HEAD", f"/files/{file_id}", headers=self._headers(request_id))
        if resp.status_code >= 400:
            self._raise_for_error(resp)
        headers_map: dict[str, str] = {k.lower(): v for (k, v) in resp.headers.items()}
        size = int(headers_map["content-length"]) if "content-length" in headers_map else 0
        etag = headers_map.get("etag", "")
        ctype = headers_map.get("content-type", "application/octet-stream")
        return {"size": size, "etag": etag, "content_type": ctype}

    def info(self, file_id: str, *, request_id: str | None = None) -> FileInfoDict:
        head = self.head(file_id, request_id=request_id)
        return {
            "file_id": file_id,
            "size": head["size"],
            "sha256": head["etag"],
            "content_type": head["content_type"],
        }

    def delete(self, file_id: str, *, request_id: str | None = None) -> None:
        resp = self._request("DELETE", f"/files/{file_id}", headers=self._headers(request_id))
        if resp.status_code >= 400:
            self._raise_for_error(resp)

    def upload(
        self,
        file_id: str,
        stream: BinaryIO,
        *,
        content_type: str = "application/octet-stream",
        request_id: str | None = None,
    ) -> FileUploadResponse:
        files = {"file": (file_id, stream, content_type)}
        resp = self._request(
            "POST",
            "/files",
            headers=self._headers(request_id),
            files=files,
            idempotent=False,
        )
        if resp.status_code >= 400:
            self._raise_for_error(resp)
        text = resp.text
        stripped = text.lstrip()
        if not (stripped.startswith("{") and text.rstrip().endswith("}")):
            raise DataBankClientError("invalid JSON response from data-bank-api")
        parsed = load_json_str(text)
        return _decode_upload_response(parsed)

    def download_to_path(
        self,
        file_id: str,
        dest: Path,
        *,
        resume: bool = True,
        request_id: str | None = None,
        verify_etag: bool = True,
        chunk_size: int = 1024 * 1024,
    ) -> HeadInfo:
        head = self.head(file_id, request_id=request_id)
        start = dest.stat().st_size if resume and dest.exists() else 0
        if start > head["size"]:
            raise RangeNotSatisfiableError("local file larger than remote")
        if start >= head["size"]:
            if verify_etag:
                self._verify_file_etag(dest, head["etag"])
            return head

        headers = self._headers(request_id)
        if start > 0:
            headers["Range"] = f"bytes={start}-"
        url = f"{self._base_url}/files/{file_id}"
        with self._client.stream("GET", url, headers=headers, timeout=self._timeout) as resp:
            if resp.status_code >= 400:
                self._raise_for_error(resp)
            if start > 0:
                with dest.open("r+b") as f:
                    f.seek(start)
                    for chunk in resp.iter_bytes(chunk_size=chunk_size):
                        f.write(chunk)
            else:
                with dest.open("wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=chunk_size):
                        f.write(chunk)
        if verify_etag:
            self._verify_file_etag(dest, head["etag"])
        return head

    @staticmethod
    def _verify_file_etag(path: Path, expected_etag: str) -> None:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for data in iter(lambda: f.read(1024 * 1024), b""):
                h.update(data)
        if h.hexdigest() != expected_etag:
            raise DataBankClientError("downloaded file hash does not match ETag")

    def stream_download(
        self,
        file_id: str,
        *,
        request_id: str | None = None,
        chunk_size: int = 8192,
    ) -> httpx.Response:
        """Start a streaming download of a file.

        Returns an httpx.Response that can be iterated to stream bytes.
        Caller is responsible for checking status_code and using iter_bytes().

        Example:
            resp = client.stream_download("file-123", request_id="req-456")
            if resp.status_code == 200:
                for chunk in resp.iter_bytes():
                    yield chunk
            resp.close()
        """
        url = f"{self._base_url}/files/{file_id}"
        headers = self._headers(request_id)
        # Use send() with stream=True to get a Response that stays open
        request = self._client.build_request("GET", url, headers=headers)
        return self._client.send(request, stream=True)


__all__ = [
    "AuthorizationError",
    "BadRequestError",
    "ConflictError",
    "DataBankClient",
    "DataBankClientError",
    "FileInfoDict",
    "ForbiddenError",
    "HeadInfo",
    "InsufficientStorageClientError",
    "NotFoundError",
    "RangeNotSatisfiableError",
    "_decode_upload_response",
]
