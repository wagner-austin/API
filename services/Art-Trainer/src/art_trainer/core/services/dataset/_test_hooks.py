"""Test hooks for dataset service.

This module provides dependency injection hooks for the dataset download
and upload services. Production code uses real HTTP client, tests replace
with fakes.
"""

from __future__ import annotations

from typing import Protocol

from typing_extensions import TypedDict


class UploadResult(TypedDict, total=True):
    """Result of uploading a file to data-bank.

    Attributes:
        file_id: Unique identifier for the uploaded file in data-bank.
        filename: Original filename that was uploaded.
    """

    file_id: str
    filename: str


class HttpGetProto(Protocol):
    """Protocol for HTTP GET with streaming response."""

    def __call__(self, url: str, headers: dict[str, str]) -> bytes:
        """Perform HTTP GET and return response bytes.

        Args:
            url: URL to fetch.
            headers: Request headers.

        Returns:
            Response body bytes.

        Raises:
            DataBankDownloadError: If download fails.
        """
        ...


class HttpUploadProto(Protocol):
    """Protocol for HTTP multipart file upload."""

    def __call__(
        self,
        url: str,
        headers: dict[str, str],
        filename: str,
        content: bytes,
    ) -> UploadResult:
        """Upload a file via HTTP multipart POST.

        Args:
            url: Upload endpoint URL.
            headers: Request headers.
            filename: Name of the file being uploaded.
            content: File content bytes.

        Returns:
            UploadResult with file_id and filename.

        Raises:
            DataBankUploadError: If upload fails.
        """
        ...


class DataBankDownloadError(Exception):
    """Error downloading from data-bank."""

    pass


class DataBankUploadError(Exception):
    """Error uploading to data-bank."""

    pass


def _default_http_get(url: str, headers: dict[str, str]) -> bytes:
    """Production HTTP GET implementation using httpx.

    Args:
        url: URL to fetch.
        headers: Request headers.

    Returns:
        Response body bytes.

    Raises:
        DataBankDownloadError: If download fails.
    """
    import httpx

    with httpx.Client(timeout=300.0) as client:
        response = client.get(url, headers=headers)
        if response.status_code != 200:
            raise DataBankDownloadError(
                f"Failed to download from data-bank: {response.status_code}"
            )
        return response.content


def _default_http_upload(
    url: str,
    headers: dict[str, str],
    filename: str,
    content: bytes,
) -> UploadResult:
    """Production HTTP upload implementation using httpx.

    Args:
        url: Upload endpoint URL.
        headers: Request headers.
        filename: Name of the file being uploaded.
        content: File content bytes.

    Returns:
        UploadResult with file_id and filename.

    Raises:
        DataBankUploadError: If upload fails.
    """
    import httpx
    from platform_core.json_utils import load_json_str

    files = {"file": (filename, content, "application/octet-stream")}

    with httpx.Client(timeout=300.0) as client:
        response = client.post(url, headers=headers, files=files)
        if response.status_code not in (200, 201):
            raise DataBankUploadError(f"Failed to upload to data-bank: {response.status_code}")
        response_json = load_json_str(response.text)
        if not isinstance(response_json, dict):
            raise DataBankUploadError("Invalid response from data-bank: not a dict")
        file_id_raw = response_json.get("file_id")
        if not isinstance(file_id_raw, str):
            raise DataBankUploadError("Invalid response from data-bank: missing file_id")

        result: UploadResult = {
            "file_id": file_id_raw,
            "filename": filename,
        }
        return result


# Hooks - production defaults, tests can replace
http_get: HttpGetProto = _default_http_get
http_upload: HttpUploadProto = _default_http_upload


def reset_hooks() -> None:
    """Reset hooks to defaults for test isolation."""
    global http_get, http_upload
    http_get = _default_http_get
    http_upload = _default_http_upload


__all__ = [
    "DataBankDownloadError",
    "DataBankUploadError",
    "HttpGetProto",
    "HttpUploadProto",
    "UploadResult",
    "http_get",
    "http_upload",
    "reset_hooks",
]
