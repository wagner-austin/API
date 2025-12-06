from __future__ import annotations

from typing import TypedDict


class FileUploadResponse(TypedDict):
    file_id: str
    size: int
    sha256: str
    content_type: str
    created_at: str | None


__all__ = ["FileUploadResponse"]
