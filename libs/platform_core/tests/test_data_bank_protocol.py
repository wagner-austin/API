from __future__ import annotations

from platform_core.data_bank_protocol import FileUploadResponse


def test_file_upload_response_shape() -> None:
    body: FileUploadResponse = {
        "file_id": "fid-123",
        "size": 1,
        "sha256": "a" * 64,
        "content_type": "text/plain",
        "created_at": "2024-01-01T00:00:00Z",
    }
    assert body["file_id"] == "fid-123"
    assert body["file_id"] == "fid-123"
