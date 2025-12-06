from __future__ import annotations

from collections.abc import Generator

from fastapi.responses import StreamingResponse
from platform_core.data_bank_client import (
    DataBankClient,
    DataBankClientError,
    NotFoundError,
)
from platform_core.errors import AppError, ErrorCode

from .config import Settings


def _require_data_bank_config(settings: Settings) -> tuple[str, str]:
    base_url = settings["data_bank_api_url"].rstrip("/")
    api_key = settings["data_bank_api_key"].strip()
    if base_url == "" or api_key == "":
        raise AppError(
            code=ErrorCode.CONFIG_ERROR,
            message="data-bank configuration missing",
            http_status=500,
        )
    return base_url, api_key


def stream_data_bank_file(job_id: str, file_id: str, settings: Settings) -> StreamingResponse:
    base_url, api_key = _require_data_bank_config(settings)
    client = DataBankClient(base_url, api_key, timeout_seconds=120.0)

    try:
        head = client.head(file_id, request_id=job_id)
    except NotFoundError:
        raise AppError(
            code=ErrorCode.JOB_FAILED,
            message="Job result expired",
            http_status=410,
        ) from None
    except DataBankClientError:
        raise AppError(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            message="data-bank head request failed",
            http_status=502,
        ) from None

    content_type = head["content_type"]
    disposition = f'attachment; filename="result_{job_id}.txt"'

    resp = client.stream_download(file_id, request_id=job_id)
    if resp.status_code == 404:
        resp.close()
        raise AppError(
            code=ErrorCode.JOB_FAILED,
            message="Job result expired",
            http_status=410,
        )
    if resp.status_code >= 400:
        resp.close()
        raise AppError(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            message="data-bank download failed",
            http_status=502,
        )

    def _iter_body() -> Generator[bytes, None, None]:
        try:
            yield from resp.iter_bytes()
        finally:
            resp.close()

    return StreamingResponse(
        _iter_body(),
        media_type=content_type,
        headers={"Content-Disposition": disposition},
    )


__all__ = ["stream_data_bank_file"]
