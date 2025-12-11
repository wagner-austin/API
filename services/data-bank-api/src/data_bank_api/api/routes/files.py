"""File storage routes for data-bank-api."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, File, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.errors import AppError, ErrorCode
from platform_core.request_context import request_id_var

from ..._test_hooks import StorageProtocol
from ...config import Settings

Permission = Literal["upload", "read", "delete"]


def _ensure_auth(cfg: Settings, perm: Permission, req: Request) -> None:
    """Validate API key for the given permission. Raises AppError on failure."""
    allowed = (
        cfg["api_upload_keys"]
        if perm == "upload"
        else cfg["api_read_keys"]
        if perm == "read"
        else cfg["api_delete_keys"]
    )
    # If no keys configured for this permission, auth is disabled.
    if len(allowed) == 0:
        return
    key = req.headers.get("X-API-Key")
    if key is None or key.strip() == "":
        raise AppError(ErrorCode.UNAUTHORIZED, "missing API key", 401)
    if key not in allowed:
        raise AppError(ErrorCode.FORBIDDEN, "invalid API key for permission", 403)


def _download_full(storage: StorageProtocol, file_id: str) -> StreamingResponse:
    """Download full file. Raises AppError on failure."""
    meta = storage.head(file_id)
    it, start, last = storage.open_range(file_id, 0, None)
    total = last - start + 1
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(total),
        "ETag": meta["sha256"],
    }
    return StreamingResponse(
        it,
        status_code=200,
        headers=headers,
        media_type=meta["content_type"],
    )


def _unsatisfiable_range_response(storage: StorageProtocol, file_id: str) -> JSONResponse:
    """Build 416 response with Content-Range header. Raises AppError if file not found."""
    total_size = storage.get_size(file_id)
    headers = {"Content-Range": f"bytes */{total_size}"}
    rid = request_id_var.get()
    body = {
        "code": "RANGE_NOT_SATISFIABLE",
        "message": "unsatisfiable range",
        "request_id": rid,
    }
    return JSONResponse(status_code=416, content=body, headers=headers)


def _download_range(
    storage: StorageProtocol, file_id: str, range_header: str
) -> StreamingResponse | JSONResponse:
    """Download file range. Raises AppError on failure."""
    if not range_header.startswith("bytes="):
        raise AppError(ErrorCode.RANGE_NOT_SATISFIABLE, "invalid range", 416)
    spec = range_header[len("bytes=") :]
    if "," in spec:
        raise AppError(ErrorCode.RANGE_NOT_SATISFIABLE, "multiple ranges not supported", 416)
    start_s, _, end_s = spec.partition("-")
    try:
        start = int(start_s) if start_s != "" else 0
        end = int(end_s) if end_s != "" else None
    except ValueError:
        raise AppError(ErrorCode.RANGE_NOT_SATISFIABLE, "invalid range", 416) from None
    # Check for range errors and return 416 with Content-Range header
    meta2 = storage.head(file_id)
    try:
        it, start_pos, last_pos = storage.open_range(file_id, start, end)
    except AppError as exc:
        if exc.http_status == 416:
            return _unsatisfiable_range_response(storage, file_id)
        raise
    total = last_pos - start_pos + 1
    total_size2 = storage.get_size(file_id)
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(total),
        "Content-Range": f"bytes {start_pos}-{last_pos}/{total_size2}",
        "ETag": meta2["sha256"],
        "Content-Type": meta2["content_type"],
    }
    return StreamingResponse(
        it,
        status_code=206,
        headers=headers,
        media_type=meta2["content_type"],
    )


def build_router(storage: StorageProtocol, cfg: Settings) -> APIRouter:
    """Build file storage router."""
    router = APIRouter()

    def _upload(
        file: Annotated[UploadFile, File(...)],
        request: Request,
    ) -> FileUploadResponse:
        _ensure_auth(cfg, "upload", request)
        ct = file.content_type or "application/octet-stream"
        meta = storage.save_stream(file.file, ct)
        return {
            "file_id": meta["file_id"],
            "size": meta["size_bytes"],
            "sha256": meta["sha256"],
            "content_type": meta["content_type"],
            "created_at": meta["created_at"],
        }

    def _head(file_id: str, request: Request) -> Response:
        _ensure_auth(cfg, "read", request)
        meta = storage.head(file_id)
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(meta["size_bytes"]),
            "ETag": meta["sha256"],
            "Content-Type": meta["content_type"],
        }
        return Response(status_code=200, headers=headers)

    def _download(file_id: str, request: Request) -> StreamingResponse | JSONResponse:
        _ensure_auth(cfg, "read", request)
        range_header = request.headers.get("Range")
        if range_header is None:
            return _download_full(storage, file_id)
        return _download_range(storage, file_id, range_header)

    def _info(file_id: str, request: Request) -> FileUploadResponse:
        _ensure_auth(cfg, "read", request)
        meta = storage.head(file_id)
        return {
            "file_id": meta["file_id"],
            "size": meta["size_bytes"],
            "sha256": meta["sha256"],
            "content_type": meta["content_type"],
            "created_at": meta["created_at"],
        }

    def _delete(file_id: str, request: Request) -> Response:
        _ensure_auth(cfg, "delete", request)
        deleted = storage.delete(file_id)
        if not deleted and cfg["delete_strict_404"]:
            raise AppError(ErrorCode.NOT_FOUND, "file not found", 404)
        return Response(status_code=204)

    router.add_api_route(
        "/files",
        _upload,
        methods=["POST"],
        status_code=status.HTTP_201_CREATED,
        response_model=None,
    )
    router.add_api_route(
        "/files/{file_id}",
        _head,
        methods=["HEAD"],
        response_model=None,
    )
    router.add_api_route(
        "/files/{file_id}",
        _download,
        methods=["GET"],
        response_model=None,
    )
    router.add_api_route(
        "/files/{file_id}/info",
        _info,
        methods=["GET"],
        response_model=None,
    )
    router.add_api_route(
        "/files/{file_id}",
        _delete,
        methods=["DELETE"],
        response_model=None,
    )

    return router


__all__ = ["build_router"]
