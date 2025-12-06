from __future__ import annotations

import pytest
from PIL import Image
from platform_core.errors import AppError, ErrorCode, HandwritingErrorCode
from starlette.datastructures import FormData

from handwriting_ai.api.routes.read import (
    _ensure_supported_content_type,
    _strict_validate_multipart,
    _validate_image_dimensions,
)
from handwriting_ai.config import Limits


def test_validate_image_dimensions_ok() -> None:
    img = Image.new("L", (16, 16), color=0)
    limits = Limits(max_bytes=1024 * 1024, max_side_px=1024)
    # Should not raise for small dimensions
    _validate_image_dimensions(img, limits)


def test_strict_multipart_missing_file_raises() -> None:
    # No 'file' parts present -> explicit malformed_multipart error path
    form = FormData([])
    with pytest.raises(AppError) as ei:
        _strict_validate_multipart(form)
    err = ei.value
    assert err.code is HandwritingErrorCode.malformed_multipart
    assert "Missing file" in err.message


def test_ensure_supported_content_type_ok_variants() -> None:
    # Accepted types should not raise
    for ctype in ("image/png", "image/jpeg", "image/jpg"):
        _ensure_supported_content_type(ctype)


def test_ensure_supported_content_type_rejects() -> None:
    with pytest.raises(AppError) as ei:
        _ensure_supported_content_type("application/pdf")
    e = ei.value
    assert e.code is ErrorCode.UNSUPPORTED_MEDIA_TYPE
