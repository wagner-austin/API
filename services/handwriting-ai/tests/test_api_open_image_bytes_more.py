from __future__ import annotations

from typing import BinaryIO

import pytest
from PIL import Image
from PIL.Image import Image as PILImage
from platform_core.errors import AppError, ErrorCode, HandwritingErrorCode

from handwriting_ai import _test_hooks
from handwriting_ai.api.routes.read import _open_image_bytes


def test_open_image_invalid_bytes_raises_invalid_image() -> None:
    with pytest.raises(AppError) as ei:
        _ = _open_image_bytes(b"not-an-image")
    e = ei.value
    assert e.code is HandwritingErrorCode.invalid_image


def test_open_image_decompression_bomb_maps_to_payload_too_large() -> None:
    def _raise_bomb(fp: BinaryIO) -> PILImage:
        _ = fp  # unused
        raise Image.DecompressionBombError("bomb")

    _test_hooks.pil_image_open = _raise_bomb
    with pytest.raises(AppError) as ei:
        _ = _open_image_bytes(b"header")
    e = ei.value
    assert e.code is ErrorCode.PAYLOAD_TOO_LARGE
