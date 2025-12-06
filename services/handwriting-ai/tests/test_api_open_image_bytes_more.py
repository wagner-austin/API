from __future__ import annotations

import pytest
from PIL import Image
from platform_core.errors import AppError, ErrorCode, HandwritingErrorCode

from handwriting_ai.api.routes.read import _open_image_bytes

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


def test_open_image_invalid_bytes_raises_invalid_image() -> None:
    with pytest.raises(AppError) as ei:
        _ = _open_image_bytes(b"not-an-image")
    e = ei.value
    assert e.code is HandwritingErrorCode.invalid_image


def test_open_image_decompression_bomb_maps_to_payload_too_large(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io

    def _raise_bomb(fp: io.BytesIO) -> Image.Image:
        raise Image.DecompressionBombError("bomb")

    monkeypatch.setattr(Image, "open", _raise_bomb, raising=True)
    with pytest.raises(AppError) as ei:
        _ = _open_image_bytes(b"header")
    e = ei.value
    assert e.code is ErrorCode.PAYLOAD_TOO_LARGE
