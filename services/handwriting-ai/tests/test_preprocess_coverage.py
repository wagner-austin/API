from __future__ import annotations

import pytest
from PIL import Image
from PIL.Image import Image as PILImage
from platform_core.errors import AppError, HandwritingErrorCode

from handwriting_ai import _test_hooks
from handwriting_ai.preprocess import (
    PreprocessOptions,
    _center_on_square,
    _component_bbox_bytes,
    _deskew_if_needed,
    _largest_component_crop,
    _principal_angle,
    _principal_angle_confidence,
    run_preprocess,
)

# =============================================================================
# Test helper functions
# =============================================================================


def _mk_white(size: tuple[int, int]) -> Image.Image:
    return Image.new("L", size, 255)


def _mk_black(size: tuple[int, int]) -> Image.Image:
    return Image.new("L", size, 0)


# =============================================================================
# Tests
# =============================================================================


def test_run_preprocess_catches_non_app_errors() -> None:
    input_img = _mk_white((8, 8))

    def _boom(img: PILImage) -> PILImage | None:
        _ = img  # unused
        raise TypeError("bad exif")

    _test_hooks.exif_transpose = _boom
    opts: PreprocessOptions = {
        "invert": None,
        "center": True,
        "visualize": False,
        "visualize_max_kb": 1,
    }
    with pytest.raises(AppError) as ei:
        _ = run_preprocess(input_img, opts)
    e = ei.value
    assert e.code is HandwritingErrorCode.preprocessing_failed and e.http_status == 400


def test_load_to_grayscale_exif_none() -> None:
    input_img = _mk_white((8, 8))

    def _none(img: PILImage) -> PILImage | None:
        _ = img  # unused
        return None

    _test_hooks.exif_transpose = _none
    opts: PreprocessOptions = {
        "invert": None,
        "center": True,
        "visualize": False,
        "visualize_max_kb": 1,
    }
    with pytest.raises(AppError) as ei:
        _ = run_preprocess(input_img, opts)
    assert ei.value.code is HandwritingErrorCode.invalid_image


def test_largest_component_buffer_size_mismatch_raises() -> None:
    rgb = Image.new("RGB", (3, 3), (0, 0, 0))
    with pytest.raises(AppError) as ei:
        _ = _largest_component_crop(rgb.convert("RGB"))
    e = ei.value
    assert e.code is HandwritingErrorCode.preprocessing_failed and "buffer size" in e.message


def test_component_bbox_updates_minx_miny() -> None:
    width, height = 3, 3
    # Build a small component including (0,0), (1,0), (0,1); start at (1,1)
    vals: list[int] = [0] * (width * height)

    def idx(x: int, y: int) -> int:
        return y * width + x

    for x, y in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        vals[idx(x, y)] = 255
    buf = bytes(vals)
    visited = [False] * (width * height)
    area, bbox = _component_bbox_bytes(buf, visited, width, height, 1, 1)
    x0, y0, x1, y1 = bbox
    assert area >= 4 and x0 == 0 and y0 == 0 and x1 >= 1 and y1 >= 1


def test_deskew_negative_angle_clamp_and_bbox_none() -> None:
    input_img = _mk_white((10, 10))

    def _angle_conf(img: PILImage, width: int, height: int) -> tuple[float, float] | None:
        _ = (img, width, height)  # unused
        return (-20.0, 0.5)

    _test_hooks.principal_angle_confidence = _angle_conf
    out = _deskew_if_needed(input_img)
    # No content so bbox is None; function returns original image
    assert out is input_img


def test_principal_angle_none_when_zero_variance() -> None:
    """Test _principal_angle returns None with low variance input.

    Uses a real image with a single pixel set - produces zero variance
    which triggers the early return path.
    """
    img = _mk_white((4, 4))
    img.putpixel((2, 2), 0)
    assert _principal_angle(img, 4, 4) is None
    assert _principal_angle_confidence(img, 4, 4) is None


def test_center_on_square_no_pixels_returns_input() -> None:
    """Test _center_on_square with all-black image returns input.

    When there are no white pixels to center, the function returns
    the original image unchanged.
    """
    img = _mk_black((6, 6))
    out = _center_on_square(img)
    assert out is img
