from __future__ import annotations

from PIL import Image
from PIL.Image import Image as PILImage

from handwriting_ai import _test_hooks
from handwriting_ai.preprocess import _deskew_if_needed


def test_deskew_confidence_gate_skips_when_low() -> None:
    input_img = Image.new("L", (10, 10), 255)

    def _fake(img: PILImage, width: int, height: int) -> tuple[float, float] | None:
        _ = (img, width, height)  # unused
        return (5.0, 0.0)

    _test_hooks.principal_angle_confidence = _fake
    out = _deskew_if_needed(input_img)
    assert out is input_img


def test_deskew_small_angle_skips() -> None:
    input_img = Image.new("L", (10, 10), 255)

    def _fake(img: PILImage, width: int, height: int) -> tuple[float, float] | None:
        _ = (img, width, height)  # unused
        return (0.5, 1.0)

    _test_hooks.principal_angle_confidence = _fake
    out = _deskew_if_needed(input_img)
    assert out is input_img


def test_deskew_negative_angle_clamps_and_bbox_none_returns_img() -> None:
    input_img = Image.new("L", (10, 10), 255)

    def _fake(img: PILImage, width: int, height: int) -> tuple[float, float] | None:
        _ = (img, width, height)  # unused
        return (-20.0, 1.0)

    _test_hooks.principal_angle_confidence = _fake
    out = _deskew_if_needed(input_img)
    assert out is input_img
