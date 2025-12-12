from __future__ import annotations

from PIL import Image
from PIL.Image import Image as PILImage

from handwriting_ai import _test_hooks
from handwriting_ai.preprocess import (
    _estimate_background_is_dark,
    _largest_component_crop,
    _otsu_binarize,
)


def test_estimate_background_median_loop_paths() -> None:
    # Force histogram with uniform counts to exercise multiple loop iterations
    def _hist(img: PILImage) -> list[int]:
        _ = img
        return [1] * 256

    _test_hooks.pil_histogram = _hist
    img = Image.new("L", (4, 4), 127)
    out = _estimate_background_is_dark(img)
    assert type(out) is bool


def test_otsu_binarize_various_branches() -> None:
    # Histogram: some dark + some bright to ensure var update and continue path
    hist = [0] * 256
    hist[10] = 10
    hist[200] = 5

    def _hist(img: PILImage) -> list[int]:
        _ = img
        return hist

    _test_hooks.pil_histogram = _hist
    img = Image.new("L", (4, 4), 0)
    bw = _otsu_binarize(img)
    assert bw.mode == "L"


def test_estimate_background_immediate_break() -> None:
    # Heavy first bin forces break at first iteration
    def _hist(img: PILImage) -> list[int]:
        _ = img
        h = [0] * 256
        h[0] = 20
        h[1] = 1
        return h

    _test_hooks.pil_histogram = _hist
    img = Image.new("L", (4, 4), 0)
    _ = _estimate_background_is_dark(img)


def test_otsu_binarize_no_continue_at_start() -> None:
    # Non-zero first bin means w_b != 0 on first iteration, covering that branch
    def _hist(img: PILImage) -> list[int]:
        _ = img
        h = [0] * 256
        h[0] = 5
        h[200] = 5
        return h

    _test_hooks.pil_histogram = _hist
    img = Image.new("L", (4, 4), 0)
    _ = _otsu_binarize(img)


def test_largest_component_equal_area_false_branch() -> None:
    # Two equal 1-pixel components ensure area>best_area false path executes
    bw = Image.new("L", (4, 4), 0)
    bw.putpixel((0, 0), 255)
    bw.putpixel((3, 3), 255)
    out = _largest_component_crop(bw)
    assert out.size[0] >= 1 and out.size[1] >= 1
