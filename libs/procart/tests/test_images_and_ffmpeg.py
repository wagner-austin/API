from __future__ import annotations

import os
import tempfile

import pytest

from procart.ffmpeg_runner import build_ffmpeg_args
from procart.images_io import resize_image, write_frame_png
from procart.math_backend import BACKEND
from procart.types import Resolution


def test_build_ffmpeg_args_validation() -> None:
    with pytest.raises(ValueError):
        build_ffmpeg_args("frames", 0, "out.mp4")
    with pytest.raises(ValueError):
        build_ffmpeg_args("", 24, "out.mp4")
    with pytest.raises(ValueError):
        build_ffmpeg_args("frames", 24, "")


def test_build_ffmpeg_args_ok() -> None:
    args = build_ffmpeg_args("C:/tmp/frames", 30, "C:/tmp/out.mp4")
    assert args[0] == "ffmpeg"
    assert "-framerate" in args and "-i" in args
    assert any(x.endswith("frame_%06d.png") for x in args)
    assert args[-1].endswith("out.mp4")


def test_images_io_resize_and_write_roundtrip() -> None:
    # Create a simple RGB image (H,W,3) in [0,1]
    h0, w0 = 8, 10
    r = BACKEND.ones(h0, w0) * 0.2
    g = BACKEND.ones(h0, w0) * 0.4
    b = BACKEND.ones(h0, w0) * 0.6
    rgb = BACKEND.stack_rgba(r, g, b, BACKEND.ones(h0, w0))
    # Drop alpha inside write; resize to a different size then write
    target: Resolution = {"width": 12, "height": 7}
    rgb_resized = resize_image(rgb, target)
    assert rgb_resized.shape[1] == target["width"]
    assert rgb_resized.shape[0] == target["height"]

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "frame.png")
        write_frame_png(path, rgb_resized)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_write_frame_png_accepts_rgba_and_collapses_alpha() -> None:
    # Build an RGBA image and ensure writer accepts it by collapsing to RGB.
    h, w = 3, 4
    r = BACKEND.ones(h, w) * 0.1
    g = BACKEND.ones(h, w) * 0.2
    b = BACKEND.ones(h, w) * 0.3
    a = BACKEND.ones(h, w)  # alpha channel present
    rgba = BACKEND.stack_rgba(r, g, b, a)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rgba.png")
        write_frame_png(path, rgba)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_write_frame_png_invalid_shape_raises() -> None:
    # 2D array should be rejected (expects H,W,C)
    zeros2d = BACKEND.zeros(4, 4)
    with pytest.raises(ValueError):
        write_frame_png("ignored.png", zeros2d)


def test_write_frame_png_invalid_channels_raises() -> None:
    # Shape (H,W,1) should be rejected (expects 3 or 4 channels)
    # Build using Protocol-typed numpy import to avoid Any in mypy.
    from procart.images_io import _NumpyModule as _NP

    np_mod: _NP = __import__("numpy")
    base2d = BACKEND.zeros(2, 3)
    arr = np_mod.stack([base2d], axis=-1)
    with pytest.raises(ValueError):
        write_frame_png("ignored.png", arr)


def test_write_frame_png_no_directory_branch_basename_ok() -> None:
    # Use a basename only to exercise the branch where no directory is created.
    h, w = 2, 3
    r = BACKEND.ones(h, w) * 0.1
    g = BACKEND.ones(h, w) * 0.2
    b = BACKEND.ones(h, w) * 0.3
    a = BACKEND.ones(h, w)
    rgba = BACKEND.stack_rgba(r, g, b, a)
    with tempfile.TemporaryDirectory() as d:
        cur = os.getcwd()
        os.chdir(d)
        try:
            write_frame_png("basename.png", rgba)
            assert os.path.exists(os.path.join(d, "basename.png"))
        finally:
            # Important on Windows: leave the temp dir before it is removed.
            os.chdir(cur)


def test_resize_image_invalid_resolution_raises() -> None:
    h, w = 3, 4
    rgba = BACKEND.stack_rgba(
        BACKEND.ones(h, w) * 0.1,
        BACKEND.ones(h, w) * 0.2,
        BACKEND.ones(h, w) * 0.3,
        BACKEND.ones(h, w),
    )
    bad_w: Resolution = {"width": 0, "height": 2}
    bad_h: Resolution = {"width": 2, "height": 0}
    with pytest.raises(ValueError):
        resize_image(rgba, bad_w)
    with pytest.raises(ValueError):
        resize_image(rgba, bad_h)
