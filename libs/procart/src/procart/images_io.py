from __future__ import annotations

import os
from typing import Protocol

from .math_backend import FloatArray
from .types import Resolution


class _NumpyModule(Protocol):
    def asarray(
        self, obj: FloatArray | list[list[list[float]]] | list[list[list[int]]], dtype: str
    ) -> FloatArray: ...
    def clip(self, a: FloatArray, a_min: float, a_max: float) -> FloatArray: ...
    def round(self, a: FloatArray) -> FloatArray: ...
    def take(self, a: FloatArray, indices: int, axis: int) -> FloatArray: ...
    def stack(self, arrays: list[FloatArray], axis: int) -> FloatArray: ...
    def reshape(self, a: FloatArray, newshape: tuple[int, int, int]) -> FloatArray: ...


class _PILImageModule(Protocol):
    BICUBIC: int

    def fromarray(self, obj: FloatArray, mode: str | None = None) -> _PILImageProto: ...


class _PILImageProto(Protocol):
    def save(self, fp: str, format: str | None = None) -> None: ...
    def resize(self, size: tuple[int, int], resample: int) -> _PILImageProto: ...
    def getdata(self) -> list[tuple[int, int, int]]: ...


def ensure_dir(path: str) -> None:
    """Ensure directory exists.

    Args:
        path: Directory path to create if missing.
    """
    os.makedirs(path, exist_ok=True)


def _rgb3_from_rgb_or_rgba(arr: FloatArray) -> FloatArray:
    np_mod: _NumpyModule = __import__("numpy")
    a = np_mod.asarray(arr, dtype="float32")
    shape = a.shape
    if len(shape) != 3:
        raise ValueError("expected an image array of shape (H, W, C)")
    c = int(shape[2])
    if c == 3:
        return a
    if c == 4:
        r = np_mod.take(a, 0, axis=-1)
        g = np_mod.take(a, 1, axis=-1)
        b = np_mod.take(a, 2, axis=-1)
        return np_mod.stack([r, g, b], axis=-1)
    raise ValueError("expected 3 or 4 channels")


def write_frame_png(path: str, rgb: FloatArray) -> None:
    """Write an RGB float32 image in [0,1] to PNG (uint8).

    Args:
        path: Output file path.
        rgb: Float32 RGB/RGBA array with values in [0,1].

    Raises:
        ValueError: If resolution cannot be inferred from array shape.
    """
    directory = os.path.dirname(path)
    if directory:
        ensure_dir(directory)
    np_mod: _NumpyModule = __import__("numpy")
    im_mod: _PILImageModule = __import__("PIL.Image", fromlist=["Image"])
    arr3 = _rgb3_from_rgb_or_rgba(rgb)
    # Scale to 0..255 and convert to uint8
    arr01 = np_mod.clip(arr3, 0.0, 1.0)
    arr255 = np_mod.round(arr01 * 255.0)
    arr_u8 = arr255.astype("uint8")
    img = im_mod.fromarray(arr_u8)
    img.save(path, format="PNG")


def resize_image(rgb: FloatArray, target_resolution: Resolution) -> FloatArray:
    """Resize an RGB float32 image using bicubic filtering.

    Args:
        rgb: Float32 RGB/RGBA array with values in [0,1].
        target_resolution: Target resolution.

    Returns:
        FloatArray: Resized RGB float32 array in [0,1].

    Raises:
        ValueError: If target resolution is invalid.
    """
    w = int(target_resolution["width"])  # Width
    h = int(target_resolution["height"])  # Height
    if w <= 0 or h <= 0:
        raise ValueError("target resolution must be positive")
    np_mod: _NumpyModule = __import__("numpy")
    im_mod: _PILImageModule = __import__("PIL.Image", fromlist=["Image"])
    arr3 = _rgb3_from_rgb_or_rgba(rgb)
    # Go through uint8 for PIL then back to float32 [0,1]
    arr01 = np_mod.clip(arr3, 0.0, 1.0)
    arr255 = np_mod.round(arr01 * 255.0)
    arr_u8 = arr255.astype("uint8")
    img = im_mod.fromarray(arr_u8)
    img2 = img.resize((w, h), resample=im_mod.BICUBIC)
    # Convert PIL image to nested list for typed np.asarray
    px = list(img2.getdata())
    rows: list[list[list[int]]] = []
    for y in range(h):
        start = y * w
        end = start + w
        row_px = px[start:end]
        row: list[list[int]] = []
        for rr, gg, bb in row_px:
            row.append([int(rr), int(gg), int(bb)])
        rows.append(row)
    arr_u8_resized = np_mod.asarray(rows, dtype="uint8")
    return arr_u8_resized.astype("float32") / 255.0


__all__ = ["ensure_dir", "resize_image", "write_frame_png"]
