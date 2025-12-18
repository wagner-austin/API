from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class FloatArray(Protocol):
    """Minimal protocol for float32 array semantics used by procart.

    Operations are purely elementwise; no mutation is required by the protocol.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    def astype(self, dtype: str, copy: bool = False) -> FloatArray: ...

    def __add__(self, other: FloatArray | float) -> FloatArray: ...
    def __radd__(self, other: FloatArray | float) -> FloatArray: ...
    def __sub__(self, other: FloatArray | float) -> FloatArray: ...
    def __rsub__(self, other: FloatArray | float) -> FloatArray: ...
    def __mul__(self, other: FloatArray | float) -> FloatArray: ...
    def __rmul__(self, other: FloatArray | float) -> FloatArray: ...
    def __truediv__(self, other: FloatArray | float) -> FloatArray: ...
    def __rtruediv__(self, other: FloatArray | float) -> FloatArray: ...
    def __pow__(self, power: float) -> FloatArray: ...
    def __getitem__(self, index: int) -> float | FloatArray: ...
    def item(self, index: int) -> float: ...
    def tolist(self) -> list[float] | list[list[float]]: ...


class _NumpyModule(Protocol):
    """Protocols for the subset of numpy APIs we consume.

    Use string dtype "float32" to avoid importing concrete numpy types in annotations.
    """

    def array(
        self,
        obj: list[float] | list[list[float]] | list[list[list[float]]],
        dtype: str,
    ) -> FloatArray: ...
    def linspace(self, start: float, stop: float, num: int, dtype: str) -> FloatArray: ...
    def broadcast_to(self, a: FloatArray, shape: tuple[int, int]) -> FloatArray: ...
    def reshape(self, a: FloatArray, newshape: tuple[int, int]) -> FloatArray: ...
    def maximum(self, a: FloatArray, b: FloatArray | float) -> FloatArray: ...
    def clip(self, a: FloatArray, a_min: float, a_max: float) -> FloatArray: ...
    def hypot(self, x: FloatArray, y: FloatArray) -> FloatArray: ...
    def power(self, a: FloatArray, exp: float) -> FloatArray: ...
    def zeros(self, shape: tuple[int, int], dtype: str) -> FloatArray: ...
    def ones(self, shape: tuple[int, int], dtype: str) -> FloatArray: ...
    def stack(self, arrays: list[FloatArray], axis: int) -> FloatArray: ...
    def abs(self, a: FloatArray) -> FloatArray: ...
    def exp(self, a: FloatArray) -> FloatArray: ...
    def take(self, a: FloatArray, indices: int, axis: int) -> FloatArray: ...
    def sin(self, a: FloatArray) -> FloatArray: ...
    def cos(self, a: FloatArray) -> FloatArray: ...
    def arctan2(self, y: FloatArray, x: FloatArray) -> FloatArray: ...


class MathBackend(Protocol):
    """Backend for creating and transforming float32 arrays.

    All functions must return float32 arrays conforming to FloatArray.
    """

    def array3(self, r: float, g: float, b: float) -> FloatArray: ...
    def from_list(self, values: list[float]) -> FloatArray: ...
    def linspace1d(self, start: float, stop: float, count: int) -> FloatArray: ...
    def broadcast_to_2d(self, arr: FloatArray, h: int, w: int) -> FloatArray: ...
    def maximum_scalar(self, a: FloatArray, b: float) -> FloatArray: ...
    def clip(self, a: FloatArray, lo: float, hi: float) -> FloatArray: ...
    def hypot(self, dx: FloatArray, dy: FloatArray) -> FloatArray: ...
    def power(self, a: FloatArray, exp: float) -> FloatArray: ...
    def min_scalar(self, a: FloatArray) -> float: ...
    def normalized_grid(self, h: int, w: int) -> tuple[FloatArray, FloatArray]: ...
    def zeros(self, h: int, w: int) -> FloatArray: ...
    def ones(self, h: int, w: int) -> FloatArray: ...
    def stack_rgba(
        self, r: FloatArray, g: FloatArray, b: FloatArray, a: FloatArray
    ) -> FloatArray: ...
    def split_rgba(
        self, rgba: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]: ...
    def abs(self, a: FloatArray) -> FloatArray: ...
    def exp(self, a: FloatArray) -> FloatArray: ...
    def channel(self, rgba: FloatArray, channel: int) -> FloatArray: ...
    def rect_mask(self, h: int, w: int, x0: int, y0: int, x1: int, y1: int) -> FloatArray: ...
    def sin(self, a: FloatArray) -> FloatArray: ...
    def cos(self, a: FloatArray) -> FloatArray: ...
    def atan2(self, y: FloatArray, x: FloatArray) -> FloatArray: ...


class _NumpyBackend(MathBackend):
    def __init__(self) -> None:
        # Dynamically import numpy and bind to protocol-typed variable to avoid Any.
        np_mod: _NumpyModule = __import__("numpy")
        self._np = np_mod

    def array3(self, r: float, g: float, b: float) -> FloatArray:
        return self._np.array([float(r), float(g), float(b)], dtype="float32")

    def from_list(self, values: list[float]) -> FloatArray:
        return self._np.array([float(v) for v in values], dtype="float32")

    def linspace1d(self, start: float, stop: float, count: int) -> FloatArray:
        return self._np.linspace(float(start), float(stop), int(count), dtype="float32")

    def broadcast_to_2d(self, arr: FloatArray, h: int, w: int) -> FloatArray:
        return self._np.broadcast_to(arr, (int(h), int(w)))

    def maximum_scalar(self, a: FloatArray, b: float) -> FloatArray:
        return self._np.maximum(a, float(b))

    def clip(self, a: FloatArray, lo: float, hi: float) -> FloatArray:
        return self._np.clip(a, float(lo), float(hi))

    def hypot(self, dx: FloatArray, dy: FloatArray) -> FloatArray:
        return self._np.hypot(dx, dy)

    def power(self, a: FloatArray, exp: float) -> FloatArray:
        return self._np.power(a, float(exp))

    def min_scalar(self, a: FloatArray) -> float:
        lst = a.tolist()
        flat: list[float] = []
        for item in lst:
            if isinstance(item, list):
                for v in item:
                    flat.append(float(v))
            else:
                flat.append(float(item))
        return min(flat) if flat else 0.0

    def normalized_grid(self, h: int, w: int) -> tuple[FloatArray, FloatArray]:
        yy1 = self._np.linspace(0.0, 1.0, int(h), dtype="float32")
        xx1 = self._np.linspace(0.0, 1.0, int(w), dtype="float32")
        yy2 = self._np.reshape(yy1, (int(h), 1))
        xx2 = self._np.reshape(xx1, (1, int(w)))
        yy = self._np.broadcast_to(yy2, (int(h), int(w)))
        xx = self._np.broadcast_to(xx2, (int(h), int(w)))
        return yy, xx

    def zeros(self, h: int, w: int) -> FloatArray:
        return self._np.zeros((int(h), int(w)), dtype="float32")

    def ones(self, h: int, w: int) -> FloatArray:
        return self._np.ones((int(h), int(w)), dtype="float32")

    def stack_rgba(self, r: FloatArray, g: FloatArray, b: FloatArray, a: FloatArray) -> FloatArray:
        return self._np.stack([r, g, b, a], axis=-1)

    def split_rgba(self, rgba: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        r = self.channel(rgba, 0)
        g = self.channel(rgba, 1)
        b = self.channel(rgba, 2)
        a = self.channel(rgba, 3)
        return r, g, b, a

    def abs(self, a: FloatArray) -> FloatArray:
        return self._np.abs(a)

    def exp(self, a: FloatArray) -> FloatArray:
        return self._np.exp(a)

    def channel(self, rgba: FloatArray, channel: int) -> FloatArray:
        return self._np.take(rgba, int(channel), axis=-1)

    def rect_mask(self, h: int, w: int, x0: int, y0: int, x1: int, y1: int) -> FloatArray:
        zeros = self._np.zeros((int(h), int(w)), dtype="float32")
        # Build mask by difference of two step fields to avoid indexed assignment.
        yy_line = self._np.linspace(0.0, float(h), int(h), dtype="float32")
        xx_line = self._np.linspace(0.0, float(w), int(w), dtype="float32")
        yy = self._np.reshape(yy_line, (int(h), 1))
        xx = self._np.reshape(xx_line, (1, int(w)))
        y0f = float(int(y0))
        y1f = float(int(y1))
        x0f = float(int(x0))
        x1f = float(int(x1))
        y_ge_y0 = (yy - y0f) * 1e6
        y_lt_y1 = (y1f - yy) * 1e6
        x_ge_x0 = (xx - x0f) * 1e6
        x_lt_x1 = (x1f - xx) * 1e6
        # Positive values indicate inside bounds; clip to binary 0/1 after big scaling
        y_mask = self._np.clip(y_ge_y0, 0.0, 1.0) * self._np.clip(y_lt_y1, 0.0, 1.0)
        x_mask = self._np.clip(x_ge_x0, 0.0, 1.0) * self._np.clip(x_lt_x1, 0.0, 1.0)
        return y_mask * x_mask + zeros * 0.0

    def sin(self, a: FloatArray) -> FloatArray:
        return self._np.sin(a)

    def cos(self, a: FloatArray) -> FloatArray:
        return self._np.cos(a)

    def atan2(self, y: FloatArray, x: FloatArray) -> FloatArray:
        return self._np.arctan2(y, x)


# Default backend used by the library.
BACKEND: MathBackend = _NumpyBackend()


__all__ = ["BACKEND", "FloatArray", "MathBackend"]
