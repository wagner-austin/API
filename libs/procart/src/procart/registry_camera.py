from __future__ import annotations

from typing import Final, Protocol

from .types import CameraConfig


class CameraPath(Protocol):
    """2D camera path returning normalized x/y offsets.

    Args:
        t_normalized: Animation time in [0, 1).

    Returns:
        tuple[float, float]: (x, y) camera offsets.
    """

    def __call__(self, t_normalized: float) -> tuple[float, float]: ...


def _circular(t_normalized: float) -> tuple[float, float]:
    import math as _m

    ang = 2.0 * _m.pi * float(t_normalized)
    return _m.cos(ang), _m.sin(ang)


def _figure_eight(t_normalized: float) -> tuple[float, float]:
    import math as _m

    ang = 2.0 * _m.pi * float(t_normalized)
    # Lemniscate-like trace
    return _m.sin(ang), _m.sin(ang) * _m.cos(ang)


_NAMES: Final[tuple[str, ...]] = ("circular", "figure_eight")


def list_available_camera_paths() -> list[str]:
    """List registered camera path names."""

    return list(_NAMES)


def get_camera_path(name: str) -> CameraPath:
    """Fetch a camera path by name.

    Args:
        name: Registered path name.

    Returns:
        CameraPath: Callable taking t_normalized -> (x, y).

    Raises:
        ValueError: If name is unknown.
    """

    if name == "circular":
        return _circular
    if name == "figure_eight":
        return _figure_eight
    raise ValueError(f"unknown camera path: {name}")


def build_camera_from_config(cfg: CameraConfig) -> CameraPath:
    """Build a camera path from a typed camera configuration.

    Applies phase offset and amplitude scaling to a base path selected by
    the discriminated ``type`` field.

    Args:
        cfg: Typed camera configuration union.

    Returns:
        CameraPath: Callable that returns (x, y) offsets for ``t`` in [0, 1).

    Raises:
        ValueError: If the camera ``type`` is unknown.
    """
    path = get_camera_path(cfg["type"])  # may raise ValueError for unknown
    amp = float(cfg["amplitude"])  # scale of offsets
    phase = float(cfg["phase"])  # additive phase in cycles

    def _wrapped(t_normalized: float) -> tuple[float, float]:
        t = float(t_normalized) + phase
        # normalize t to avoid drift in user inputs; keep periodicity within [0,1)
        t = t - float(int(t)) if t >= 1.0 or t < 0.0 else t
        x, y = path(t)
        return amp * x, amp * y

    return _wrapped


__all__ = [
    "CameraPath",
    "build_camera_from_config",
    "get_camera_path",
    "list_available_camera_paths",
]
