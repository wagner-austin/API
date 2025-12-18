from __future__ import annotations

from math import cos, pi, sin
from typing import Final

from .math_backend import BACKEND, FloatArray
from .modules.base import VisualModule
from .types import FrameIndex, RenderTimingConfig, Resolution, ToneMappingConfig


def _alpha_over(back: FloatArray, front: FloatArray) -> FloatArray:
    r_b, g_b, b_b, a_b = BACKEND.split_rgba(back)
    r_f, g_f, b_f, a_f = BACKEND.split_rgba(front)
    one = BACKEND.ones(int(r_b.shape[0]), int(r_b.shape[1]))
    inv_a_f = one - a_f
    r = r_f + r_b * inv_a_f
    g = g_f + g_b * inv_a_f
    b = b_f + b_b * inv_a_f
    a = a_f + a_b * inv_a_f
    return BACKEND.stack_rgba(r, g, b, a)


class Layer:
    id: str
    module: VisualModule
    opacity: float
    parallax_depth: float

    def __init__(
        self, *, id: str, module: VisualModule, opacity: float, parallax_depth: float
    ) -> None:
        self.id = id
        self.module = module
        self.opacity = float(opacity)
        self.parallax_depth = float(parallax_depth)


class Scene:
    id: str
    description: str
    resolution: Resolution
    timing: RenderTimingConfig
    tone_mapping: ToneMappingConfig
    layers: list[Layer]

    def __init__(
        self,
        *,
        id: str,
        description: str,
        resolution: Resolution,
        timing: RenderTimingConfig,
        tone_mapping: ToneMappingConfig,
        layers: list[Layer],
    ) -> None:
        self.id = id
        self.description = description
        self.resolution = resolution
        self.timing = timing
        self.tone_mapping = tone_mapping
        self.layers = list(layers)

    def _camera(self, t: float) -> tuple[float, float]:
        amp: Final[float] = 0.05
        x = amp * sin(2.0 * pi * t)
        y = amp * cos(2.0 * pi * t)
        return x, y

    def render_frame(self, frame_index: FrameIndex) -> FloatArray:
        fps = int(self.timing["fps"])  # Fps
        duration = float(self.timing["duration_seconds"])  # Seconds
        if fps <= 0 or duration <= 0.0:
            raise ValueError("fps and duration must be positive")
        total_frames = fps * int(duration)
        if not (0 <= int(frame_index) < total_frames):
            raise ValueError("frame_index out of range")
        t = float(frame_index) / float(total_frames)
        cam_x, cam_y = self._camera(t)
        w = int(self.resolution["width"])  # Width
        h = int(self.resolution["height"])  # Height
        acc = BACKEND.stack_rgba(
            BACKEND.zeros(h, w),
            BACKEND.zeros(h, w),
            BACKEND.zeros(h, w),
            BACKEND.zeros(h, w),
        )
        for layer in self.layers:
            lx = cam_x * layer.parallax_depth
            ly = cam_y * layer.parallax_depth
            rgba = layer.module.render_frame(t, self.resolution, 0, lx, ly)
            r, g, b, a = BACKEND.split_rgba(rgba)
            a_scaled = a * float(layer.opacity)
            rgba_scaled = BACKEND.stack_rgba(r, g, b, a_scaled)
            acc = _alpha_over(acc, rgba_scaled)
        return acc


__all__ = ["Layer", "Scene"]
