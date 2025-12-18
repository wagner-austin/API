from __future__ import annotations

from ..geometry import create_normalized_grid
from ..math_backend import BACKEND, FloatArray
from ..types import Resolution, SpiralFlowLayerConfig


class SpiralFlow:
    """Radial spiral field with angular modulation.

    Computes angle via atan2 and radius from center, then applies simple
    periodic modulation to create a spiral-like brightness.
    """

    def __init__(self, config: SpiralFlowLayerConfig) -> None:
        self._cfg = config

    def render_frame(
        self,
        t_normalized: float,
        resolution: Resolution,
        seed: int,
        camera_x: float,
        camera_y: float,
    ) -> FloatArray:
        w = int(resolution["width"])  # Width
        h = int(resolution["height"])  # Height
        if w <= 0 or h <= 0:
            raise ValueError("resolution width and height must be positive")
        yy, xx = create_normalized_grid(resolution)
        cx = xx - 0.5 + camera_x
        cy = yy - 0.5 + camera_y
        theta = BACKEND.atan2(cy, cx)
        r = BACKEND.hypot(cx, cy)
        # Parameters
        turns = float(self._cfg["turns"])  # number of windings
        rg = float(self._cfg["radial_gain"])
        ag = float(self._cfg["angular_gain"])
        fall = float(self._cfg["falloff"])
        phase = float(seed % 1000) * 0.001 + t_normalized
        # Optional schedule for turns
        if "turns_schedule" in self._cfg:
            sched = self._cfg["turns_schedule"]
            if sched["type"] == "constant":
                turns = float(sched["value"])
            if sched["type"] == "linear":
                start = float(sched["start"])  # linear discriminator
                end = float(sched["end"])  # linear discriminator
                turns = start + (end - start) * float(t_normalized)
        # Brightness modulation
        value = BACKEND.cos(theta * ag + r * turns * 6.2831853 + phase * 6.2831853)
        value = (value + 1.0) / 2.0
        value = value / (1.0 + r * fall + rg)
        value = BACKEND.clip(value, 0.0, 1.0)
        a = BACKEND.ones(h, w)
        return BACKEND.stack_rgba(value, value, value, a)


__all__ = ["SpiralFlow"]
