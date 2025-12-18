from __future__ import annotations

from math import cos, sin

from ..color import hsv_to_rgb
from ..geometry import create_normalized_grid, normalized_distance
from ..math_backend import BACKEND, FloatArray
from ..types import NeonOrbsLayerConfig, Resolution


class NeonOrbs:
    """Neon orbs with core and halo glow over black.

    Deterministic motion and color selection based on seed and index.
    Returns HDR-linear RGBA with alpha in [0,1].
    """

    def __init__(self, config: NeonOrbsLayerConfig) -> None:
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
        acc_r = BACKEND.zeros(h, w)
        acc_g = BACKEND.zeros(h, w)
        acc_b = BACKEND.zeros(h, w)
        acc_a = BACKEND.zeros(h, w)
        count = int(self._cfg["count"])  # Count
        glow = self._cfg["glow"]
        core_r = float(glow["core_radius"])  # Core radius
        halo_r = float(glow["halo_radius"])  # Halo radius
        core_int = float(glow["core_intensity"])  # Core intensity
        halo_int = float(glow["halo_intensity"])  # Halo intensity

        for i in range(count):
            phase = (t_normalized + float(i + seed) * 0.12345) % 1.0
            cx = 0.5 + 0.25 * cos(2.0 * 3.14159265 * (phase + i * 0.07)) + camera_x * 0.2
            cy = 0.5 + 0.25 * sin(2.0 * 3.14159265 * (phase + i * 0.11)) + camera_y * 0.2
            d = normalized_distance(xx, yy, cx, cy)
            # Hard-edged circles: 1 inside radius, 0 outside
            core_mask = BACKEND.clip((core_r - d) * 500.0, 0.0, 1.0)
            halo_mask = BACKEND.clip((halo_r - d) * 100.0, 0.0, 1.0) - core_mask
            hue = (phase + float(i) * 0.21) % 1.0
            core_rgb = hsv_to_rgb(hue, 1.0, core_int)
            halo_rgb = hsv_to_rgb(hue, 0.7, halo_int)
            # Accumulate RGB (HDR additive)
            cr, cg, cb = core_rgb[0], core_rgb[1], core_rgb[2]
            hr, hg, hb = halo_rgb[0], halo_rgb[1], halo_rgb[2]
            acc_r = acc_r + core_mask * cr + halo_mask * hr
            acc_g = acc_g + core_mask * cg + halo_mask * hg
            acc_b = acc_b + core_mask * cb + halo_mask * hb
            # Alpha accumulates bounded
            acc_a = BACKEND.clip(acc_a + BACKEND.clip(core_mask + halo_mask, 0.0, 1.0), 0.0, 1.0)

        return BACKEND.stack_rgba(acc_r, acc_g, acc_b, acc_a)


__all__ = ["NeonOrbs"]
