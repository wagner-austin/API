from __future__ import annotations

from procart.color import hsv_to_rgb
from procart.math_backend import BACKEND, FloatArray


def test_hsv_sector_branches_cover_all() -> None:
    # Cover i == 0..5 branches by choosing hues across sectors
    sectors = [0.0, 0.2, 0.35, 0.55, 0.75, 0.95]
    for h in sectors:
        rgb: FloatArray = hsv_to_rgb(h, 0.8, 0.9)
        assert rgb.shape == (3,)
        # Components should be non-negative in linear space; use backend helper
        assert BACKEND.min_scalar(rgb) >= 0.0
