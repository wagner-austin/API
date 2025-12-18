from __future__ import annotations

import pytest

from procart.math_backend import BACKEND
from procart.modules.neon_orbs import NeonOrbs
from procart.modules.recursive_rects import RecursiveRects
from procart.types import NeonOrbsLayerConfig, RecursiveRectsLayerConfig, Resolution


def _res(w: int, h: int) -> Resolution:
    return {"width": w, "height": h}


def test_neon_orbs_basic_invariants() -> None:
    cfg: NeonOrbsLayerConfig = {
        "module": "neon_orbs",
        "id": "orbs",
        "opacity": 1.0,
        "parallax_depth": 0.5,
        "count": 3,
        "glow": {
            "core_intensity": 2.0,
            "halo_intensity": 0.5,
            "core_radius": 0.05,
            "halo_radius": 0.12,
        },
        "composite": "normal",
    }
    mod = NeonOrbs(cfg)
    rgba = mod.render_frame(0.25, _res(16, 12), 42, 0.01, -0.02)
    assert rgba.shape == (12, 16, 4)
    r, g, b, a = BACKEND.split_rgba(rgba)
    # No negative RGB, alpha clamped to [0,1]
    assert BACKEND.min_scalar(r) >= 0.0
    assert BACKEND.min_scalar(g) >= 0.0
    assert BACKEND.min_scalar(b) >= 0.0
    assert BACKEND.min_scalar(a) >= 0.0


def test_recursive_rects_basic_invariants() -> None:
    cfg: RecursiveRectsLayerConfig = {
        "module": "recursive_rects",
        "id": "rects",
        "opacity": 1.0,
        "parallax_depth": 0.5,
        "max_depth": 3,
        "min_size": 2,
        "composite": "normal",
    }
    mod = RecursiveRects(cfg)
    rgba = mod.render_frame(0.5, _res(20, 10), 7, -0.03, 0.02)
    assert rgba.shape == (10, 20, 4)
    r, g, b, a = BACKEND.split_rgba(rgba)
    # Alpha should be >= 0, RGB non-negative
    assert BACKEND.min_scalar(r) >= 0.0
    assert BACKEND.min_scalar(g) >= 0.0
    assert BACKEND.min_scalar(b) >= 0.0
    assert BACKEND.min_scalar(a) >= 0.0


def test_neon_orbs_invalid_resolution_raises() -> None:
    cfg: NeonOrbsLayerConfig = {
        "module": "neon_orbs",
        "id": "orbs",
        "opacity": 0.8,
        "parallax_depth": 0.2,
        "count": 1,
        "glow": {
            "core_intensity": 1.5,
            "halo_intensity": 0.3,
            "core_radius": 0.05,
            "halo_radius": 0.1,
        },
        "composite": "normal",
    }
    mod = NeonOrbs(cfg)
    with pytest.raises(ValueError):
        _ = mod.render_frame(0.0, {"width": 0, "height": 4}, 0, 0.0, 0.0)


def test_recursive_rects_invalid_resolution_raises() -> None:
    cfg: RecursiveRectsLayerConfig = {
        "module": "recursive_rects",
        "id": "rects",
        "opacity": 1.0,
        "parallax_depth": 0.0,
        "max_depth": 1,
        "min_size": 1,
        "composite": "normal",
    }
    mod = RecursiveRects(cfg)
    with pytest.raises(ValueError):
        _ = mod.render_frame(0.0, {"width": -1, "height": 4}, 0, 0.0, 0.0)
