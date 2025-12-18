from __future__ import annotations

import pytest

from procart.modules.fractal_mandelbrot import FractalMandelbrot
from procart.modules.spiral_flow import SpiralFlow
from procart.types import (
    FractalMandelbrotLayerConfig,
    Resolution,
    SpiralFlowLayerConfig,
)


def _res(w: int, h: int) -> Resolution:
    return {"width": int(w), "height": int(h)}


def test_fractal_mandelbrot_basic_and_invalid() -> None:
    fm: FractalMandelbrotLayerConfig = {
        "module": "fractal_mandelbrot",
        "id": "fm",
        "opacity": 1.0,
        "parallax_depth": 0.0,
        "composite": "normal",
        "max_iter": 3,
        "bailout": 2.0,
        "zoom": 1.0,
        "pan_x": 0.0,
        "pan_y": 0.0,
        "coloring": "smooth",
    }
    mod = FractalMandelbrot(fm)
    out = mod.render_frame(0.0, _res(8, 6), 0, 0.0, 0.0)
    assert out.shape == (6, 8, 4)
    with pytest.raises(ValueError):
        _ = mod.render_frame(0.0, _res(0, 6), 0, 0.0, 0.0)


def test_spiral_flow_basic_and_invalid() -> None:
    sp: SpiralFlowLayerConfig = {
        "module": "spiral_flow",
        "id": "sp",
        "opacity": 1.0,
        "parallax_depth": 0.0,
        "composite": "normal",
        "turns": 2.0,
        "radial_gain": 0.0,
        "angular_gain": 1.0,
        "falloff": 1.0,
    }
    mod = SpiralFlow(sp)
    out = mod.render_frame(0.25, _res(7, 5), 123, 0.0, 0.0)
    assert out.shape == (5, 7, 4)
    with pytest.raises(ValueError):
        _ = mod.render_frame(0.0, _res(-1, 5), 0, 0.0, 0.0)

    # Schedule coverage: constant and linear
    sp_const: SpiralFlowLayerConfig = {
        **sp,
        "turns_schedule": {"type": "constant", "value": 0.5},
    }
    out2 = SpiralFlow(sp_const).render_frame(0.5, _res(4, 3), 0, 0.0, 0.0)
    assert out2.shape == (3, 4, 4)

    sp_lin: SpiralFlowLayerConfig = {
        **sp,
        "turns_schedule": {"type": "linear", "start": 0.0, "end": 2.0},
    }
    out3 = SpiralFlow(sp_lin).render_frame(0.5, _res(4, 3), 0, 0.0, 0.0)
    assert out3.shape == (3, 4, 4)
    # Exercise branch where no schedule is provided again to confirm stability
    out4 = SpiralFlow(sp).render_frame(0.75, _res(3, 2), 7, 0.0, 0.0)
    assert out4.shape == (2, 3, 4)
