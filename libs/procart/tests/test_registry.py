from __future__ import annotations

import pytest

from procart.registry import build_module, get_factory, list_available_modules
from procart.types import (
    BlackBackgroundLayerConfig,
    NeonOrbsLayerConfig,
    RecursiveRectsLayerConfig,
    Resolution,
)


def test_registry_lists_and_get_factory() -> None:
    names = list_available_modules()
    assert "black_background" in names
    assert "neon_orbs" in names
    assert "recursive_rects" in names
    assert "fractal_mandelbrot" in names
    assert "spiral_flow" in names

    f2 = get_factory("neon_orbs")
    assert callable(f2)
    with pytest.raises(ValueError):
        get_factory("unknown")


def test_build_module_from_config() -> None:
    bb: BlackBackgroundLayerConfig = {
        "module": "black_background",
        "id": "bb",
        "opacity": 1.0,
        "parallax_depth": 0.0,
        "composite": "normal",
    }
    no: NeonOrbsLayerConfig = {
        "module": "neon_orbs",
        "id": "no",
        "opacity": 1.0,
        "parallax_depth": 1.0,
        "composite": "normal",
        "count": 3,
        "glow": {
            "core_intensity": 1.4,
            "halo_intensity": 0.2,
            "core_radius": 0.02,
            "halo_radius": 0.2,
        },
    }
    rr: RecursiveRectsLayerConfig = {
        "module": "recursive_rects",
        "id": "rr",
        "opacity": 1.0,
        "parallax_depth": 0.5,
        "composite": "normal",
        "max_depth": 3,
        "min_size": 2,
    }

    m1 = build_module(bb)
    m2 = build_module(no)
    m3 = build_module(rr)
    # Also verify fractal_mandelbrot factory path works
    from procart.types import FractalMandelbrotLayerConfig

    fm: FractalMandelbrotLayerConfig = {
        "module": "fractal_mandelbrot",
        "id": "fm",
        "opacity": 1.0,
        "parallax_depth": 0.0,
        "composite": "normal",
        "max_iter": 2,
        "bailout": 2.0,
        "zoom": 1.0,
        "pan_x": 0.0,
        "pan_y": 0.0,
        "coloring": "smooth",
    }
    m4 = build_module(fm)
    res: Resolution = {"width": 8, "height": 6}
    out1 = m1.render_frame(0.0, res, 0, 0.0, 0.0)
    out2 = m2.render_frame(0.0, res, 0, 0.0, 0.0)
    out3 = m3.render_frame(0.0, res, 0, 0.0, 0.0)
    out4 = m4.render_frame(0.0, res, 0, 0.0, 0.0)
    assert out1.shape[0] == 6 and out1.shape[1] == 8 and out1.shape[2] == 4
    assert out2.shape[0] == 6 and out2.shape[1] == 8 and out2.shape[2] == 4
    assert out3.shape[0] == 6 and out3.shape[1] == 8 and out3.shape[2] == 4
    assert out4.shape[0] == 6 and out4.shape[1] == 8 and out4.shape[2] == 4


def test_registry_factory_mismatch_raises() -> None:
    # Prepare one config for each module, then invoke the wrong factory to
    # exercise the config mismatch validation branches inside closures.
    bb: BlackBackgroundLayerConfig = {
        "module": "black_background",
        "id": "bb",
        "opacity": 1.0,
        "parallax_depth": 0.0,
        "composite": "normal",
    }
    no: NeonOrbsLayerConfig = {
        "module": "neon_orbs",
        "id": "no",
        "opacity": 1.0,
        "parallax_depth": 1.0,
        "composite": "normal",
        "count": 1,
        "glow": {
            "core_intensity": 1.0,
            "halo_intensity": 0.1,
            "core_radius": 0.05,
            "halo_radius": 0.2,
        },
    }
    rr: RecursiveRectsLayerConfig = {
        "module": "recursive_rects",
        "id": "rr",
        "opacity": 1.0,
        "parallax_depth": 0.5,
        "composite": "normal",
        "max_depth": 1,
        "min_size": 1,
    }

    f_bb = get_factory("black_background")
    f_no = get_factory("neon_orbs")
    f_rr = get_factory("recursive_rects")

    with pytest.raises(ValueError):
        f_bb(no)
    with pytest.raises(ValueError):
        f_no(bb)
    with pytest.raises(ValueError):
        f_rr(no)
    # Also verify correct usage succeeds to avoid linter complaining about unused rr
    _ = f_rr(rr)
    # Exercise unknown module path for branch coverage
    with pytest.raises(ValueError):
        _ = get_factory("does_not_exist")


def test_registry_factories_fractal_and_spiral_paths() -> None:
    from procart.types import FractalMandelbrotLayerConfig, SpiralFlowLayerConfig

    fm: FractalMandelbrotLayerConfig = {
        "module": "fractal_mandelbrot",
        "id": "fm2",
        "opacity": 1.0,
        "parallax_depth": 0.0,
        "composite": "normal",
        "max_iter": 2,
        "bailout": 2.0,
        "zoom": 1.0,
        "pan_x": 0.0,
        "pan_y": 0.0,
        "coloring": "smooth",
    }
    sp: SpiralFlowLayerConfig = {
        "module": "spiral_flow",
        "id": "sp2",
        "opacity": 1.0,
        "parallax_depth": 0.0,
        "composite": "normal",
        "turns": 1.0,
        "radial_gain": 0.0,
        "angular_gain": 1.0,
        "falloff": 1.0,
    }
    f_fm = get_factory("fractal_mandelbrot")
    f_sp = get_factory("spiral_flow")
    m_fm = f_fm(fm)
    m_sp = f_sp(sp)
    res: Resolution = {"width": 4, "height": 3}
    o1 = m_fm.render_frame(0.0, res, 0, 0.0, 0.0)
    o2 = m_sp.render_frame(0.0, res, 0, 0.0, 0.0)
    assert o1.shape[0] == 3 and o2.shape[1] == 4
    # Mismatch branches to exercise raises inside factory functions
    import pytest

    with pytest.raises(ValueError):
        _ = f_fm(sp)  # wrong config for fractal factory
    with pytest.raises(ValueError):
        _ = f_sp(fm)  # wrong config for spiral factory
