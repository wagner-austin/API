from __future__ import annotations

import pytest

from procart.modules.background import BlackBackground
from procart.scene import Layer, Scene
from procart.types import RenderTimingConfig, Resolution, ToneMappingConfigExposureGamma


def _res(w: int, h: int) -> Resolution:
    return {"width": w, "height": h}


def _timing() -> RenderTimingConfig:
    return {"duration_seconds": 2.0, "fps": 24, "supersample_factor": 1}


def _tone() -> ToneMappingConfigExposureGamma:
    return {"type": "exposure_gamma", "exposure": 1.0, "gamma": 2.2}


def test_black_background_module() -> None:
    mod = BlackBackground()
    rgba = mod.render_frame(0.0, _res(8, 4), 0, 0.0, 0.0)
    assert rgba.shape == (4, 8, 4)
    # Alpha channel is 1 everywhere
    from procart.math_backend import BACKEND

    r, g, b, a = BACKEND.split_rgba(rgba)
    assert BACKEND.min_scalar(r) == 0.0
    assert BACKEND.min_scalar(g) == 0.0
    assert BACKEND.min_scalar(b) == 0.0
    assert BACKEND.min_scalar(a) == 1.0
    # And ensure RGB all zeros by recomposing
    zeros = BACKEND.stack_rgba(r, g, b, a)
    r2, g2, b2, _ = BACKEND.split_rgba(zeros)
    assert BACKEND.min_scalar(r2) == 0.0
    assert BACKEND.min_scalar(g2) == 0.0
    assert BACKEND.min_scalar(b2) == 0.0


def test_scene_with_single_background_layer() -> None:
    scene = Scene(
        id="bg_only",
        description="background only",
        resolution=_res(6, 5),
        timing=_timing(),
        tone_mapping=_tone(),
        layers=[Layer(id="bg", module=BlackBackground(), opacity=1.0, parallax_depth=0.0)],
    )
    out = scene.render_frame(0)
    assert out.shape == (5, 6, 4)


def test_scene_frame_index_bounds() -> None:
    scene = Scene(
        id="bg_only",
        description="background only",
        resolution=_res(4, 4),
        timing=_timing(),
        tone_mapping=_tone(),
        layers=[Layer(id="bg", module=BlackBackground(), opacity=1.0, parallax_depth=0.0)],
    )
    with pytest.raises(ValueError):
        scene.render_frame(10_000)


def test_background_invalid_resolution_raises() -> None:
    mod = BlackBackground()
    with pytest.raises(ValueError):
        _ = mod.render_frame(0.0, {"width": 0, "height": 4}, 0, 0.0, 0.0)


def test_scene_invalid_timing_raises() -> None:
    scene = Scene(
        id="bad_timing",
        description="invalid timing",
        resolution=_res(4, 4),
        timing={"duration_seconds": 2.0, "fps": 0, "supersample_factor": 1},
        tone_mapping=_tone(),
        layers=[Layer(id="bg", module=BlackBackground(), opacity=1.0, parallax_depth=0.0)],
    )
    with pytest.raises(ValueError):
        _ = scene.render_frame(0)


def test_alpha_over_composites_front_over_back() -> None:
    from procart.math_backend import BACKEND
    from procart.scene import _alpha_over

    h, w = 2, 3
    zeros = BACKEND.zeros(h, w)
    ones = BACKEND.ones(h, w)
    back = BACKEND.stack_rgba(zeros, zeros, zeros, zeros)
    front = BACKEND.stack_rgba(ones, zeros, zeros, ones)
    out = _alpha_over(back, front)
    r, g, b, a = BACKEND.split_rgba(out)
    assert BACKEND.min_scalar(r) == 1.0
    assert BACKEND.min_scalar(g) == 0.0
    assert BACKEND.min_scalar(b) == 0.0
    assert BACKEND.min_scalar(a) == 1.0
