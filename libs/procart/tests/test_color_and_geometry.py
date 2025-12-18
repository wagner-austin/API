from __future__ import annotations

import pytest

from procart.color import apply_tone_map, hsv_to_rgb
from procart.geometry import create_normalized_grid, normalized_distance
from procart.math_backend import BACKEND, FloatArray
from procart.types import Resolution, ToneMappingConfigExposureGamma


def _flatten(arr: FloatArray) -> list[float]:
    vals = arr.tolist()
    out: list[float] = []
    if isinstance(vals, list):
        for item in vals:
            if isinstance(item, list):
                for v in item:
                    out.append(float(v))
            else:
                out.append(float(item))
    return out


def test_hsv_to_rgb_basic() -> None:
    rgb = hsv_to_rgb(0.0, 0.0, 0.5)
    assert rgb.shape == (3,)
    expected = BACKEND.from_list([0.5, 0.5, 0.5])
    diff = hsv_to_rgb(0.0, 0.0, 0.5) - expected
    diff_vals = _flatten(diff)
    assert max(abs(v) for v in diff_vals) < 1e-6


def test_hsv_to_rgb_invalid_raises() -> None:
    # s outside [0,1]
    with pytest.raises(ValueError):
        hsv_to_rgb(0.1, 1.5, 0.5)


def test_apply_tone_map_exposure_gamma_monotonic() -> None:
    cfg: ToneMappingConfigExposureGamma = {
        "type": "exposure_gamma",
        "exposure": 1.5,
        "gamma": 2.2,
    }
    inp = BACKEND.from_list([0.1, 1.0, 5.0])
    out = apply_tone_map(inp, cfg)
    assert out.shape == (3,)
    arr = _flatten(out)
    assert all(0.0 <= v <= 1.0 for v in arr)
    assert arr[0] <= arr[1] <= arr[2]


def test_geometry_grid_and_distance() -> None:
    res: Resolution = {"width": 4, "height": 3}
    yy, xx = create_normalized_grid(res)
    assert yy.shape == (3, 4) and xx.shape == (3, 4)
    d = normalized_distance(xx, yy, 0.5, 0.5)
    assert d.shape == (3, 4)
    assert BACKEND.min_scalar(d) >= 0.0


def test_geometry_invalid_resolution_raises() -> None:
    bad: Resolution = {"width": 0, "height": 2}
    with pytest.raises(ValueError):
        create_normalized_grid(bad)


def test_distance_mismatched_shapes_raises() -> None:
    # Build mismatched shapes using backend
    _yy1, xx1 = BACKEND.normalized_grid(2, 2)
    yy2, _ = BACKEND.normalized_grid(3, 3)
    with pytest.raises(ValueError):
        _ = normalized_distance(xx1, yy2, 0.5, 0.5)
