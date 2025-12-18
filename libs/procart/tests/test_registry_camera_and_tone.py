from __future__ import annotations

import pytest

from procart.math_backend import BACKEND
from procart.registry_camera import (
    build_camera_from_config,
    get_camera_path,
    list_available_camera_paths,
)
from procart.registry_composite import list_available_composite_ops
from procart.registry_post import list_available_post_effects
from procart.registry_tone import (
    get_tone_mapper,
    list_available_tone_mappers,
)


def test_camera_registry_list_and_fetch() -> None:
    names = list_available_camera_paths()
    assert names == ["circular", "figure_eight"]

    circ = get_camera_path("circular")
    fig8 = get_camera_path("figure_eight")

    # Spot-check deterministic values
    assert circ(0.0) == (1.0, 0.0)
    x1, y1 = circ(0.25)
    assert abs(x1 - 0.0) < 1e-6 and abs(y1 - 1.0) < 1e-6

    x2, y2 = fig8(0.0)
    assert x2 == 0.0 and y2 == 0.0
    x3, y3 = fig8(0.25)
    assert x3 == 1.0 and abs(y3 - 0.0) < 1e-6

    # Ensure the outputs are bounded in [-1, 1]
    for t in [0.0, 0.1, 0.5, 0.9]:
        cx, cy = circ(t)
        fx, fy = fig8(t)
        for v in (cx, cy, fx, fy):
            assert -1.0 <= v <= 1.0

    with pytest.raises(ValueError):
        get_camera_path("unknown")

    # Builder with typed config applies amplitude/phase
    cam = build_camera_from_config({"type": "circular", "amplitude": 0.5, "phase": 0.25})
    # circular at t+phase=0.25 -> (0,1) scaled by 0.5
    x, y = cam(0.0)
    assert abs(x - 0.0) < 1e-6 and abs(y - 0.5) < 1e-6


def _flatten_1d_or_2d(values: list[float] | list[list[float]]) -> list[float]:
    out: list[float] = []
    for item in values:
        if isinstance(item, list):
            for v in item:
                out.append(float(v))
        else:
            out.append(float(item))
    return out


def test_tone_registry_list_and_apply_all_paths() -> None:
    names = list_available_tone_mappers()
    assert names == ["exposure_gamma", "reinhard", "filmic"]

    # Use a bright vector to exercise clamping and curve behavior
    v = BACKEND.array3(2.0, 0.5, 0.1)
    for name in names:
        tm = get_tone_mapper(name)
        out = tm(v)
        # Shape preserved (3,)
        assert out.shape == v.shape
        # Values clamped to [0,1]
        lst = out.tolist()
        flat = _flatten_1d_or_2d(lst)
        for y in flat:
            assert 0.0 <= float(y) <= 1.0

    with pytest.raises(ValueError):
        get_tone_mapper("nope")


def test_misc_registries_names() -> None:
    posts = list_available_post_effects()
    comps = list_available_composite_ops()
    assert posts == ["bloom"]
    assert "normal" in comps and "screen" in comps and "darken" in comps
