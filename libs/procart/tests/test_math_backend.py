from __future__ import annotations

import pytest

from procart.math_backend import BACKEND


def test_min_scalar_on_1d_and_2d() -> None:
    a1 = BACKEND.from_list([0.3, 0.1, 0.2])
    assert BACKEND.min_scalar(a1) == pytest.approx(0.1, abs=1e-6)
    yy, xx = BACKEND.normalized_grid(2, 3)
    # All values are in [0,1], min should be exactly 0.0
    assert BACKEND.min_scalar(yy) == 0.0
    assert BACKEND.min_scalar(xx) == 0.0


def test_normalized_grid_shapes() -> None:
    yy, xx = BACKEND.normalized_grid(3, 5)
    assert yy.shape == (3, 5)
    assert xx.shape == (3, 5)


def test_linspace_and_broadcast() -> None:
    arr = BACKEND.linspace1d(0.0, 1.0, 4)
    assert arr.shape == (4,)
    grid = BACKEND.broadcast_to_2d(arr, 3, 4)
    assert grid.shape == (3, 4)


def test_min_scalar_empty_returns_zero() -> None:
    empty = BACKEND.from_list([])
    assert BACKEND.min_scalar(empty) == 0.0


def test_abs_and_exp_cover() -> None:
    a = BACKEND.from_list([-1.0, 0.0, 1.0])
    aa = BACKEND.abs(a)
    v0 = float(aa.item(0))
    v1 = float(aa.item(1))
    assert v0 == 1.0 and v1 == 0.0
    ee = BACKEND.exp(BACKEND.from_list([0.0]))
    assert float(ee.item(0)) == pytest.approx(1.0, abs=1e-6)


def test_trig_and_atan2_cover() -> None:
    a = BACKEND.from_list([0.0, 3.14159265 / 2.0])
    s = BACKEND.sin(a)
    c = BACKEND.cos(a)
    # sin(0)=0, cos(0)=1
    assert float(s.item(0)) == pytest.approx(0.0, abs=1e-6)
    assert float(c.item(0)) == pytest.approx(1.0, abs=1e-6)
    # atan2 on a simple grid
    yy, xx = BACKEND.normalized_grid(2, 2)
    ang = BACKEND.atan2(yy, xx)
    assert ang.shape == yy.shape
