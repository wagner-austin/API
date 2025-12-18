from __future__ import annotations

import pytest

from procart.registry_schedule import (
    get_param_schedule,
    list_available_param_schedules,
)


def test_schedule_registry_lists_and_get() -> None:
    names = list_available_param_schedules()
    assert names == ["constant", "linear"]

    const = get_param_schedule("constant", start=3.0)
    assert const(0.0) == 3.0 and const(0.5) == 3.0

    lin = get_param_schedule("linear", start=0.0, end=10.0)
    assert lin(0.0) == 0.0
    assert lin(0.5) == 5.0
    assert lin(1.0) == 10.0

    with pytest.raises(ValueError):
        get_param_schedule("nope")


def test_schedule_linear_bounds_and_values() -> None:
    lin = get_param_schedule("linear", start=-1.0, end=1.0)
    # Below 0 clamps to 0 -> returns start
    assert lin(-0.5) == -1.0
    # Midpoint
    assert lin(0.25) == -0.5
    # Above 1 clamps to 1 -> returns end
    assert lin(2.5) == 1.0
