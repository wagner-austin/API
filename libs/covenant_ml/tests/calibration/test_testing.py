"""Tests for calibration testing utilities.

Tests the public test utility functions exported for consumers.
"""

from __future__ import annotations

from covenant_ml.calibration.testing import (
    make_isotonic_config,
    make_isotonic_state,
    make_platt_config,
    make_platt_state,
)

# =============================================================================
# Config Factory Tests
# =============================================================================


def test_make_isotonic_config_defaults() -> None:
    """make_isotonic_config creates config with defaults."""
    config = make_isotonic_config()

    assert config["method"] == "isotonic"
    assert config["clip_proba"] is True
    assert config["eps"] == 1e-10


def test_make_isotonic_config_custom_clip() -> None:
    """make_isotonic_config accepts custom clip_proba."""
    config = make_isotonic_config(clip_proba=False)

    assert config["clip_proba"] is False


def test_make_isotonic_config_custom_eps() -> None:
    """make_isotonic_config accepts custom eps."""
    config = make_isotonic_config(eps=1e-5)

    assert config["eps"] == 1e-5


def test_make_platt_config_defaults() -> None:
    """make_platt_config creates config with defaults."""
    config = make_platt_config()

    assert config["method"] == "platt"
    assert config["clip_proba"] is True
    assert config["eps"] == 1e-10


def test_make_platt_config_custom_clip() -> None:
    """make_platt_config accepts custom clip_proba."""
    config = make_platt_config(clip_proba=False)

    assert config["clip_proba"] is False


def test_make_platt_config_custom_eps() -> None:
    """make_platt_config accepts custom eps."""
    config = make_platt_config(eps=1e-8)

    assert config["eps"] == 1e-8


# =============================================================================
# State Factory Tests
# =============================================================================


def test_make_isotonic_state_defaults() -> None:
    """make_isotonic_state creates state with defaults."""
    state = make_isotonic_state()

    assert state["method"] == "isotonic"
    assert state["config"]["method"] == "isotonic"
    assert state["config"]["clip_proba"] is True
    assert state["config"]["eps"] == 1e-10
    assert state["params"]["X_thresholds"] == [0.0, 0.5, 1.0]
    assert state["params"]["y_values"] == [0.1, 0.5, 0.9]


def test_make_isotonic_state_custom_thresholds() -> None:
    """make_isotonic_state accepts custom thresholds."""
    state = make_isotonic_state(
        x_thresholds=[0.0, 0.25, 0.75, 1.0],
        y_values=[0.0, 0.3, 0.7, 1.0],
    )

    assert state["params"]["X_thresholds"] == [0.0, 0.25, 0.75, 1.0]
    assert state["params"]["y_values"] == [0.0, 0.3, 0.7, 1.0]


def test_make_isotonic_state_custom_config() -> None:
    """make_isotonic_state accepts custom config params."""
    state = make_isotonic_state(clip_proba=False, eps=1e-5)

    assert state["config"]["clip_proba"] is False
    assert state["config"]["eps"] == 1e-5


def test_make_platt_state_defaults() -> None:
    """make_platt_state creates state with defaults."""
    state = make_platt_state()

    assert state["method"] == "platt"
    assert state["config"]["method"] == "platt"
    assert state["config"]["clip_proba"] is True
    assert state["config"]["eps"] == 1e-10
    assert state["params"]["A"] == 1.0
    assert state["params"]["B"] == 0.0


def test_make_platt_state_custom_params() -> None:
    """make_platt_state accepts custom slope and intercept."""
    state = make_platt_state(slope=2.5, intercept=-0.5)

    assert state["params"]["A"] == 2.5
    assert state["params"]["B"] == -0.5


def test_make_platt_state_custom_config() -> None:
    """make_platt_state accepts custom config params."""
    state = make_platt_state(clip_proba=False, eps=1e-8)

    assert state["config"]["clip_proba"] is False
    assert state["config"]["eps"] == 1e-8


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_make_isotonic_state_empty_thresholds() -> None:
    """make_isotonic_state works with empty thresholds."""
    state = make_isotonic_state(x_thresholds=[], y_values=[])

    assert state["params"]["X_thresholds"] == []
    assert state["params"]["y_values"] == []


def test_make_isotonic_state_single_point() -> None:
    """make_isotonic_state works with single threshold."""
    state = make_isotonic_state(x_thresholds=[0.5], y_values=[0.5])

    assert state["params"]["X_thresholds"] == [0.5]
    assert state["params"]["y_values"] == [0.5]


def test_make_platt_state_zero_params() -> None:
    """make_platt_state works with zero parameters."""
    state = make_platt_state(slope=0.0, intercept=0.0)

    assert state["params"]["A"] == 0.0
    assert state["params"]["B"] == 0.0


def test_make_platt_state_negative_params() -> None:
    """make_platt_state works with negative parameters."""
    state = make_platt_state(slope=-1.5, intercept=-2.0)

    assert state["params"]["A"] == -1.5
    assert state["params"]["B"] == -2.0


def test_make_isotonic_state_extreme_eps() -> None:
    """make_isotonic_state works with extreme epsilon."""
    state = make_isotonic_state(eps=1e-20)
    assert state["config"]["eps"] == 1e-20


def test_make_platt_state_large_params() -> None:
    """make_platt_state works with large parameters."""
    state = make_platt_state(slope=100.0, intercept=-50.0)

    assert state["params"]["A"] == 100.0
    assert state["params"]["B"] == -50.0
