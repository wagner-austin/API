"""Tests for probability calibrator implementations.

Tests isotonic regression and Platt scaling calibrators for
improving probability estimates from machine learning models.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.calibration import (
    Calibrator,
    create_isotonic_calibrator,
    create_platt_calibrator,
)
from covenant_ml.calibration.calibrator import (
    _apply_sigmoid,
    _clip_probabilities,
    _compute_brier_score,
    _compute_ece,
)
from covenant_ml.calibration.testing import (
    make_isotonic_config,
    make_isotonic_state,
    make_platt_config,
    make_platt_state,
)


def _int_array(*values: int) -> NDArray[np.int64]:
    """Create int64 array from values.

    Args:
        values: Integer values.

    Returns:
        NDArray of int64.
    """
    vals: list[int] = list(values)
    return np.array(vals, dtype=np.int64)


def _float_array(*values: float) -> NDArray[np.float64]:
    """Create float64 array from values.

    Args:
        values: Float values.

    Returns:
        NDArray of float64.
    """
    vals: list[float] = list(values)
    return np.array(vals, dtype=np.float64)


def _make_calibration_data(
    n_samples: int = 200,
    seed: int = 42,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Create synthetic calibration data.

    Args:
        n_samples: Number of samples.
        seed: Random seed.

    Returns:
        Tuple of (true_labels, predicted_probabilities).
    """
    rng = np.random.default_rng(seed)

    # Create labels
    y_true = np.zeros(n_samples, dtype=np.int64)
    y_true[: n_samples // 2] = 1
    rng.shuffle(y_true)

    # Create uncalibrated probabilities (somewhat correlated with labels)
    y_prob = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        # Use slice indexing with np.asarray for typed extraction
        label_slice = np.asarray(y_true[i : i + 1], dtype=np.int64).flat
        label: int = int(label_slice[0])
        if label == 1:
            y_prob[i] = rng.uniform(0.4, 0.9)
        else:
            y_prob[i] = rng.uniform(0.1, 0.6)

    return y_true, y_prob


# =============================================================================
# Isotonic Calibrator Tests
# =============================================================================


def test_create_isotonic_calibrator() -> None:
    """create_isotonic_calibrator creates Calibrator with isotonic config."""
    calibrator = create_isotonic_calibrator()
    assert calibrator.config["method"] == "isotonic"
    assert calibrator.config["clip_proba"] is True
    assert calibrator.config["eps"] == 1e-10


def test_create_isotonic_calibrator_custom_params() -> None:
    """create_isotonic_calibrator accepts custom parameters."""
    calibrator = create_isotonic_calibrator(clip_proba=False, eps=1e-8)
    assert calibrator.config["clip_proba"] is False
    assert calibrator.config["eps"] == 1e-8


def test_isotonic_calibrator_fit() -> None:
    """Isotonic calibrator fits on validation data."""
    calibrator = create_isotonic_calibrator()
    y_true, y_prob = _make_calibration_data()

    result = calibrator.fit(y_true, y_prob)

    # Check result structure
    assert result["state"]["method"] == "isotonic"
    # Isotonic regression produces multiple thresholds from the data
    x_thresholds = result["state"]["params"]["X_thresholds"]
    y_values = result["state"]["params"]["y_values"]
    assert len(x_thresholds) >= 2, "Isotonic should have at least 2 thresholds"
    assert len(y_values) >= 2, "Isotonic should have at least 2 y_values"
    assert len(x_thresholds) == len(y_values), "Thresholds and values must match"

    # Brier score should be reasonable
    assert 0.0 <= result["train_brier_before"] <= 1.0
    assert 0.0 <= result["train_brier_after"] <= 1.0

    # ECE should be reasonable
    assert 0.0 <= result["train_ece_before"] <= 1.0
    assert 0.0 <= result["train_ece_after"] <= 1.0


def test_isotonic_calibrator_is_fitted() -> None:
    """Isotonic calibrator.is_fitted returns correct state."""
    calibrator = create_isotonic_calibrator()
    assert calibrator.is_fitted is False

    y_true, y_prob = _make_calibration_data()
    calibrator.fit(y_true, y_prob)

    assert calibrator.is_fitted is True


def test_isotonic_calibrator_transform() -> None:
    """Isotonic calibrator transforms probabilities."""
    calibrator = create_isotonic_calibrator()
    y_true, y_prob = _make_calibration_data()
    calibrator.fit(y_true, y_prob)

    result = calibrator.transform(y_prob)

    assert result["method"] == "isotonic"
    assert result["raw_proba"].shape == y_prob.shape
    assert result["calibrated_proba"].shape == y_prob.shape

    # Calibrated probabilities should be in [0, 1]
    min_val: float = float(np.min(result["calibrated_proba"]))
    max_val: float = float(np.max(result["calibrated_proba"]))
    assert min_val >= 0.0
    assert max_val <= 1.0


def test_isotonic_calibrator_transform_not_fitted() -> None:
    """Isotonic calibrator transform raises if not fitted."""
    calibrator = create_isotonic_calibrator()
    y_prob = _float_array(0.3, 0.5, 0.7)

    with pytest.raises(RuntimeError, match="Calibrator not fitted"):
        calibrator.transform(y_prob)


def test_isotonic_calibrator_get_state() -> None:
    """Isotonic calibrator returns serializable state."""
    calibrator = create_isotonic_calibrator()
    y_true, y_prob = _make_calibration_data()
    calibrator.fit(y_true, y_prob)

    state = calibrator.get_state()

    assert state["method"] == "isotonic"
    assert "X_thresholds" in state["params"]
    assert "y_values" in state["params"]


def test_isotonic_calibrator_get_state_not_fitted() -> None:
    """Isotonic calibrator get_state raises if not fitted."""
    calibrator = create_isotonic_calibrator()

    with pytest.raises(RuntimeError, match="Calibrator not fitted"):
        calibrator.get_state()


def test_isotonic_calibrator_from_state() -> None:
    """Isotonic calibrator can be reconstructed from state."""
    calibrator = create_isotonic_calibrator()
    y_true, y_prob = _make_calibration_data()
    calibrator.fit(y_true, y_prob)
    state = calibrator.get_state()

    # Reconstruct from state
    restored = Calibrator.from_state(state)

    assert restored.is_fitted is True
    assert restored.config["method"] == "isotonic"

    # Transform should work
    result = restored.transform(y_prob)
    assert result["method"] == "isotonic"


def test_isotonic_calibrator_from_testing_state() -> None:
    """Isotonic calibrator can be reconstructed from testing module state."""
    state = make_isotonic_state(
        x_thresholds=[0.0, 0.3, 0.6, 1.0],
        y_values=[0.05, 0.3, 0.7, 0.95],
    )

    calibrator = Calibrator.from_state(state)

    assert calibrator.is_fitted is True
    y_prob = _float_array(0.1, 0.4, 0.8)
    result = calibrator.transform(y_prob)
    assert result["method"] == "isotonic"


# =============================================================================
# Platt Scaling Tests
# =============================================================================


def test_create_platt_calibrator() -> None:
    """create_platt_calibrator creates Calibrator with platt config."""
    calibrator = create_platt_calibrator()
    assert calibrator.config["method"] == "platt"
    assert calibrator.config["clip_proba"] is True


def test_create_platt_calibrator_custom_params() -> None:
    """create_platt_calibrator accepts custom parameters."""
    calibrator = create_platt_calibrator(clip_proba=False, eps=1e-5)
    assert calibrator.config["clip_proba"] is False
    assert calibrator.config["eps"] == 1e-5


def test_platt_calibrator_fit() -> None:
    """Platt calibrator fits on validation data."""
    calibrator = create_platt_calibrator()
    y_true, y_prob = _make_calibration_data()

    result = calibrator.fit(y_true, y_prob)

    # Check result structure
    assert result["state"]["method"] == "platt"
    assert "A" in result["state"]["params"]
    assert "B" in result["state"]["params"]

    # Brier score should be reasonable
    assert 0.0 <= result["train_brier_before"] <= 1.0
    assert 0.0 <= result["train_brier_after"] <= 1.0


def test_platt_calibrator_is_fitted() -> None:
    """Platt calibrator.is_fitted returns correct state."""
    calibrator = create_platt_calibrator()
    assert calibrator.is_fitted is False

    y_true, y_prob = _make_calibration_data()
    calibrator.fit(y_true, y_prob)

    assert calibrator.is_fitted is True


def test_platt_calibrator_transform() -> None:
    """Platt calibrator transforms probabilities."""
    calibrator = create_platt_calibrator()
    y_true, y_prob = _make_calibration_data()
    calibrator.fit(y_true, y_prob)

    result = calibrator.transform(y_prob)

    assert result["method"] == "platt"
    assert result["raw_proba"].shape == y_prob.shape
    assert result["calibrated_proba"].shape == y_prob.shape

    # Calibrated probabilities should be in [0, 1]
    min_val: float = float(np.min(result["calibrated_proba"]))
    max_val: float = float(np.max(result["calibrated_proba"]))
    assert min_val >= 0.0
    assert max_val <= 1.0


def test_platt_calibrator_transform_not_fitted() -> None:
    """Platt calibrator transform raises if not fitted."""
    calibrator = create_platt_calibrator()
    y_prob = _float_array(0.3, 0.5, 0.7)

    with pytest.raises(RuntimeError, match="Calibrator not fitted"):
        calibrator.transform(y_prob)


def test_platt_calibrator_get_state() -> None:
    """Platt calibrator returns serializable state."""
    calibrator = create_platt_calibrator()
    y_true, y_prob = _make_calibration_data()
    calibrator.fit(y_true, y_prob)

    state = calibrator.get_state()

    assert state["method"] == "platt"
    assert "A" in state["params"]
    assert "B" in state["params"]


def test_platt_calibrator_from_state() -> None:
    """Platt calibrator can be reconstructed from state."""
    calibrator = create_platt_calibrator()
    y_true, y_prob = _make_calibration_data()
    calibrator.fit(y_true, y_prob)
    state = calibrator.get_state()

    # Reconstruct from state
    restored = Calibrator.from_state(state)

    assert restored.is_fitted is True
    assert restored.config["method"] == "platt"

    # Transform should work
    result = restored.transform(y_prob)
    assert result["method"] == "platt"


def test_platt_calibrator_from_testing_state() -> None:
    """Platt calibrator can be reconstructed from testing module state."""
    state = make_platt_state(slope=2.0, intercept=-0.5)

    calibrator = Calibrator.from_state(state)

    assert calibrator.is_fitted is True
    y_prob = _float_array(0.1, 0.4, 0.8)
    result = calibrator.transform(y_prob)
    assert result["method"] == "platt"


# =============================================================================
# Direct Calibrator Instantiation Tests
# =============================================================================


def test_calibrator_direct_instantiation_isotonic() -> None:
    """Calibrator can be instantiated directly with config."""
    config = make_isotonic_config()
    calibrator = Calibrator(config)

    assert calibrator.config["method"] == "isotonic"
    assert calibrator.is_fitted is False


def test_calibrator_direct_instantiation_platt() -> None:
    """Calibrator can be instantiated directly with platt config."""
    config = make_platt_config()
    calibrator = Calibrator(config)

    assert calibrator.config["method"] == "platt"
    assert calibrator.is_fitted is False


# =============================================================================
# No Clipping Tests
# =============================================================================


def test_isotonic_calibrator_without_clipping() -> None:
    """Isotonic calibrator works without probability clipping."""
    calibrator = create_isotonic_calibrator(clip_proba=False)
    y_true, y_prob = _make_calibration_data()

    result = calibrator.fit(y_true, y_prob)
    assert result["state"]["config"]["clip_proba"] is False

    transform_result = calibrator.transform(y_prob)
    assert transform_result["method"] == "isotonic"


def test_platt_calibrator_without_clipping() -> None:
    """Platt calibrator works without probability clipping."""
    calibrator = create_platt_calibrator(clip_proba=False)
    y_true, y_prob = _make_calibration_data()

    result = calibrator.fit(y_true, y_prob)
    assert result["state"]["config"]["clip_proba"] is False

    transform_result = calibrator.transform(y_prob)
    assert transform_result["method"] == "platt"


# =============================================================================
# Defensive Check Tests
# =============================================================================


def test_isotonic_calibrator_transform_inconsistent_state_raises() -> None:
    """Isotonic calibrator raises RuntimeError if _iso_model is None.

    This tests the defensive check that guards against inconsistent internal
    state where _state is set but _iso_model is not initialized. This can only
    happen through direct manipulation of internal state.
    """
    calibrator = create_isotonic_calibrator()
    y_prob = _float_array(0.3, 0.5, 0.7)

    # Create inconsistent state: set _state without fitting (leaves _iso_model None)
    fake_state = make_isotonic_state()
    calibrator._state = fake_state

    with pytest.raises(RuntimeError, match="Isotonic model not initialized"):
        calibrator.transform(y_prob)


# =============================================================================
# Helper Function Tests
# =============================================================================


def test_compute_brier_score_perfect() -> None:
    """Brier score is 0 for perfect predictions."""
    y_true = _int_array(0, 1, 0, 1)
    y_prob = _float_array(0.0, 1.0, 0.0, 1.0)

    score = _compute_brier_score(y_true, y_prob)
    assert score == 0.0


def test_compute_brier_score_worst() -> None:
    """Brier score is 1 for worst predictions."""
    y_true = _int_array(0, 1, 0, 1)
    y_prob = _float_array(1.0, 0.0, 1.0, 0.0)

    score = _compute_brier_score(y_true, y_prob)
    assert score == 1.0


def test_compute_brier_score_random() -> None:
    """Brier score for 0.5 predictions is 0.25."""
    y_true = _int_array(0, 1, 0, 1)
    y_prob = _float_array(0.5, 0.5, 0.5, 0.5)

    score = _compute_brier_score(y_true, y_prob)
    assert abs(score - 0.25) < 1e-10


def test_compute_ece_perfect() -> None:
    """ECE is 0 for perfectly calibrated predictions."""
    # Predictions match frequency in each bin
    y_true = _int_array(0, 0, 0, 1, 1, 1)
    y_prob = _float_array(0.05, 0.05, 0.05, 0.95, 0.95, 0.95)

    ece = _compute_ece(y_true, y_prob, n_bins=10)
    # Should be close to 0
    assert ece < 0.1


def test_compute_ece_miscalibrated() -> None:
    """ECE is high for miscalibrated predictions."""
    # All predictions are 0.9 but half are negative
    y_true = _int_array(0, 0, 1, 1)
    y_prob = _float_array(0.9, 0.9, 0.9, 0.9)

    ece = _compute_ece(y_true, y_prob, n_bins=10)
    # Should be around 0.4 (0.9 - 0.5)
    assert ece > 0.3


def test_clip_probabilities() -> None:
    """Clip probabilities to [eps, 1-eps]."""
    proba = _float_array(0.0, 0.5, 1.0)
    eps = 0.01

    clipped = _clip_probabilities(proba, eps)

    # Use slice indexing with np.asarray for typed extraction
    clipped_slice_0 = np.asarray(clipped[0:1], dtype=np.float64).flat
    clipped_slice_1 = np.asarray(clipped[1:2], dtype=np.float64).flat
    clipped_slice_2 = np.asarray(clipped[2:3], dtype=np.float64).flat
    assert float(clipped_slice_0[0]) == 0.01
    assert float(clipped_slice_1[0]) == 0.5
    assert float(clipped_slice_2[0]) == 0.99


def test_apply_sigmoid() -> None:
    """Apply sigmoid transformation correctly."""
    y_prob = _float_array(0.0, 0.5, 1.0)
    slope = 1.0
    intercept = 0.0

    result = _apply_sigmoid(y_prob, slope, intercept)

    # Use slice indexing with np.asarray for typed extraction
    result_slice_0 = np.asarray(result[0:1], dtype=np.float64).flat
    result_slice_1 = np.asarray(result[1:2], dtype=np.float64).flat
    result_slice_2 = np.asarray(result[2:3], dtype=np.float64).flat
    result_0: float = float(result_slice_0[0])
    result_1: float = float(result_slice_1[0])
    result_2: float = float(result_slice_2[0])

    # At z=0, sigmoid = 0.5
    assert abs(result_0 - 0.5) < 1e-10
    # At z=0.5, sigmoid ~ 0.622
    assert 0.6 < result_1 < 0.65
    # At z=1.0, sigmoid ~ 0.731
    assert 0.7 < result_2 < 0.75


def test_apply_sigmoid_with_intercept() -> None:
    """Apply sigmoid with intercept."""
    y_prob = _float_array(0.5)
    slope = 2.0
    intercept = -1.0

    result = _apply_sigmoid(y_prob, slope, intercept)

    # Use slice indexing for typed extraction
    result_slice_0 = np.asarray(result[0:1], dtype=np.float64).flat
    result_0: float = float(result_slice_0[0])
    # z = 2*0.5 - 1 = 0, sigmoid = 0.5
    assert abs(result_0 - 0.5) < 1e-10


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_isotonic_calibrator_small_dataset() -> None:
    """Isotonic calibrator works on small dataset."""
    calibrator = create_isotonic_calibrator()
    y_true = _int_array(0, 0, 1, 1)
    y_prob = _float_array(0.2, 0.3, 0.7, 0.8)

    result = calibrator.fit(y_true, y_prob)
    assert result["state"]["method"] == "isotonic"


def test_platt_calibrator_small_dataset() -> None:
    """Platt calibrator works on small dataset."""
    calibrator = create_platt_calibrator()
    y_true = _int_array(0, 0, 1, 1)
    y_prob = _float_array(0.2, 0.3, 0.7, 0.8)

    result = calibrator.fit(y_true, y_prob)
    assert result["state"]["method"] == "platt"


def test_isotonic_calibrator_extreme_probabilities() -> None:
    """Isotonic calibrator handles extreme probabilities."""
    calibrator = create_isotonic_calibrator(clip_proba=True, eps=1e-10)
    y_true = _int_array(0, 1, 0, 1)
    y_prob = _float_array(1e-15, 0.5, 0.5, 1.0 - 1e-15)

    result = calibrator.fit(y_true, y_prob)
    assert result["state"]["method"] == "isotonic"


def test_platt_calibrator_extreme_probabilities() -> None:
    """Platt calibrator handles extreme probabilities."""
    calibrator = create_platt_calibrator(clip_proba=True, eps=1e-10)
    y_true = _int_array(0, 1, 0, 1)
    y_prob = _float_array(1e-15, 0.5, 0.5, 1.0 - 1e-15)

    result = calibrator.fit(y_true, y_prob)
    assert result["state"]["method"] == "platt"
