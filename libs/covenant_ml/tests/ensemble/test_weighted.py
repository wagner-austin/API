"""Tests for weighted ensemble functions."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.ensemble.types import (
    EnsembleOOFData,
    EnsembleWeights,
    ModelOOFPredictions,
)
from covenant_ml.ensemble.weighted import (
    compute_weighted_predictions,
    create_equal_weights,
    create_oof_data,
    extract_prediction_matrix,
    validate_oof_data,
    validate_weights,
)


def _float_array(*values: float) -> NDArray[np.float64]:
    """Create typed float64 array from values.

    Args:
        *values: Float values for the array.

    Returns:
        NDArray of float64.
    """
    return np.array(values, dtype=np.float64)


def _int_array(*values: int) -> NDArray[np.int64]:
    """Create typed int64 array from values.

    Args:
        *values: Int values for the array.

    Returns:
        NDArray of int64.
    """
    return np.array(values, dtype=np.int64)


def _make_model_predictions(
    name: str,
    predictions: tuple[float, ...],
) -> ModelOOFPredictions:
    """Create ModelOOFPredictions for testing.

    Args:
        name: Model name.
        predictions: Prediction values.

    Returns:
        ModelOOFPredictions instance.
    """
    n_samples = len(predictions)
    return ModelOOFPredictions(
        model_name=name,
        predictions=np.array(predictions, dtype=np.float64),
        fold_indices=np.zeros(n_samples, dtype=np.int64),
    )


def _make_oof_data(
    model_preds: tuple[tuple[str, tuple[float, ...]], ...],
    labels: tuple[int, ...],
) -> EnsembleOOFData:
    """Create EnsembleOOFData for testing.

    Args:
        model_preds: Tuple of (model_name, predictions) tuples.
        labels: True labels.

    Returns:
        EnsembleOOFData instance.
    """
    preds = tuple(_make_model_predictions(name, vals) for name, vals in model_preds)
    return EnsembleOOFData(
        model_predictions=preds,
        labels=np.array(labels, dtype=np.int64),
        n_samples=len(labels),
        n_models=len(model_preds),
    )


class TestValidateOOFData:
    """Tests for validate_oof_data function."""

    def test_valid_oof_data(self) -> None:
        """validate_oof_data accepts valid data."""
        oof_data = _make_oof_data(
            (("m1", (0.1, 0.9)), ("m2", (0.2, 0.8))),
            (0, 1),
        )

        # Should not raise
        validate_oof_data(oof_data)

    def test_raises_on_single_model(self) -> None:
        """validate_oof_data raises when n_models < 2."""
        oof_data = EnsembleOOFData(
            model_predictions=(_make_model_predictions("m1", (0.1, 0.9)),),
            labels=_int_array(0, 1),
            n_samples=2,
            n_models=1,
        )

        with pytest.raises(ValueError, match="at least 2 models"):
            validate_oof_data(oof_data)

    def test_raises_on_labels_mismatch(self) -> None:
        """validate_oof_data raises when labels length doesn't match n_samples."""
        oof_data = EnsembleOOFData(
            model_predictions=(
                _make_model_predictions("m1", (0.1, 0.9)),
                _make_model_predictions("m2", (0.2, 0.8)),
            ),
            labels=_int_array(0, 1, 0),  # 3 labels, 2 samples
            n_samples=2,
            n_models=2,
        )

        with pytest.raises(ValueError, match="Labels length"):
            validate_oof_data(oof_data)

    def test_raises_on_predictions_mismatch(self) -> None:
        """validate_oof_data raises when prediction length doesn't match n_samples."""
        m1 = ModelOOFPredictions(
            model_name="m1",
            predictions=_float_array(0.1, 0.9, 0.5),  # 3 predictions
            fold_indices=_int_array(0, 0, 0),
        )
        m2 = _make_model_predictions("m2", (0.2, 0.8))  # 2 predictions

        oof_data = EnsembleOOFData(
            model_predictions=(m1, m2),
            labels=_int_array(0, 1),
            n_samples=2,
            n_models=2,
        )

        with pytest.raises(ValueError, match="m1 has 3 predictions"):
            validate_oof_data(oof_data)

    def test_raises_on_fold_indices_mismatch(self) -> None:
        """validate_oof_data raises when fold_indices length doesn't match."""
        m1 = ModelOOFPredictions(
            model_name="m1",
            predictions=_float_array(0.1, 0.9),
            fold_indices=_int_array(0, 0, 0),  # 3 indices
        )
        m2 = _make_model_predictions("m2", (0.2, 0.8))

        oof_data = EnsembleOOFData(
            model_predictions=(m1, m2),
            labels=_int_array(0, 1),
            n_samples=2,
            n_models=2,
        )

        with pytest.raises(ValueError, match="m1 has 3 fold_indices"):
            validate_oof_data(oof_data)

    def test_raises_on_model_count_mismatch(self) -> None:
        """validate_oof_data raises when model_predictions length doesn't match n_models."""
        oof_data = EnsembleOOFData(
            model_predictions=(
                _make_model_predictions("m1", (0.1, 0.9)),
                _make_model_predictions("m2", (0.2, 0.8)),
            ),
            labels=_int_array(0, 1),
            n_samples=2,
            n_models=3,  # Says 3 but only 2 provided
        )

        with pytest.raises(ValueError, match="model_predictions length"):
            validate_oof_data(oof_data)


class TestValidateWeights:
    """Tests for validate_weights function."""

    def test_valid_weights(self) -> None:
        """validate_weights accepts valid weights."""
        weights = EnsembleWeights(
            weights=_float_array(0.6, 0.4),
            model_names=("m1", "m2"),
        )

        # Should not raise
        validate_weights(weights, n_models=2)

    def test_raises_on_length_mismatch(self) -> None:
        """validate_weights raises when weights length doesn't match n_models."""
        weights = EnsembleWeights(
            weights=_float_array(0.6, 0.4),
            model_names=("m1", "m2"),
        )

        with pytest.raises(ValueError, match="Weights length"):
            validate_weights(weights, n_models=3)

    def test_raises_on_names_mismatch(self) -> None:
        """validate_weights raises when model_names length doesn't match n_models."""
        weights = EnsembleWeights(
            weights=_float_array(0.6, 0.4),
            model_names=("m1",),  # Only 1 name
        )

        with pytest.raises(ValueError, match="model_names length"):
            validate_weights(weights, n_models=2)

    def test_raises_on_sum_not_one(self) -> None:
        """validate_weights raises when weights don't sum to 1."""
        weights = EnsembleWeights(
            weights=_float_array(0.5, 0.4),  # Sum = 0.9
            model_names=("m1", "m2"),
        )

        with pytest.raises(ValueError, match=r"sum to 1\.0"):
            validate_weights(weights, n_models=2)

    def test_raises_on_negative_weight(self) -> None:
        """validate_weights raises when any weight is negative."""
        weights = EnsembleWeights(
            weights=_float_array(1.2, -0.2),  # Sum = 1.0 but negative
            model_names=("m1", "m2"),
        )

        with pytest.raises(ValueError, match="non-negative"):
            validate_weights(weights, n_models=2)


class TestCreateEqualWeights:
    """Tests for create_equal_weights function."""

    def test_creates_equal_weights(self) -> None:
        """create_equal_weights creates weights summing to 1."""
        weights = create_equal_weights(("m1", "m2", "m3"))

        assert len(weights["weights"]) == 3
        assert weights["model_names"] == ("m1", "m2", "m3")
        assert np.isclose(float(np.sum(weights["weights"])), 1.0)
        expected = _float_array(1 / 3, 1 / 3, 1 / 3)
        assert np.allclose(weights["weights"], expected)

    def test_two_models(self) -> None:
        """create_equal_weights works with 2 models."""
        weights = create_equal_weights(("a", "b"))

        expected = _float_array(0.5, 0.5)
        assert np.allclose(weights["weights"], expected)

    def test_raises_on_single_model(self) -> None:
        """create_equal_weights raises with fewer than 2 models."""
        with pytest.raises(ValueError, match="at least 2 models"):
            create_equal_weights(("m1",))

    def test_raises_on_empty(self) -> None:
        """create_equal_weights raises with empty tuple."""
        with pytest.raises(ValueError, match="at least 2 models"):
            create_equal_weights(())


class TestCreateOOFData:
    """Tests for create_oof_data function."""

    def test_creates_valid_oof_data(self) -> None:
        """create_oof_data creates validated OOF data."""
        m1 = _make_model_predictions("m1", (0.1, 0.9))
        m2 = _make_model_predictions("m2", (0.2, 0.8))
        labels: NDArray[np.int64] = _int_array(0, 1)

        oof_data = create_oof_data((m1, m2), labels)

        assert oof_data["n_samples"] == 2
        assert oof_data["n_models"] == 2

    def test_raises_on_single_model(self) -> None:
        """create_oof_data raises with single model."""
        m1 = _make_model_predictions("m1", (0.1, 0.9))
        labels: NDArray[np.int64] = _int_array(0, 1)

        with pytest.raises(ValueError, match="at least 2 models"):
            create_oof_data((m1,), labels)


class TestComputeWeightedPredictions:
    """Tests for compute_weighted_predictions function."""

    def test_equal_weights(self) -> None:
        """compute_weighted_predictions with equal weights averages predictions."""
        oof_data = _make_oof_data(
            (("m1", (0.2, 0.8)), ("m2", (0.4, 0.6))),
            (0, 1),
        )
        weights = create_equal_weights(("m1", "m2"))

        result = compute_weighted_predictions(oof_data, weights)

        # Average of (0.2, 0.8) and (0.4, 0.6) = (0.3, 0.7)
        expected = _float_array(0.3, 0.7)
        assert np.allclose(result["predictions"], expected)

    def test_custom_weights(self) -> None:
        """compute_weighted_predictions applies custom weights correctly."""
        oof_data = _make_oof_data(
            (("m1", (0.0, 1.0)), ("m2", (1.0, 0.0))),
            (0, 1),
        )
        weights = EnsembleWeights(
            weights=_float_array(0.7, 0.3),
            model_names=("m1", "m2"),
        )

        result = compute_weighted_predictions(oof_data, weights)

        # 0.7 * (0, 1) + 0.3 * (1, 0) = (0.3, 0.7)
        expected = _float_array(0.3, 0.7)
        assert np.allclose(result["predictions"], expected)

    def test_model_contributions(self) -> None:
        """compute_weighted_predictions returns per-model contributions."""
        oof_data = _make_oof_data(
            (("m1", (0.2, 0.8)), ("m2", (0.4, 0.6))),
            (0, 1),
        )
        weights = EnsembleWeights(
            weights=_float_array(0.6, 0.4),
            model_names=("m1", "m2"),
        )

        result = compute_weighted_predictions(oof_data, weights)

        # Contributions: 0.6 * (0.2, 0.8) = (0.12, 0.48)
        #                0.4 * (0.4, 0.6) = (0.16, 0.24)
        contributions = result["model_contributions"]
        assert contributions.shape == (2, 2)
        expected_m1 = _float_array(0.12, 0.48)
        expected_m2 = _float_array(0.16, 0.24)
        assert np.allclose(contributions[0, :], expected_m1)
        assert np.allclose(contributions[1, :], expected_m2)


class TestExtractPredictionMatrix:
    """Tests for extract_prediction_matrix function."""

    def test_extracts_correct_shape(self) -> None:
        """extract_prediction_matrix returns (n_models, n_samples) matrix."""
        oof_data = _make_oof_data(
            (("m1", (0.1, 0.2, 0.3)), ("m2", (0.4, 0.5, 0.6))),
            (0, 1, 0),
        )

        matrix = extract_prediction_matrix(oof_data)

        assert matrix.shape == (2, 3)

    def test_extracts_correct_values(self) -> None:
        """extract_prediction_matrix preserves prediction values."""
        oof_data = _make_oof_data(
            (("m1", (0.1, 0.9)), ("m2", (0.2, 0.8))),
            (0, 1),
        )

        matrix = extract_prediction_matrix(oof_data)

        expected_m1 = _float_array(0.1, 0.9)
        expected_m2 = _float_array(0.2, 0.8)
        assert np.allclose(matrix[0, :], expected_m1)
        assert np.allclose(matrix[1, :], expected_m2)
