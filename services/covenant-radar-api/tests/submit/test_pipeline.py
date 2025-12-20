"""Tests for submit pipeline functions.

Tests data loading, prediction, and submission writing.
Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from scripts.submit.pipeline import (
    load_training_data,
    predict,
    write_submission,
)

from .conftest import get_captured_console

# =============================================================================
# Fake Classifiers for Testing
# =============================================================================


class FakeClassifier2D:
    """Fake classifier returning 2D probability arrays."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return 2D probability array.

        Args:
            x: Input features.

        Returns:
            2D array with shape (n_samples, 2).
        """
        n_samples: int = int(x.shape[0])
        probs: NDArray[np.float64] = np.column_stack(
            [
                np.full(n_samples, 0.3, dtype=np.float64),
                np.full(n_samples, 0.7, dtype=np.float64),
            ]
        )
        return probs


class FakeClassifier1D:
    """Fake classifier returning 1D probability arrays."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return 1D probability array.

        Args:
            x: Input features.

        Returns:
            1D array with shape (n_samples,).
        """
        n_samples: int = int(x.shape[0])
        return np.full(n_samples, 0.8, dtype=np.float64)


# =============================================================================
# Test Array Factories
# =============================================================================


def make_test_array_2x2() -> NDArray[np.float64]:
    """Create a 2x2 test array with known values."""
    arr: NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
    arr[0, 0] = 1.0
    arr[0, 1] = 2.0
    arr[1, 0] = 3.0
    arr[1, 1] = 4.0
    return arr


def make_test_array_1x2() -> NDArray[np.float64]:
    """Create a 1x2 test array with known values."""
    arr: NDArray[np.float64] = np.zeros((1, 2), dtype=np.float64)
    arr[0, 0] = 1.0
    arr[0, 1] = 2.0
    return arr


# =============================================================================
# Tests
# =============================================================================


class TestLoadTrainingData:
    """Tests for load_training_data function."""

    def test_load_training_data(self, timeseries_fixture_dir: Path) -> None:
        """Test loading training data from fixture."""
        dataset = load_training_data(
            data_dir=timeseries_fixture_dir,
            aggregation="last",
            include_rank_features=False,
            include_diff_features=False,
        )

        # Extract typed arrays
        x_features: NDArray[np.float64] = dataset["x"]
        y_labels: NDArray[np.int64] = dataset["y"]
        n_samples_x: int = int(x_features.shape[0])
        n_samples_y: int = int(y_labels.shape[0])

        assert n_samples_x == 3  # 3 entities
        assert n_samples_y == 3
        assert len(dataset["meta"]["feature_names"]) == dataset["meta"]["n_features"]

        # Check console output
        captured = get_captured_console()
        assert len(captured.messages) >= 2


class TestPredict:
    """Tests for predict function."""

    def test_predict_with_valid_input(self) -> None:
        """Test prediction with valid input."""
        classifier = FakeClassifier2D()
        x_test = make_test_array_2x2()
        entity_ids: tuple[str, ...] = ("A", "B")

        result = predict(classifier, x_test, entity_ids)

        assert result["n_samples"] == 2
        assert result["entity_ids"] == ("A", "B")
        assert len(result["predictions"]) == 2
        assert abs(result["predictions"][0] - 0.7) < 0.001
        assert abs(result["predictions"][1] - 0.7) < 0.001

    def test_predict_mismatched_entity_ids_raises(self) -> None:
        """Test that mismatched entity IDs raises ValueError."""
        classifier = FakeClassifier2D()
        x_test = make_test_array_2x2()
        entity_ids: tuple[str, ...] = ("A",)  # Only 1, but 2 samples

        with pytest.raises(ValueError, match="entity_ids length 1 != samples 2"):
            predict(classifier, x_test, entity_ids)

    def test_predict_with_1d_proba_output(self) -> None:
        """Test prediction with 1D probability output (some backends)."""
        classifier = FakeClassifier1D()
        x_test = make_test_array_1x2()
        entity_ids: tuple[str, ...] = ("A",)

        result = predict(classifier, x_test, entity_ids)

        assert result["n_samples"] == 1
        assert abs(result["predictions"][0] - 0.8) < 0.001


class TestWriteSubmission:
    """Tests for write_submission function."""

    def test_write_submission_creates_file(self, tmp_path: Path) -> None:
        """Test that write_submission creates a CSV file."""
        output_path = tmp_path / "submission.csv"
        entity_ids: tuple[str, ...] = ("A", "B", "C")
        predictions: tuple[float, ...] = (0.1, 0.5, 0.9)

        n_rows = write_submission(output_path, entity_ids, predictions)

        assert n_rows == 3
        assert output_path.exists()

        # Read and verify content
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 4  # Header + 3 rows
        assert lines[0] == "customer_ID,prediction"
        assert lines[1].startswith("A,0.1")
        assert lines[2].startswith("B,0.5")
        assert lines[3].startswith("C,0.9")

    def test_write_submission_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that write_submission creates parent directories."""
        output_path = tmp_path / "nested" / "dir" / "submission.csv"
        entity_ids: tuple[str, ...] = ("X",)
        predictions: tuple[float, ...] = (0.5,)

        n_rows = write_submission(output_path, entity_ids, predictions)

        assert n_rows == 1
        assert output_path.exists()
        assert output_path.parent.exists()

    def test_write_submission_mismatched_lengths_raises(self, tmp_path: Path) -> None:
        """Test that mismatched lengths raises ValueError."""
        output_path = tmp_path / "submission.csv"
        entity_ids: tuple[str, ...] = ("A", "B")
        predictions: tuple[float, ...] = (0.1,)  # Only 1

        with pytest.raises(ValueError, match="entity_ids length 2 != predictions 1"):
            write_submission(output_path, entity_ids, predictions)
