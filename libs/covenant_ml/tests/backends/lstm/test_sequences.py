"""Tests for LSTM sequence building utilities.

Tests build_sequences and reshape_flat_to_pseudo_sequences functions.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.backends.lstm.sequences import (
    SequenceData,
    build_sequences,
    reshape_flat_to_pseudo_sequences,
)


def _make_entity_year_data(
    n_entities: int = 5,
    years_per_entity: int = 4,
    n_features: int = 3,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Create synthetic entity-year panel data for testing.

    Args:
        n_entities: Number of unique entities.
        years_per_entity: Number of years of data per entity.
        n_features: Number of features per sample.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (x_features, y_labels, entity_ids, years).
    """
    rng = np.random.default_rng(seed)
    n_samples = n_entities * years_per_entity

    x: NDArray[np.float64] = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    # Create binary labels using integers() which has proper typing
    y: NDArray[np.int64] = rng.integers(0, 2, size=n_samples, dtype=np.int64)

    # Create entity IDs and years
    entity_ids_list: list[int] = []
    years_list: list[int] = []
    base_year = 2015

    for entity_idx in range(n_entities):
        for year_offset in range(years_per_entity):
            entity_ids_list.append(entity_idx + 1)  # 1-indexed entity IDs
            years_list.append(base_year + year_offset)

    entity_ids: NDArray[np.int64] = np.array(entity_ids_list, dtype=np.int64)
    years: NDArray[np.int64] = np.array(years_list, dtype=np.int64)

    return x, y, entity_ids, years


class TestBuildSequences:
    """Tests for build_sequences function."""

    def test_build_sequences_basic(self) -> None:
        """Build sequences from panel data with multiple entities."""
        x, y, entity_ids, years = _make_entity_year_data(
            n_entities=5, years_per_entity=4, n_features=3
        )

        seq_data: SequenceData = build_sequences(
            x_features=x,
            y_labels=y,
            entity_ids=entity_ids,
            years=years,
            sequence_length=3,
        )

        # With 5 entities, 4 years each, seq_len=3: each entity produces 2 sequences
        # (years 1-3 and years 2-4)
        assert seq_data["n_sequences"] == 10  # 5 entities * 2 sequences
        assert seq_data["n_features"] == 3
        assert seq_data["sequence_length"] == 3
        assert seq_data["x_sequences"].shape == (10, 3, 3)
        assert seq_data["y_sequences"].shape == (10,)
        assert seq_data["sequence_entity_ids"].shape == (10,)

    def test_build_sequences_single_sequence_per_entity(self) -> None:
        """When years equals sequence_length, each entity produces one sequence."""
        x, y, entity_ids, years = _make_entity_year_data(
            n_entities=3, years_per_entity=3, n_features=2
        )

        seq_data: SequenceData = build_sequences(
            x_features=x,
            y_labels=y,
            entity_ids=entity_ids,
            years=years,
            sequence_length=3,
        )

        # Exactly 1 sequence per entity when years == seq_len
        assert seq_data["n_sequences"] == 3
        assert seq_data["x_sequences"].shape == (3, 3, 2)

    def test_build_sequences_labels_from_final_year(self) -> None:
        """Sequence labels come from the final year in each sequence."""
        # Create data where we know exactly what the labels should be
        n_entities = 2
        years_per_entity = 3
        n_features = 2
        n_samples = n_entities * years_per_entity

        x: NDArray[np.float64] = np.ones((n_samples, n_features), dtype=np.float64)
        # Entity 1: years 2015, 2016, 2017 -> labels 0, 0, 1
        # Entity 2: years 2015, 2016, 2017 -> labels 1, 1, 0
        y_list: list[int] = [0, 0, 1, 1, 1, 0]
        y: NDArray[np.int64] = np.array(y_list, dtype=np.int64)
        entity_list: list[int] = [1, 1, 1, 2, 2, 2]
        entity_ids: NDArray[np.int64] = np.array(entity_list, dtype=np.int64)
        years_list: list[int] = [2015, 2016, 2017, 2015, 2016, 2017]
        years: NDArray[np.int64] = np.array(years_list, dtype=np.int64)

        seq_data: SequenceData = build_sequences(
            x_features=x,
            y_labels=y,
            entity_ids=entity_ids,
            years=years,
            sequence_length=2,
        )

        # Each entity produces 2 sequences (years 1-2 and 2-3)
        # Entity 1: seq1 label=y[2016]=0, seq2 label=y[2017]=1
        # Entity 2: seq1 label=y[2016]=1, seq2 label=y[2017]=0
        assert seq_data["n_sequences"] == 4
        # Labels for each sequence (sorted by entity, then by start year)
        expected_labels: list[int] = [0, 1, 1, 0]
        y_seqs: NDArray[np.int64] = seq_data["y_sequences"]
        y_seqs_list: list[int] = y_seqs.tolist()
        for i, expected_y in enumerate(expected_labels):
            actual_val: int = y_seqs_list[i]
            assert actual_val == expected_y

    def test_build_sequences_entity_ids_tracked(self) -> None:
        """Sequence entity IDs correctly track source entity."""
        x, y, entity_ids, years = _make_entity_year_data(
            n_entities=3, years_per_entity=2, n_features=2
        )

        seq_data: SequenceData = build_sequences(
            x_features=x,
            y_labels=y,
            entity_ids=entity_ids,
            years=years,
            sequence_length=2,
        )

        # Each entity produces 1 sequence (years 1-2)
        assert seq_data["n_sequences"] == 3
        # Entity IDs should be 1, 2, 3
        seq_entity_ids: list[int] = seq_data["sequence_entity_ids"].tolist()
        assert sorted(seq_entity_ids) == [1, 2, 3]

    def test_build_sequences_skips_entities_with_insufficient_years(self) -> None:
        """Entities with fewer years than sequence_length are skipped."""
        # Create data with varying years per entity
        x: NDArray[np.float64] = np.ones((7, 2), dtype=np.float64)
        y: NDArray[np.int64] = np.zeros(7, dtype=np.int64)
        # Entity 1: 4 years, Entity 2: 2 years, Entity 3: 1 year
        entity_list: list[int] = [1, 1, 1, 1, 2, 2, 3]
        entity_ids: NDArray[np.int64] = np.array(entity_list, dtype=np.int64)
        years_list: list[int] = [2015, 2016, 2017, 2018, 2015, 2016, 2015]
        years: NDArray[np.int64] = np.array(years_list, dtype=np.int64)

        seq_data: SequenceData = build_sequences(
            x_features=x,
            y_labels=y,
            entity_ids=entity_ids,
            years=years,
            sequence_length=3,
        )

        # Only entity 1 has enough years (4 >= 3)
        # Entity 1 produces 2 sequences (years 1-3 and 2-4)
        assert seq_data["n_sequences"] == 2
        seq_entity_ids: list[int] = seq_data["sequence_entity_ids"].tolist()
        assert all(eid == 1 for eid in seq_entity_ids)

    def test_build_sequences_sequence_length_1(self) -> None:
        """Sequence length of 1 treats each sample as its own sequence."""
        x, y, entity_ids, years = _make_entity_year_data(
            n_entities=2, years_per_entity=3, n_features=2
        )

        seq_data: SequenceData = build_sequences(
            x_features=x,
            y_labels=y,
            entity_ids=entity_ids,
            years=years,
            sequence_length=1,
        )

        # Each sample becomes its own sequence
        assert seq_data["n_sequences"] == 6  # 2 entities * 3 years
        assert seq_data["x_sequences"].shape == (6, 1, 2)

    def test_build_sequences_raises_on_invalid_sequence_length(self) -> None:
        """Raises ValueError when sequence_length < 1."""
        x, y, entity_ids, years = _make_entity_year_data()

        with pytest.raises(ValueError, match="sequence_length must be >= 1"):
            build_sequences(
                x_features=x,
                y_labels=y,
                entity_ids=entity_ids,
                years=years,
                sequence_length=0,
            )

    def test_build_sequences_raises_when_no_valid_sequences(self) -> None:
        """Raises ValueError when no entities have enough years."""
        x, y, entity_ids, years = _make_entity_year_data(
            n_entities=2, years_per_entity=2, n_features=2
        )

        # Sequence length 3 but all entities have only 2 years
        with pytest.raises(ValueError, match="No valid sequences could be built"):
            build_sequences(
                x_features=x,
                y_labels=y,
                entity_ids=entity_ids,
                years=years,
                sequence_length=3,
            )

    def test_build_sequences_handles_unordered_years(self) -> None:
        """Years within an entity are sorted correctly even if input is unordered."""
        # Create 6 samples with 1 feature each
        x: NDArray[np.float64] = np.zeros((6, 1), dtype=np.float64)
        for i in range(6):
            x[i, 0] = float(i)
        y_list: list[int] = [0, 1, 0, 1, 0, 1]
        y: NDArray[np.int64] = np.array(y_list, dtype=np.int64)
        entity_list: list[int] = [1, 1, 1, 1, 1, 1]
        entity_ids: NDArray[np.int64] = np.array(entity_list, dtype=np.int64)
        # Unordered years
        years_list: list[int] = [2018, 2015, 2017, 2016, 2020, 2019]
        years: NDArray[np.int64] = np.array(years_list, dtype=np.int64)

        seq_data: SequenceData = build_sequences(
            x_features=x,
            y_labels=y,
            entity_ids=entity_ids,
            years=years,
            sequence_length=3,
        )

        # Years sorted: 2015, 2016, 2017, 2018, 2019, 2020
        # With seq_len=3, produces 4 sequences
        assert seq_data["n_sequences"] == 4


class TestReshapeFlatToPseudoSequences:
    """Tests for reshape_flat_to_pseudo_sequences function."""

    def test_reshape_basic(self) -> None:
        """Basic reshape when n_features is divisible by sequence_length."""
        # Create 2 samples with 12 features each
        x: NDArray[np.float64] = np.zeros((2, 12), dtype=np.float64)
        for i in range(24):
            x[i // 12, i % 12] = float(i)

        result = reshape_flat_to_pseudo_sequences(x, sequence_length=3)

        # 12 features / 3 steps = 4 features per step
        assert result.shape == (2, 3, 4)

    def test_reshape_with_padding(self) -> None:
        """Reshape pads features when not divisible by sequence_length."""
        # Create 4 samples with 5 features each
        x: NDArray[np.float64] = np.zeros((4, 5), dtype=np.float64)
        for i in range(20):
            x[i // 5, i % 5] = float(i)

        result = reshape_flat_to_pseudo_sequences(x, sequence_length=3)

        # 5 features -> needs padding to 6 (2 per step * 3 steps)
        assert result.shape == (4, 3, 2)

    def test_reshape_single_feature_per_step(self) -> None:
        """Reshape with sequence_length equal to n_features."""
        # Create 5 samples with 4 features each
        x: NDArray[np.float64] = np.zeros((5, 4), dtype=np.float64)
        for i in range(20):
            x[i // 4, i % 4] = float(i)

        result = reshape_flat_to_pseudo_sequences(x, sequence_length=4)

        # 4 features / 4 steps = 1 feature per step
        assert result.shape == (5, 4, 1)

    def test_reshape_more_steps_than_features(self) -> None:
        """Reshape when sequence_length > n_features (heavy padding)."""
        # Create 2 samples with 3 features each
        x: NDArray[np.float64] = np.zeros((2, 3), dtype=np.float64)
        for i in range(6):
            x[i // 3, i % 3] = float(i)

        result = reshape_flat_to_pseudo_sequences(x, sequence_length=5)

        # 3 features -> needs padding to 5 (1 per step * 5 steps)
        assert result.shape == (2, 5, 1)

    def test_reshape_preserves_values(self) -> None:
        """Original values are preserved in reshaped output."""
        row0: list[float] = [1.0, 2.0, 3.0, 4.0]
        row1: list[float] = [5.0, 6.0, 7.0, 8.0]
        x: NDArray[np.float64] = np.zeros((2, 4), dtype=np.float64)
        for i, val in enumerate(row0):
            x[0, i] = val
        for i, val in enumerate(row1):
            x[1, i] = val

        result: NDArray[np.float64] = reshape_flat_to_pseudo_sequences(x, sequence_length=2)

        # 4 features / 2 steps = 2 features per step
        # First sample: [[1, 2], [3, 4]]
        # Second sample: [[5, 6], [7, 8]]
        exp0: list[list[float]] = [[1.0, 2.0], [3.0, 4.0]]
        exp1: list[list[float]] = [[5.0, 6.0], [7.0, 8.0]]
        expected_sample_0: NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
        expected_sample_1: NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                expected_sample_0[i, j] = exp0[i][j]
                expected_sample_1[i, j] = exp1[i][j]

        # Extract slices to avoid Any from indexing
        result_0: NDArray[np.float64] = result[0:1, :, :].reshape((2, 2))
        result_1: NDArray[np.float64] = result[1:2, :, :].reshape((2, 2))
        np.testing.assert_array_almost_equal(result_0, expected_sample_0)
        np.testing.assert_array_almost_equal(result_1, expected_sample_1)

    def test_reshape_padding_is_zeros(self) -> None:
        """Padding consists of zeros."""
        row0: list[float] = [1.0, 2.0, 3.0]
        x: NDArray[np.float64] = np.zeros((1, 3), dtype=np.float64)
        for i, val in enumerate(row0):
            x[0, i] = val

        result: NDArray[np.float64] = reshape_flat_to_pseudo_sequences(x, sequence_length=2)

        # 3 features -> pads to 4 (2 per step * 2 steps)
        # Original: [1, 2, 3] -> Padded: [1, 2, 3, 0]
        # Reshaped: [[1, 2], [3, 0]]
        assert result.shape == (1, 2, 2)
        flat_result: list[float] = result.flatten().tolist()
        assert float(flat_result[3]) == 0.0  # Last position is padding


class TestExports:
    """Tests for module exports."""

    def test_sequence_data_exported(self) -> None:
        """SequenceData TypedDict is exported."""
        from covenant_ml.backends.lstm.sequences import __all__

        assert "SequenceData" in __all__

    def test_build_sequences_exported(self) -> None:
        """build_sequences function is exported."""
        from covenant_ml.backends.lstm.sequences import __all__

        assert "build_sequences" in __all__

    def test_reshape_flat_to_pseudo_sequences_exported(self) -> None:
        """reshape_flat_to_pseudo_sequences function is exported."""
        from covenant_ml.backends.lstm.sequences import __all__

        assert "reshape_flat_to_pseudo_sequences" in __all__
