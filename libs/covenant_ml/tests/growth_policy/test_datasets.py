"""Tests for the dataset loaders and the group-disjoint partition.

The partition test is a regression lock rather than a shape check. The recorded
figures were produced by a specific permutation, so this module reimplements
that original expression and asserts the package reproduces it index for index.
If the two ever part, every number in the write-up stops following from this
code, and that must fail loudly here rather than silently at the next re-run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.growth_policy.datasets import (
    TRAIN_FRACTION,
    company_disjoint_indices,
    describe_dataset,
    encode_column,
    load_bankruptcy,
    load_german_credit,
    load_taiwan_bankruptcy,
    sorted_group_codes,
)
from covenant_ml.growth_policy.types import (
    ERR_EMPTY_DATASET,
    ERR_EMPTY_SPLIT,
    ERR_MISSING_COLUMN,
    ERR_MISSING_VALUE,
    ERR_RAGGED_ROWS,
)

from .factories import (
    ROWS_PER_COMPANY,
    make_separable_dataset,
    write_bankruptcy_csv,
    write_german_data,
    write_taiwan_csv,
)
from .numeric import as_float_list, as_int_list, ints, label_mask, mean_of, select


def _original_split(companies: list[str], seed: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Partition rows exactly as the recorded experiment script did.

    Kept verbatim so the package's partition can be compared against the
    expression that produced the published numbers.

    Args:
        companies: Company name per row.
        seed: Seed controlling the permutation.

    Returns:
        Train and test row indices.
    """
    unique: list[str] = sorted(set(companies))
    rng = np.random.default_rng(seed)
    permutation: NDArray[np.int64] = rng.permutation(len(unique))
    cut = int(TRAIN_FRACTION * len(unique))
    chosen: list[int] = as_int_list(permutation)[:cut]
    train_companies = {unique[index] for index in chosen}
    indices: NDArray[np.int64] = np.arange(len(companies), dtype=np.int64)
    flags: list[bool] = [company in train_companies for company in companies]
    mask: NDArray[np.bool_] = np.asarray(flags, dtype=np.bool_)
    train_index: NDArray[np.int64] = indices[mask]
    test_index: NDArray[np.int64] = indices[~mask]
    return train_index, test_index


class TestSortedGroupCodes:
    """Codes are assigned in sorted-name order."""

    def test_assigns_codes_by_sorted_name(self) -> None:
        """The alphabetically first name should take code zero."""
        codes, count = sorted_group_codes(["b", "a", "b"])

        assert as_int_list(codes) == [1, 0, 1]
        assert count == 2

    def test_counts_distinct_groups(self) -> None:
        """Repeated names should count once."""
        _, count = sorted_group_codes(["x", "x", "x"])

        assert count == 1


class TestCompanyDisjointIndices:
    """The partition keeps groups whole and reproduces the recorded expression."""

    @pytest.mark.parametrize("seed", [42, 43, 44, 7, 1234])
    def test_matches_the_original_expression(self, seed: int) -> None:
        """The package partition must equal the recorded script's, index for index."""
        rng = np.random.default_rng(0)
        companies = [f"C{int(rng.integers(0, 60))}" for _ in range(500)]

        expected_train, expected_test = _original_split(companies, seed)
        train, test = company_disjoint_indices(companies, seed)

        assert as_int_list(train) == as_int_list(expected_train)
        assert as_int_list(test) == as_int_list(expected_test)

    def test_no_group_appears_on_both_sides(self) -> None:
        """A company's rows must land entirely in one fold."""
        companies = [f"C{index % 10}" for index in range(80)]

        train, test = company_disjoint_indices(companies, 42)

        train_names = {companies[index] for index in as_int_list(train)}
        test_names = {companies[index] for index in as_int_list(test)}
        assert train_names.isdisjoint(test_names)

    def test_covers_every_row_exactly_once(self) -> None:
        """The two folds must partition the rows, not sample them."""
        companies = [f"C{index % 10}" for index in range(80)]

        train, test = company_disjoint_indices(companies, 42)

        assert sorted(as_int_list(train) + as_int_list(test)) == list(range(80))

    def test_rejects_a_partition_that_empties_a_side(self) -> None:
        """A single group cannot be split, so it must fail rather than return empty."""
        with pytest.raises(ValueError, match=ERR_EMPTY_SPLIT):
            company_disjoint_indices(["only"] * 5, 42)


class TestDescribeDataset:
    """The report header's dataset summary."""

    def test_reports_shape_and_positive_rate(self) -> None:
        """Row count, feature count and positive rate should come from the arrays."""
        features: NDArray[np.float64] = np.zeros((4, 3), dtype=np.float64)
        labels: NDArray[np.int64] = ints([1, 0, 1, 0])

        info = describe_dataset("synthetic", features, labels)

        assert info["row_count"] == 4
        assert info["feature_count"] == 3
        assert info["positive_rate"] == 0.5

    def test_rejects_an_empty_dataset(self) -> None:
        """A dataset with no rows leaves the positive rate undefined."""
        with pytest.raises(ValueError, match=ERR_EMPTY_DATASET):
            empty_features: NDArray[np.float64] = np.zeros((0, 3), dtype=np.float64)
            empty_labels: NDArray[np.int64] = np.zeros((0,), dtype=np.int64)
            describe_dataset("empty", empty_features, empty_labels)


class TestEncodeColumn:
    """Numeric passthrough versus ordinal encoding."""

    def test_passes_numeric_values_through(self) -> None:
        """A wholly numeric column should keep its magnitudes."""
        assert encode_column(["1", "2.5", "-3", "4e2"]) == [1.0, 2.5, -3.0, 400.0]

    def test_ordinal_encodes_a_categorical_column(self) -> None:
        """Codes should follow the sorted value set."""
        assert encode_column(["A12", "A11", "A12"]) == [1.0, 0.0, 1.0]

    def test_one_non_numeric_value_makes_the_column_categorical(self) -> None:
        """A mixed column must not be read half as magnitudes and half as codes."""
        assert encode_column(["1", "2", "unknown"]) == [0.0, 1.0, 2.0]


class TestLoadBankruptcy:
    """Loading the American-bankruptcy layout."""

    def test_loads_features_labels_and_groups(self, tmp_path: Path) -> None:
        """Shapes and the alive/failed mapping should follow the file."""
        target = tmp_path / "american_bankruptcy.csv"
        write_bankruptcy_csv(target, company_count=6)

        dataset = load_bankruptcy(target)

        assert dataset.features.shape == (6 * ROWS_PER_COMPANY, 18)
        assert len(dataset.groups) == 6 * ROWS_PER_COMPANY
        assert set(as_int_list(dataset.labels)) == {0, 1}

    def test_maps_alive_to_zero(self, tmp_path: Path) -> None:
        """Only the literal alive status should score zero."""
        target = tmp_path / "american_bankruptcy.csv"
        header = ["company_name", "status_label"] + [f"X{i}" for i in range(1, 19)]
        values = ",".join(["0.0"] * 18)
        target.write_text(
            ",".join(header) + f"\nA,alive,{values}\nB,failed,{values}\n",
            encoding="utf-8",
        )

        dataset = load_bankruptcy(target)

        assert as_int_list(dataset.labels) == [0, 1]

    def test_rejects_a_missing_column(self, tmp_path: Path) -> None:
        """A file missing a feature column should be refused."""
        target = tmp_path / "american_bankruptcy.csv"
        target.write_text("company_name,status_label\nA,alive\n", encoding="utf-8")

        with pytest.raises(ValueError, match=ERR_MISSING_COLUMN):
            load_bankruptcy(target)

    def test_rejects_a_header_with_no_rows(self, tmp_path: Path) -> None:
        """A header-only file should be refused as an empty dataset."""
        target = tmp_path / "american_bankruptcy.csv"
        header = ["company_name", "status_label"] + [f"X{i}" for i in range(1, 19)]
        target.write_text(",".join(header) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match=ERR_EMPTY_DATASET):
            load_bankruptcy(target)


class TestLoadTaiwanBankruptcy:
    """Loading the Taiwan-bankruptcy layout."""

    def test_takes_the_first_column_as_the_label(self, tmp_path: Path) -> None:
        """Features should exclude the label column."""
        target = tmp_path / "data.csv"
        write_taiwan_csv(target, row_count=10, feature_count=5)

        dataset = load_taiwan_bankruptcy(target)

        assert dataset.features.shape == (10, 5)
        assert as_int_list(dataset.labels) == [index % 2 for index in range(10)]

    def test_rejects_a_short_row(self, tmp_path: Path) -> None:
        """A truncated row is padded with null by the reader and must be refused.

        This is the dangerous direction: polars does not raise on a short line,
        it fills the missing cell, so without this check a truncated file would
        reach a learner as a NaN feature and produce a plausible wrong number.
        """
        target = tmp_path / "data.csv"
        target.write_text("y,a,b\n1,2,3\n0,4\n", encoding="utf-8")

        with pytest.raises(ValueError, match=ERR_MISSING_VALUE):
            load_taiwan_bankruptcy(target)

    def test_rejects_an_empty_cell(self, tmp_path: Path) -> None:
        """A blank field is absent data, not a zero."""
        target = tmp_path / "data.csv"
        target.write_text("y,a,b\n1,2,3\n0,,5\n", encoding="utf-8")

        with pytest.raises(ValueError, match=ERR_MISSING_VALUE):
            load_taiwan_bankruptcy(target)

    def test_rejects_a_header_with_no_rows(self, tmp_path: Path) -> None:
        """A header-only file should be refused."""
        target = tmp_path / "data.csv"
        target.write_text("y,a,b\n", encoding="utf-8")

        with pytest.raises(ValueError, match=ERR_EMPTY_DATASET):
            load_taiwan_bankruptcy(target)


class TestLoadGermanCredit:
    """Loading the German-credit layout."""

    def test_encodes_categoricals_and_maps_the_label(self, tmp_path: Path) -> None:
        """Bad credit should become the positive class."""
        target = tmp_path / "german.data"
        write_german_data(target, row_count=10)

        dataset = load_german_credit(target)

        assert dataset.features.shape == (10, 3)
        assert as_int_list(dataset.labels) == [1 if row % 2 == 0 else 0 for row in range(10)]

    def test_ordinal_encodes_the_categorical_column(self, tmp_path: Path) -> None:
        """The categorical column should become codes over its sorted value set."""
        target = tmp_path / "german.data"
        target.write_text("A12 1 2\nA11 3 4\n", encoding="utf-8")

        dataset = load_german_credit(target)

        first_column: NDArray[np.float64] = dataset.features[:, 0]
        assert as_float_list(first_column) == [1.0, 0.0]

    def test_rejects_ragged_rows(self, tmp_path: Path) -> None:
        """A short row must be refused rather than silently misaligned."""
        target = tmp_path / "german.data"
        target.write_text("A11 1 2 1\nA12 3 1\n", encoding="utf-8")

        with pytest.raises(ValueError, match=ERR_RAGGED_ROWS):
            load_german_credit(target)

    def test_rejects_an_empty_file(self, tmp_path: Path) -> None:
        """A file with no rows should be refused."""
        target = tmp_path / "german.data"
        target.write_text("\n\n", encoding="utf-8")

        with pytest.raises(ValueError, match=ERR_EMPTY_DATASET):
            load_german_credit(target)


class TestSeparableFixture:
    """The shared fixture really is learnable, so quality tests mean something."""

    def test_labels_track_the_first_feature(self) -> None:
        """A model that reads feature zero should be able to score above chance."""
        features, labels = make_separable_dataset(row_count=50, feature_count=3)

        first: NDArray[np.float64] = features[:, 0]
        positives = select(first, label_mask(labels, 1))
        negatives = select(first, label_mask(labels, 0))
        assert mean_of(positives) > mean_of(negatives)
