"""Tests for target column detection functions."""

from __future__ import annotations

from scripts.discover_datasets.detection import (
    calculate_positive_ratio,
    detect_positive_negative_values,
    find_exclude_columns,
    find_target_candidates,
    is_binary_column,
    recommend_target,
)
from scripts.discover_datasets.types import TargetColumnCandidate


class TestIsBinaryColumn:
    """Tests for is_binary_column function."""

    def test_binary_0_1(self) -> None:
        """Test binary column with 0 and 1 values."""
        assert is_binary_column(("0", "1")) is True

    def test_binary_1_0(self) -> None:
        """Test binary column with 1 and 0 values (reversed)."""
        assert is_binary_column(("1", "0")) is True

    def test_binary_float_0_1(self) -> None:
        """Test binary column with float 0.0 and 1.0 values."""
        assert is_binary_column(("0.0", "1.0")) is True

    def test_binary_yes_no(self) -> None:
        """Test binary column with yes and no values."""
        assert is_binary_column(("yes", "no")) is True

    def test_binary_yes_no_uppercase(self) -> None:
        """Test binary column with YES and NO values."""
        assert is_binary_column(("YES", "NO")) is True

    def test_binary_true_false(self) -> None:
        """Test binary column with true and false values."""
        assert is_binary_column(("true", "false")) is True

    def test_binary_positive_negative(self) -> None:
        """Test binary column with positive and negative values."""
        assert is_binary_column(("positive", "negative")) is True

    def test_binary_alive_failed(self) -> None:
        """Test binary column with alive and failed values."""
        assert is_binary_column(("alive", "failed")) is True

    def test_binary_existing_attrited(self) -> None:
        """Test binary column with Existing Customer and Attrited Customer."""
        assert is_binary_column(("Existing Customer", "Attrited Customer")) is True

    def test_binary_good_bad(self) -> None:
        """Test binary column with good and bad values."""
        assert is_binary_column(("good", "bad")) is True

    def test_binary_pass_fail(self) -> None:
        """Test binary column with pass and fail values."""
        assert is_binary_column(("pass", "fail")) is True

    def test_binary_approved_rejected(self) -> None:
        """Test binary column with approved and rejected values."""
        assert is_binary_column(("approved", "rejected")) is True

    def test_binary_y_n(self) -> None:
        """Test binary column with Y and N values."""
        assert is_binary_column(("Y", "N")) is True

    def test_not_binary_three_values(self) -> None:
        """Test non-binary column with three values."""
        assert is_binary_column(("a", "b", "c")) is False

    def test_not_binary_single_value(self) -> None:
        """Test non-binary column with single value."""
        assert is_binary_column(("a",)) is False

    def test_binary_any_two_values(self) -> None:
        """Test that any two-value column is treated as binary candidate."""
        # Any 2-value column is potentially binary for classification
        assert is_binary_column(("foo", "bar")) is True
        assert is_binary_column(("ClassA", "ClassB")) is True


class TestFindTargetCandidates:
    """Tests for find_target_candidates function."""

    def test_finds_target_column(self) -> None:
        """Test finding target column candidate."""
        columns = ("feature1", "feature2", "target")
        sample_rows = (("1", "2", "0"), ("3", "4", "1"))

        candidates = find_target_candidates(columns, sample_rows)

        assert len(candidates) == 1
        assert candidates[0]["column_name"] == "target"
        assert candidates[0]["is_binary"] is True

    def test_finds_class_column(self) -> None:
        """Test finding class column candidate."""
        columns = ("x", "class")
        sample_rows = (("1", "A"), ("2", "B"))

        candidates = find_target_candidates(columns, sample_rows)

        assert len(candidates) == 1
        assert candidates[0]["column_name"] == "class"

    def test_finds_multiple_candidates(self) -> None:
        """Test finding multiple target candidates."""
        columns = ("target", "label", "class")
        sample_rows = (("0", "a", "x"), ("1", "b", "y"))

        candidates = find_target_candidates(columns, sample_rows)

        assert len(candidates) == 3

    def test_no_candidates(self) -> None:
        """Test when no target candidates found."""
        columns = ("feature1", "feature2", "feature3")
        sample_rows = (("1", "2", "3"),)

        candidates = find_target_candidates(columns, sample_rows)

        assert len(candidates) == 0

    def test_handles_empty_sample(self) -> None:
        """Test handling empty sample rows."""
        columns = ("target",)
        sample_rows: tuple[tuple[str, ...], ...] = ()

        candidates = find_target_candidates(columns, sample_rows)

        assert len(candidates) == 1
        assert candidates[0]["n_unique"] == 0

    def test_handles_short_rows(self) -> None:
        """Test handling rows shorter than column index."""
        columns = ("a", "b", "target")
        sample_rows = (("1", "2"), ("3", "4"))

        candidates = find_target_candidates(columns, sample_rows)

        assert len(candidates) == 1
        assert candidates[0]["column_name"] == "target"
        assert candidates[0]["n_unique"] == 0

    def test_matches_raw_lowercase_pattern(self) -> None:
        """Test matching patterns via raw lowercase when normalization fails.

        Patterns like "seriousdlqin2yrs" should match columns with matching
        raw lowercase form, even if the normalized form differs.
        """
        # Column "SeriousDlqin2Yrs" normalizes to "serious_dlqin2_yrs"
        # but raw lowercase "seriousdlqin2yrs" should still match
        columns = ("feature1", "SeriousDlqin2Yrs")
        sample_rows = (("1", "0"), ("2", "1"))

        candidates = find_target_candidates(columns, sample_rows)

        assert len(candidates) == 1
        assert candidates[0]["column_name"] == "SeriousDlqin2Yrs"
        assert candidates[0]["is_binary"] is True


class TestFindExcludeColumns:
    """Tests for find_exclude_columns function."""

    def test_finds_id_column(self) -> None:
        """Test finding ID column to exclude."""
        columns = ("id", "feature1", "target")

        excludes = find_exclude_columns(columns)

        assert "id" in excludes

    def test_finds_customer_id(self) -> None:
        """Test finding customer_id column to exclude."""
        columns = ("customer_id", "amount", "target")

        excludes = find_exclude_columns(columns)

        assert "customer_id" in excludes

    def test_finds_name_column(self) -> None:
        """Test finding name column to exclude."""
        columns = ("company_name", "revenue", "status")

        excludes = find_exclude_columns(columns)

        assert "company_name" in excludes

    def test_finds_date_column(self) -> None:
        """Test finding date column to exclude."""
        columns = ("transaction_date", "amount")

        excludes = find_exclude_columns(columns)

        assert "transaction_date" in excludes

    def test_no_excludes(self) -> None:
        """Test when no columns to exclude."""
        columns = ("feature1", "feature2", "target")

        excludes = find_exclude_columns(columns)

        assert len(excludes) == 0

    def test_finds_unnamed_column(self) -> None:
        """Test finding unnamed column to exclude."""
        columns = ("Unnamed: 0", "feature1")

        excludes = find_exclude_columns(columns)

        assert "Unnamed: 0" in excludes


class TestRecommendTarget:
    """Tests for recommend_target function."""

    def test_empty_candidates(self) -> None:
        """Test with no candidates."""
        candidates: tuple[TargetColumnCandidate, ...] = ()

        result = recommend_target(candidates)

        assert result == ""

    def test_prefers_binary(self) -> None:
        """Test preferring binary column."""
        candidates: tuple[TargetColumnCandidate, ...] = (
            {
                "column_name": "class",
                "unique_values": ("a", "b", "c"),
                "n_unique": 3,
                "is_binary": False,
            },
            {
                "column_name": "target",
                "unique_values": ("0", "1"),
                "n_unique": 2,
                "is_binary": True,
            },
        )

        result = recommend_target(candidates)

        assert result == "target"

    def test_prefers_few_unique(self) -> None:
        """Test preferring column with few unique values when no binary."""
        candidates: tuple[TargetColumnCandidate, ...] = (
            {
                "column_name": "many",
                "unique_values": tuple(str(i) for i in range(10)),
                "n_unique": 10,
                "is_binary": False,
            },
            {
                "column_name": "few",
                "unique_values": ("a", "b", "c"),
                "n_unique": 3,
                "is_binary": False,
            },
        )

        result = recommend_target(candidates)

        assert result == "few"

    def test_fallback_to_first(self) -> None:
        """Test fallback to first candidate."""
        candidates: tuple[TargetColumnCandidate, ...] = (
            {
                "column_name": "first",
                "unique_values": tuple(str(i) for i in range(10)),
                "n_unique": 10,
                "is_binary": False,
            },
        )

        result = recommend_target(candidates)

        assert result == "first"


class TestDetectPositiveNegativeValues:
    """Tests for detect_positive_negative_values function."""

    def test_non_binary_returns_empty(self) -> None:
        """Test non-binary values (not 2 values) returns empty strings."""
        result = detect_positive_negative_values(("a", "b", "c"))
        assert result == ("", "", "binary_int")

    def test_single_value_returns_empty(self) -> None:
        """Test single value returns empty strings."""
        result = detect_positive_negative_values(("a",))
        assert result == ("", "", "binary_int")

    def test_numeric_0_1(self) -> None:
        """Test numeric 0/1 detection."""
        result = detect_positive_negative_values(("0", "1"))
        assert result == ("1", "0", "binary_int")

    def test_numeric_1_0(self) -> None:
        """Test numeric 1/0 detection (reversed order)."""
        result = detect_positive_negative_values(("1", "0"))
        assert result == ("1", "0", "binary_int")

    def test_yes_no(self) -> None:
        """Test yes/no detection."""
        result = detect_positive_negative_values(("yes", "no"))
        assert result == ("yes", "no", "binary_str")

    def test_no_yes(self) -> None:
        """Test no/yes detection (a is negative, b is positive)."""
        result = detect_positive_negative_values(("no", "yes"))
        assert result == ("yes", "no", "binary_str")

    def test_failed_alive(self) -> None:
        """Test failed/alive detection."""
        result = detect_positive_negative_values(("failed", "alive"))
        assert result == ("failed", "alive", "binary_str")

    def test_alive_failed(self) -> None:
        """Test alive/failed detection (a is negative, b is positive)."""
        result = detect_positive_negative_values(("alive", "failed"))
        assert result == ("failed", "alive", "binary_str")

    def test_good_bad(self) -> None:
        """Test good/bad detection (a is negative)."""
        result = detect_positive_negative_values(("good", "bad"))
        assert result == ("bad", "good", "binary_str")

    def test_a_negative_b_unknown(self) -> None:
        """Test case where a is negative and b is unknown (not positive).

        This hits the 'a is negative' branch without 'b is positive' matching first.
        """
        result = detect_positive_negative_values(("good", "unknown"))
        assert result == ("unknown", "good", "binary_str")

    def test_a_unknown_b_negative(self) -> None:
        """Test case where a is unknown and b is negative.

        This hits the 'b is negative' branch.
        """
        result = detect_positive_negative_values(("unknown", "good"))
        assert result == ("unknown", "good", "binary_str")

    def test_unknown_values_default(self) -> None:
        """Test unknown values default to first as positive."""
        result = detect_positive_negative_values(("ClassA", "ClassB"))
        assert result == ("ClassA", "ClassB", "binary_str")

    def test_attrited_existing_customer(self) -> None:
        """Test attrited/existing customer detection."""
        result = detect_positive_negative_values(("Attrited Customer", "Existing Customer"))
        assert result == ("Attrited Customer", "Existing Customer", "binary_str")


class TestCalculatePositiveRatio:
    """Tests for calculate_positive_ratio function."""

    def test_empty_target_returns_zero(self) -> None:
        """Test empty target column returns 0.0."""
        result = calculate_positive_ratio((("1", "a"),), ("target", "feature"), "", "1")
        assert result == 0.0

    def test_empty_positive_value_returns_zero(self) -> None:
        """Test empty positive value returns 0.0."""
        result = calculate_positive_ratio((("1", "a"),), ("target", "feature"), "target", "")
        assert result == 0.0

    def test_empty_sample_rows_returns_zero(self) -> None:
        """Test empty sample rows returns 0.0."""
        result = calculate_positive_ratio((), ("target", "feature"), "target", "1")
        assert result == 0.0

    def test_target_column_not_found_returns_zero(self) -> None:
        """Test target column not in columns returns 0.0."""
        result = calculate_positive_ratio((("1", "a"),), ("other", "feature"), "target", "1")
        assert result == 0.0

    def test_calculates_ratio_correctly(self) -> None:
        """Test correct ratio calculation."""
        sample_rows = (
            ("1", "a"),
            ("0", "b"),
            ("1", "c"),
            ("1", "d"),
            ("0", "e"),
        )
        result = calculate_positive_ratio(sample_rows, ("target", "feature"), "target", "1")
        assert result == 0.6  # 3 out of 5

    def test_all_positive(self) -> None:
        """Test 100% positive ratio."""
        sample_rows = (("1", "a"), ("1", "b"), ("1", "c"))
        result = calculate_positive_ratio(sample_rows, ("target",), "target", "1")
        assert result == 1.0

    def test_all_negative(self) -> None:
        """Test 0% positive ratio."""
        sample_rows = (("0", "a"), ("0", "b"), ("0", "c"))
        result = calculate_positive_ratio(sample_rows, ("target",), "target", "1")
        assert result == 0.0

    def test_short_rows_skipped(self) -> None:
        """Test rows shorter than target index are skipped."""
        sample_rows = (
            ("a",),  # Too short, target at index 1
            ("a", "1"),
            ("a", "0"),
        )
        result = calculate_positive_ratio(sample_rows, ("feature", "target"), "target", "1")
        assert result == 0.5  # 1 out of 2 valid rows

    def test_all_rows_too_short_returns_zero(self) -> None:
        """Test all rows too short returns 0.0."""
        sample_rows = (
            ("a",),  # Too short, target at index 1
            ("b",),  # Too short
        )
        result = calculate_positive_ratio(sample_rows, ("feature", "target"), "target", "1")
        assert result == 0.0
