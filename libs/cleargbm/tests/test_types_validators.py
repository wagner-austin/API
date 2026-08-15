"""Tests for cleargbm.types: scalar validators."""

from __future__ import annotations

import pytest

from cleargbm.types import (
    require_n_jobs,
    require_non_negative_float,
    require_non_negative_int,
    require_positive_float,
    require_positive_int,
    require_unit_float,
)

# =============================================================================
# Validation Helpers Tests
# =============================================================================


class TestRequirePositiveInt:
    """Tests for require_positive_int."""

    def test_accepts_positive(self) -> None:
        """Positive integers should pass."""
        assert require_positive_int(1, "x") == 1
        assert require_positive_int(100, "x") == 100

    def test_rejects_zero(self) -> None:
        """Zero should raise ValueError."""
        with pytest.raises(ValueError, match="x must be positive, got 0"):
            require_positive_int(0, "x")

    def test_rejects_negative(self) -> None:
        """Negative integers should raise ValueError."""
        with pytest.raises(ValueError, match="x must be positive, got -5"):
            require_positive_int(-5, "x")


class TestRequireNonNegativeInt:
    """Tests for require_non_negative_int."""

    def test_accepts_positive(self) -> None:
        """Positive integers should pass."""
        assert require_non_negative_int(1, "x") == 1

    def test_accepts_zero(self) -> None:
        """Zero should pass."""
        assert require_non_negative_int(0, "x") == 0

    def test_rejects_negative(self) -> None:
        """Negative integers should raise ValueError."""
        with pytest.raises(ValueError, match="x must be non-negative, got -1"):
            require_non_negative_int(-1, "x")


class TestRequirePositiveFloat:
    """Tests for require_positive_float."""

    def test_accepts_positive(self) -> None:
        """Positive floats should pass."""
        assert require_positive_float(0.1, "x") == 0.1
        assert require_positive_float(100.5, "x") == 100.5

    def test_rejects_zero(self) -> None:
        """Zero should raise ValueError."""
        with pytest.raises(ValueError, match=r"x must be positive, got 0\.0"):
            require_positive_float(0.0, "x")

    def test_rejects_negative(self) -> None:
        """Negative floats should raise ValueError."""
        with pytest.raises(ValueError, match=r"x must be positive, got -0\.5"):
            require_positive_float(-0.5, "x")


class TestRequireUnitFloat:
    """Tests for require_unit_float."""

    def test_accepts_in_range(self) -> None:
        """Values in (0, 1] should pass."""
        assert require_unit_float(0.5, "x") == 0.5
        assert require_unit_float(1.0, "x") == 1.0
        assert require_unit_float(0.001, "x") == 0.001

    def test_rejects_zero(self) -> None:
        """Zero should raise ValueError."""
        with pytest.raises(ValueError, match=r"x must be in \(0, 1\], got 0.0"):
            require_unit_float(0.0, "x")

    def test_rejects_greater_than_one(self) -> None:
        """Values > 1 should raise ValueError."""
        with pytest.raises(ValueError, match=r"x must be in \(0, 1\], got 1.5"):
            require_unit_float(1.5, "x")

    def test_rejects_negative(self) -> None:
        """Negative values should raise ValueError."""
        with pytest.raises(ValueError, match=r"x must be in \(0, 1\], got -0.1"):
            require_unit_float(-0.1, "x")


class TestRequireNonNegativeFloat:
    """Tests for require_non_negative_float."""

    def test_accepts_positive(self) -> None:
        """Positive floats should pass."""
        assert require_non_negative_float(0.5, "x") == 0.5

    def test_accepts_zero(self) -> None:
        """Zero should pass."""
        assert require_non_negative_float(0.0, "x") == 0.0

    def test_rejects_negative(self) -> None:
        """Negative floats should raise ValueError."""
        with pytest.raises(ValueError, match=r"x must be non-negative, got -0\.1"):
            require_non_negative_float(-0.1, "x")


class TestRequireNJobs:
    """Tests for require_n_jobs."""

    def test_accepts_positive(self) -> None:
        """Positive integers should pass."""
        assert require_n_jobs(1, "n_jobs") == 1
        assert require_n_jobs(4, "n_jobs") == 4
        assert require_n_jobs(100, "n_jobs") == 100

    def test_accepts_minus_one(self) -> None:
        """-1 should pass (use all cores)."""
        assert require_n_jobs(-1, "n_jobs") == -1

    def test_rejects_zero(self) -> None:
        """Zero should raise ValueError."""
        with pytest.raises(ValueError, match="n_jobs must be -1 or positive, got 0"):
            require_n_jobs(0, "n_jobs")

    def test_rejects_negative_other_than_minus_one(self) -> None:
        """Negative values other than -1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_jobs must be -1 or positive, got -2"):
            require_n_jobs(-2, "n_jobs")
        with pytest.raises(ValueError, match="n_jobs must be -1 or positive, got -5"):
            require_n_jobs(-5, "n_jobs")
