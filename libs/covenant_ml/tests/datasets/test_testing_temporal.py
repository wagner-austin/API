"""Tests for create_synthetic_daily_timeseries in testing module."""

from __future__ import annotations

import numpy as np
import pytest

from covenant_ml.datasets.testing import (
    create_synthetic_daily_timeseries,
    create_synthetic_trending_metrics,
)

# =============================================================================
# Validation tests
# =============================================================================


def test_create_synthetic_raises_on_zero_years() -> None:
    """Raises ValueError when n_years < 1."""
    with pytest.raises(ValueError, match="n_years must be >= 1"):
        create_synthetic_daily_timeseries(
            n_years=0,
            n_locations=1,
            n_harmonics=1,
            seed=42,
        )


def test_create_synthetic_raises_on_zero_locations() -> None:
    """Raises ValueError when n_locations < 1."""
    with pytest.raises(ValueError, match="n_locations must be >= 1"):
        create_synthetic_daily_timeseries(
            n_years=1,
            n_locations=0,
            n_harmonics=1,
            seed=42,
        )


def test_create_synthetic_raises_on_zero_harmonics() -> None:
    """Raises ValueError when n_harmonics < 1."""
    with pytest.raises(ValueError, match="n_harmonics must be >= 1"):
        create_synthetic_daily_timeseries(
            n_years=1,
            n_locations=1,
            n_harmonics=0,
            seed=42,
        )


# =============================================================================
# Structure tests
# =============================================================================


def test_create_synthetic_correct_shapes() -> None:
    """daily_values, day_of_year, year_labels have correct shapes."""
    data = create_synthetic_daily_timeseries(
        n_years=3,
        n_locations=2,
        n_harmonics=2,
        seed=42,
    )

    assert data["daily_values"].shape == (3 * 365, 2)
    assert data["day_of_year"].shape == (3 * 365,)
    assert data["month_labels"].shape == (3 * 365,)
    assert data["year_labels"].shape == (3 * 365,)


def test_create_synthetic_correct_dtypes() -> None:
    """Arrays have correct dtypes."""
    data = create_synthetic_daily_timeseries(
        n_years=2,
        n_locations=1,
        n_harmonics=1,
        seed=42,
    )

    assert data["daily_values"].dtype == np.float64
    assert data["day_of_year"].dtype == np.int64
    assert data["month_labels"].dtype == np.int64
    assert data["year_labels"].dtype == np.int64


def test_create_synthetic_day_of_year_range() -> None:
    """Day-of-year values are in 1-365 range."""
    data = create_synthetic_daily_timeseries(
        n_years=2,
        n_locations=1,
        n_harmonics=1,
        seed=42,
    )

    assert int(np.min(data["day_of_year"])) == 1
    assert int(np.max(data["day_of_year"])) == 365


def test_create_synthetic_month_labels_follow_the_calendar() -> None:
    """Month labels place each day-of-year in its non-leap calendar month.

    The season restriction is driven entirely by these labels, so an
    off-by-one here would shift which days the tail thresholds describe.
    """
    data = create_synthetic_daily_timeseries(
        n_years=1,
        n_locations=1,
        n_harmonics=1,
        seed=42,
    )
    months = data["month_labels"]

    # Boundaries of each month on a 365-day calendar, as (day_of_year, month).
    for day, expected in (
        (1, 1),
        (31, 1),
        (32, 2),
        (59, 2),
        (60, 3),
        (152, 6),
        (181, 6),
        (182, 7),
        (213, 8),
        (244, 9),
        (334, 11),
        (335, 12),
        (365, 12),
    ):
        assert int(months.flat[day - 1]) == expected, f"day {day}"


def test_create_synthetic_month_labels_cover_every_month() -> None:
    """All twelve months appear, with the right number of days each."""
    data = create_synthetic_daily_timeseries(
        n_years=1,
        n_locations=1,
        n_harmonics=1,
        seed=42,
    )
    months = data["month_labels"]

    counts = [
        sum(1 for i in range(int(months.shape[0])) if int(months.flat[i]) == month)
        for month in range(1, 13)
    ]
    assert counts == [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    assert sum(counts) == 365


def test_create_synthetic_month_labels_repeat_each_year() -> None:
    """Every year carries the same month pattern, since no year is a leap year."""
    data = create_synthetic_daily_timeseries(
        n_years=3,
        n_locations=1,
        n_harmonics=1,
        seed=42,
    )
    months = data["month_labels"]

    np.testing.assert_array_equal(months[0:365], months[365:730])
    np.testing.assert_array_equal(months[365:730], months[730:1095])


def test_create_synthetic_year_labels() -> None:
    """Year labels start at 2000 and increment."""
    data = create_synthetic_daily_timeseries(
        n_years=3,
        n_locations=1,
        n_harmonics=1,
        seed=42,
    )

    unique_years = np.unique(data["year_labels"])
    years_list = [int(unique_years.flat[i]) for i in range(3)]
    assert years_list == [2000, 2001, 2002]


def test_create_synthetic_metadata() -> None:
    """Metadata fields match construction parameters."""
    data = create_synthetic_daily_timeseries(
        n_years=3,
        n_locations=2,
        n_harmonics=3,
        seed=42,
        noise_std=0.5,
    )

    assert data["n_years"] == 3
    assert data["n_locations"] == 2
    assert data["n_harmonics"] == 3
    assert data["noise_std"] == 0.5


# =============================================================================
# Coefficient structure tests
# =============================================================================


def test_create_synthetic_coefficient_shapes() -> None:
    """Ground-truth coefficients have correct shapes."""
    data = create_synthetic_daily_timeseries(
        n_years=2,
        n_locations=3,
        n_harmonics=2,
        seed=42,
    )

    assert len(data["true_cos_coefficients"]) == 2  # n_harmonics
    assert len(data["true_cos_coefficients"][0]) == 3  # n_locations
    assert len(data["true_sin_coefficients"]) == 2
    assert len(data["true_sin_coefficients"][0]) == 3
    assert len(data["true_mean"]) == 3


def test_create_synthetic_location_scaling() -> None:
    """Location j has coefficients scaled by (1 + 0.1 * j)."""
    data = create_synthetic_daily_timeseries(
        n_years=2,
        n_locations=3,
        n_harmonics=1,
        seed=42,
    )

    cos_0 = data["true_cos_coefficients"][0][0]
    cos_1 = data["true_cos_coefficients"][0][1]
    cos_2 = data["true_cos_coefficients"][0][2]

    # cos_1 / cos_0 should be 1.1 / 1.0 = 1.1
    assert abs(cos_1 / cos_0 - 1.1) < 1e-10
    # cos_2 / cos_0 should be 1.2 / 1.0 = 1.2
    assert abs(cos_2 / cos_0 - 1.2) < 1e-10


def test_create_synthetic_mean_offset() -> None:
    """Location j has mean = base_mean + 2.0 * j."""
    data = create_synthetic_daily_timeseries(
        n_years=2,
        n_locations=3,
        n_harmonics=1,
        seed=42,
        mean_value=20.0,
    )

    assert abs(data["true_mean"][0] - 20.0) < 1e-10
    assert abs(data["true_mean"][1] - 22.0) < 1e-10
    assert abs(data["true_mean"][2] - 24.0) < 1e-10


# =============================================================================
# Determinism tests
# =============================================================================


def test_create_synthetic_deterministic() -> None:
    """Same seed produces identical output."""
    data1 = create_synthetic_daily_timeseries(
        n_years=2,
        n_locations=2,
        n_harmonics=2,
        seed=42,
    )
    data2 = create_synthetic_daily_timeseries(
        n_years=2,
        n_locations=2,
        n_harmonics=2,
        seed=42,
    )

    np.testing.assert_array_equal(data1["daily_values"], data2["daily_values"])
    np.testing.assert_array_equal(data1["day_of_year"], data2["day_of_year"])
    np.testing.assert_array_equal(data1["year_labels"], data2["year_labels"])


def test_create_synthetic_different_seed() -> None:
    """Different seeds produce different output."""
    data1 = create_synthetic_daily_timeseries(
        n_years=2,
        n_locations=1,
        n_harmonics=1,
        seed=42,
    )
    data2 = create_synthetic_daily_timeseries(
        n_years=2,
        n_locations=1,
        n_harmonics=1,
        seed=99,
    )

    assert not np.allclose(data1["daily_values"], data2["daily_values"])


def test_create_synthetic_zero_noise_exact_seasonal() -> None:
    """With noise_std=0, daily_values match the deterministic seasonal cycle."""
    data1 = create_synthetic_daily_timeseries(
        n_years=2,
        n_locations=2,
        n_harmonics=2,
        seed=42,
        noise_std=0.0,
    )
    data2 = create_synthetic_daily_timeseries(
        n_years=2,
        n_locations=2,
        n_harmonics=2,
        seed=99,
        noise_std=0.0,
    )

    # With zero noise, seed shouldn't matter
    np.testing.assert_array_equal(data1["daily_values"], data2["daily_values"])


# =============================================================================
# create_synthetic_trending_metrics tests
# =============================================================================


def test_trending_raises_on_too_few_years() -> None:
    """Raises ValueError when n_years < 3."""
    with pytest.raises(ValueError, match="n_years must be >= 3"):
        create_synthetic_trending_metrics(
            n_years=2,
            n_locations=3,
            seed=42,
        )


def test_trending_raises_on_zero_locations() -> None:
    """Raises ValueError when n_locations < 1."""
    with pytest.raises(ValueError, match="n_locations must be >= 1"):
        create_synthetic_trending_metrics(
            n_years=5,
            n_locations=0,
            seed=42,
        )


def test_trending_single_location_latitude_zero() -> None:
    """Single location gets latitude 0.0."""
    data = create_synthetic_trending_metrics(
        n_years=5,
        n_locations=1,
        seed=42,
    )
    assert int(data["latitudes"].shape[0]) == 1
    assert float(data["latitudes"].flat[0]) == 0.0
