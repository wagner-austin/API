"""Test utilities for the pluggable dataset loading system.

Provides fake implementations for testing without real filesystem access.
These are public utilities exported for consumers to use in their tests.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import math as _math
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from covenant_ml.datasets.types import (
    DatasetConfig,
    DatasetMeta,
    LoadedDataset,
    RegressionDatasetConfig,
    RegressionDatasetMeta,
    RegressionLoadedDataset,
)


class FakeDatasetLoader:
    """Fake dataset loader for testing.

    Returns deterministic synthetic datasets based on config.
    Does not access the filesystem.
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 10,
        positive_ratio: float = 0.3,
        random_state: int = 42,
    ) -> None:
        """Initialize fake loader with dataset parameters.

        Args:
            n_samples: Number of samples to generate.
            n_features: Number of features per sample.
            positive_ratio: Fraction of positive class samples.
            random_state: Random seed for reproducibility.
        """
        self._n_samples = n_samples
        self._n_features = n_features
        self._positive_ratio = positive_ratio
        self._random_state = random_state

    def load(
        self,
        config: DatasetConfig,
        external_dir: Path,
    ) -> LoadedDataset:
        """Generate a synthetic dataset based on config.

        Args:
            config: Dataset configuration (used for metadata).
            external_dir: Ignored for fake loader.

        Returns:
            LoadedDataset with synthetic data matching config name.
        """
        rng = np.random.default_rng(self._random_state)

        # Generate features
        x_array: NDArray[np.float64] = rng.standard_normal(
            (self._n_samples, self._n_features)
        ).astype(np.float64)

        # Generate labels with specified positive ratio
        n_positive = int(self._n_samples * self._positive_ratio)
        n_negative = self._n_samples - n_positive

        y_array: NDArray[np.int64] = np.zeros(self._n_samples, dtype=np.int64)
        y_array[:n_positive] = 1
        rng.shuffle(y_array)

        # Generate feature names
        feature_names = tuple(f"feature_{i}" for i in range(self._n_features))

        meta = DatasetMeta(
            name=config["name"],
            n_samples=self._n_samples,
            n_features=self._n_features,
            n_positive=n_positive,
            n_negative=n_negative,
            positive_ratio=self._positive_ratio,
            feature_names=feature_names,
            categorical_encodings=(),  # Fake loader generates numeric data only
        )

        return LoadedDataset(meta=meta, x=x_array, y=y_array)


def create_fake_dataset_loader(
    n_samples: int = 100,
    n_features: int = 10,
    positive_ratio: float = 0.3,
    random_state: int = 42,
) -> FakeDatasetLoader:
    """Factory function for creating fake dataset loader.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features per sample.
        positive_ratio: Fraction of positive class samples.
        random_state: Random seed for reproducibility.

    Returns:
        FakeDatasetLoader configured with specified parameters.
    """
    return FakeDatasetLoader(
        n_samples=n_samples,
        n_features=n_features,
        positive_ratio=positive_ratio,
        random_state=random_state,
    )


class FakeRegressionDatasetLoader:
    """Fake dataset loader for regression testing.

    Returns deterministic synthetic datasets with continuous float64 targets.
    Does not access the filesystem.
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 10,
        random_state: int = 42,
    ) -> None:
        """Initialize fake regression loader with dataset parameters.

        Args:
            n_samples: Number of samples to generate.
            n_features: Number of features per sample.
            random_state: Random seed for reproducibility.
        """
        self._n_samples = n_samples
        self._n_features = n_features
        self._random_state = random_state

    def load(
        self,
        config: RegressionDatasetConfig,
        external_dir: Path,
    ) -> RegressionLoadedDataset:
        """Generate a synthetic regression dataset based on config.

        Args:
            config: Regression dataset configuration (used for metadata).
            external_dir: Ignored for fake loader.

        Returns:
            RegressionLoadedDataset with synthetic continuous targets.
        """
        _ = external_dir

        # Generate features deterministically
        x_array: NDArray[np.float64] = np.zeros(
            (self._n_samples, self._n_features), dtype=np.float64
        )
        for i in range(self._n_samples):
            for j in range(self._n_features):
                x_array[i, j] = float((i * 7 + j * 3 + self._random_state) % 100) / 100.0

        # Generate continuous targets: linear combination of first two features
        y_array: NDArray[np.float64] = np.zeros(self._n_samples, dtype=np.float64)
        for i in range(self._n_samples):
            y_array[i] = (
                float(x_array.item((i, 0))) * 3.0
                + float(x_array.item((i, min(1, self._n_features - 1)))) * 1.5
                + 2.0
            )

        # Compute target statistics using explicit sum/len to avoid Any from np.mean
        n = self._n_samples
        y_sum = float(np.sum(y_array))
        target_mean = y_sum / n
        y_sq_diff_sum = float(np.sum((y_array - target_mean) ** 2))
        target_std = _math.sqrt(y_sq_diff_sum / n)
        target_min = float(np.min(y_array))
        target_max = float(np.max(y_array))

        feature_names = tuple(f"feature_{i}" for i in range(self._n_features))

        meta = RegressionDatasetMeta(
            name=config["name"],
            n_samples=self._n_samples,
            n_features=self._n_features,
            target_mean=target_mean,
            target_std=target_std,
            target_min=target_min,
            target_max=target_max,
            feature_names=feature_names,
            categorical_encodings=(),
        )

        return RegressionLoadedDataset(meta=meta, x=x_array, y=y_array)


def create_fake_regression_dataset_loader(
    n_samples: int = 100,
    n_features: int = 10,
    random_state: int = 42,
) -> FakeRegressionDatasetLoader:
    """Factory function for creating fake regression dataset loader.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features per sample.
        random_state: Random seed for reproducibility.

    Returns:
        FakeRegressionDatasetLoader configured with specified parameters.
    """
    return FakeRegressionDatasetLoader(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
    )


class SyntheticTemporalData(TypedDict, total=True):
    """Synthetic daily time-series data for temporal feature testing.

    Contains generated daily values with a known Fourier seasonal cycle
    and additive noise, plus ground-truth coefficients for assertion.
    Multi-location: each location has the same seasonal structure but
    different noise and an offset based on location index.

    Attributes:
        daily_values: Daily temperature values, shape (n_days, n_locations).
        day_of_year: Day-of-year for each observation, shape (n_days,).
            Values 1-365 (no leap years).
        month_labels: Calendar month, 1-12, for each day, shape (n_days,).
            Derived from day_of_year on a non-leap calendar, so the fitted
            season restriction can be exercised on this data.
        year_labels: Year label for each day, shape (n_days,).
        true_mean: Ground-truth mean per location, shape (n_locations,).
        true_cos_coefficients: Ground-truth cosine coefficients,
            shape (n_harmonics, n_locations).
        true_sin_coefficients: Ground-truth sine coefficients,
            shape (n_harmonics, n_locations).
        n_years: Number of complete years of data.
        n_locations: Number of spatial locations.
        n_harmonics: Number of Fourier harmonics in the seasonal cycle.
        noise_std: Standard deviation of additive Gaussian noise.
    """

    daily_values: NDArray[np.float64]
    day_of_year: NDArray[np.int64]
    month_labels: NDArray[np.int64]
    year_labels: NDArray[np.int64]
    true_mean: tuple[float, ...]
    true_cos_coefficients: tuple[tuple[float, ...], ...]
    true_sin_coefficients: tuple[tuple[float, ...], ...]
    n_years: int
    n_locations: int
    n_harmonics: int
    noise_std: float


# Length of each calendar month on a non-leap year, which is the calendar
# this generator produces.
_MONTH_LENGTHS: tuple[int, ...] = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


# Month of each day-of-year, expanded once so the lookup is a plain index.
# Indexing rather than searching also means a day-of-year outside the
# calendar raises immediately instead of being folded into December.
_MONTH_BY_DAY_OF_YEAR: tuple[int, ...] = tuple(
    month for month, length in enumerate(_MONTH_LENGTHS, start=1) for _ in range(length)
)


def _build_temporal_labels(
    n_years: int,
    n_days_per_year: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Build day-of-year, month and year label arrays.

    Args:
        n_years: Number of years.
        n_days_per_year: Days per year (365).

    Returns:
        Tuple of (day_of_year, month_labels, year_labels).
    """
    doy_list: list[int] = []
    month_list: list[int] = []
    yr_list: list[int] = []
    for year_idx in range(n_years):
        for day in range(1, n_days_per_year + 1):
            doy_list.append(day)
            month_list.append(_MONTH_BY_DAY_OF_YEAR[day - 1])
            yr_list.append(2000 + year_idx)
    day_of_year: NDArray[np.int64] = np.array(doy_list, dtype=np.int64)
    month_labels: NDArray[np.int64] = np.array(month_list, dtype=np.int64)
    year_labels: NDArray[np.int64] = np.array(yr_list, dtype=np.int64)
    return day_of_year, month_labels, year_labels


def _build_location_coefficients(
    n_harmonics: int,
    n_locations: int,
    seasonal_amplitude: float,
    mean_value: float,
) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]], list[float]]:
    """Build deterministic per-location Fourier coefficients and means.

    Args:
        n_harmonics: Number of Fourier harmonics.
        n_locations: Number of spatial locations.
        seasonal_amplitude: Base amplitude for first harmonic.
        mean_value: Base mean temperature for location 0.

    Returns:
        Tuple of (cos_coeffs, sin_coeffs, means) where cos/sin are
        lists of tuples (n_harmonics, n_locations) and means is
        a list of n_locations floats.
    """
    base_cos: list[float] = []
    base_sin: list[float] = []
    for k in range(1, n_harmonics + 1):
        sign_cos = 1.0 if k % 2 == 1 else -1.0
        sign_sin = -1.0 if k % 2 == 1 else 1.0
        base_cos.append(seasonal_amplitude * sign_cos / k)
        base_sin.append(seasonal_amplitude * sign_sin / (2.0 * k))

    cos_coeffs: list[tuple[float, ...]] = []
    sin_coeffs: list[tuple[float, ...]] = []
    for k_idx in range(n_harmonics):
        cos_row: list[float] = []
        sin_row: list[float] = []
        for j in range(n_locations):
            scale = 1.0 + 0.1 * j
            cos_row.append(base_cos[k_idx] * scale)
            sin_row.append(base_sin[k_idx] * scale)
        cos_coeffs.append(tuple(cos_row))
        sin_coeffs.append(tuple(sin_row))

    means = [mean_value + 2.0 * j for j in range(n_locations)]
    return cos_coeffs, sin_coeffs, means


def _build_seasonal_signal(
    day_of_year: NDArray[np.int64],
    n_locations: int,
    n_harmonics: int,
    cos_coeffs: list[tuple[float, ...]],
    sin_coeffs: list[tuple[float, ...]],
    means: list[float],
) -> NDArray[np.float64]:
    """Build deterministic seasonal cycle signal for all locations.

    Args:
        day_of_year: Day-of-year values, shape (n_days,).
        n_locations: Number of spatial locations.
        n_harmonics: Number of Fourier harmonics.
        cos_coeffs: Cosine coefficients (n_harmonics, n_locations).
        sin_coeffs: Sine coefficients (n_harmonics, n_locations).
        means: Mean values per location.

    Returns:
        Seasonal signal of shape (n_days, n_locations).
    """
    n_total_days = int(day_of_year.shape[0])
    n_days_per_year = 365
    daily_values = np.zeros((n_total_days, n_locations), dtype=np.float64)
    for j in range(n_locations):
        daily_values[:, j] = means[j]

    doy_float: NDArray[np.float64] = day_of_year.astype(np.float64)
    for k_idx in range(n_harmonics):
        k = k_idx + 1
        angle_values: list[float] = []
        for i in range(n_total_days):
            angle_values.append(2.0 * _math.pi * k * float(doy_float.flat[i]) / n_days_per_year)
        angle: NDArray[np.float64] = np.array(angle_values, dtype=np.float64)
        cos_vals = np.zeros(n_total_days, dtype=np.float64)
        sin_vals = np.zeros(n_total_days, dtype=np.float64)
        for i in range(n_total_days):
            cos_vals[i] = _math.cos(float(angle.flat[i]))
            sin_vals[i] = _math.sin(float(angle.flat[i]))
        for j in range(n_locations):
            daily_values[:, j] += cos_coeffs[k_idx][j] * cos_vals + sin_coeffs[k_idx][j] * sin_vals
    return daily_values


def create_synthetic_daily_timeseries(
    n_years: int,
    n_locations: int,
    n_harmonics: int,
    seed: int,
    seasonal_amplitude: float = 10.0,
    noise_std: float = 1.0,
    mean_value: float = 20.0,
) -> SyntheticTemporalData:
    """Create synthetic multi-location daily time-series with known Fourier cycle.

    Generates ``n_years * 365`` daily values at ``n_locations`` from::

        y(t, loc) = mean[loc] + sum_k(a_k[loc] * cos(2*pi*k*doy/365)
                                     + b_k[loc] * sin(2*pi*k*doy/365)) + noise

    Each location gets the same base coefficients scaled by ``1 + 0.1 * loc``
    and a mean offset of ``loc * 2.0``. Coefficients per harmonic:
    ``a_k = amplitude * (-1)^(k+1) / k``, ``b_k = amplitude * (-1)^k / (2*k)``.

    Args:
        n_years: Number of complete years (each 365 days, no leap years).
        n_locations: Number of spatial locations.
        n_harmonics: Number of Fourier harmonics in the seasonal cycle.
        seed: Random seed for noise generation.
        seasonal_amplitude: Base amplitude for the first harmonic.
        noise_std: Standard deviation of additive Gaussian noise.
        mean_value: Base mean temperature value (location 0).

    Returns:
        SyntheticTemporalData with generated data and ground-truth coefficients.

    Raises:
        ValueError: If n_years < 1, n_locations < 1, or n_harmonics < 1.
    """
    if n_years < 1:
        raise ValueError(f"n_years must be >= 1, got {n_years}")
    if n_locations < 1:
        raise ValueError(f"n_locations must be >= 1, got {n_locations}")
    if n_harmonics < 1:
        raise ValueError(f"n_harmonics must be >= 1, got {n_harmonics}")

    n_days_per_year = 365
    n_total_days = n_years * n_days_per_year

    day_of_year, month_labels, year_labels = _build_temporal_labels(n_years, n_days_per_year)
    cos_coeffs, sin_coeffs, means = _build_location_coefficients(
        n_harmonics,
        n_locations,
        seasonal_amplitude,
        mean_value,
    )
    daily_values = _build_seasonal_signal(
        day_of_year,
        n_locations,
        n_harmonics,
        cos_coeffs,
        sin_coeffs,
        means,
    )

    # Add noise per location
    rng = np.random.default_rng(seed)
    noise: NDArray[np.float64] = (
        rng.standard_normal((n_total_days, n_locations)).astype(np.float64) * noise_std
    )
    daily_values = daily_values + noise

    return SyntheticTemporalData(
        daily_values=daily_values,
        day_of_year=day_of_year,
        month_labels=month_labels,
        year_labels=year_labels,
        true_mean=tuple(means),
        true_cos_coefficients=tuple(cos_coeffs),
        true_sin_coefficients=tuple(sin_coeffs),
        n_years=n_years,
        n_locations=n_locations,
        n_harmonics=n_harmonics,
        noise_std=noise_std,
    )


class SyntheticTrendingData(TypedDict, total=True):
    """Synthetic multi-location metric data with known temporal trends.

    Contains generated metric values where each metric has a controlled
    linear trend in rank space, allowing tests to verify that rank-trend
    analysis detects trends correctly.

    Attributes:
        metrics: Metric values, shape (n_years, n_metrics, n_locations).
        metric_names: Ordered tuple of metric names.
        latitudes: Latitude values in degrees, shape (n_locations,).
        true_slopes: Known trend slope per metric (in raw value space).
        n_years: Number of years of data.
        n_locations: Number of spatial locations.
    """

    metrics: NDArray[np.float64]
    metric_names: tuple[str, ...]
    latitudes: NDArray[np.float64]
    true_slopes: tuple[float, ...]
    n_years: int
    n_locations: int


def create_synthetic_trending_metrics(
    n_years: int,
    n_locations: int,
    seed: int,
    trend_slope: float = 1.0,
    noise_std: float = 0.1,
) -> SyntheticTrendingData:
    """Create synthetic metric data with known linear trends for testing.

    Generates two metrics: ``seasonal_max`` (hot-ranked, with positive
    trend so rank 1 shifts toward later years) and ``seasonal_min``
    (cold-ranked, with negative trend so rank 1 shifts toward later
    years). Both trends should be detectable by rank-trend analysis.

    Each metric at each location follows::

        value(year, loc) = trend_slope * year + offset(loc) + noise

    where offset varies by location. Latitudes are evenly spaced
    from -60 to 60 degrees.

    Args:
        n_years: Number of years (must be >= 3 for meaningful trends).
        n_locations: Number of spatial locations (must be >= 1).
        seed: Random seed for noise generation.
        trend_slope: Linear trend magnitude per year.
        noise_std: Standard deviation of additive noise.

    Returns:
        SyntheticTrendingData with generated data and ground-truth slopes.

    Raises:
        ValueError: If n_years < 3 or n_locations < 1.
    """
    if n_years < 3:
        raise ValueError(f"n_years must be >= 3, got {n_years}")
    if n_locations < 1:
        raise ValueError(f"n_locations must be >= 1, got {n_locations}")

    rng = np.random.default_rng(seed)
    metric_names: tuple[str, ...] = ("seasonal_max", "seasonal_min")
    n_metrics = 2

    metrics: NDArray[np.float64] = np.zeros((n_years, n_metrics, n_locations), dtype=np.float64)

    for loc in range(n_locations):
        loc_offset = float(loc) * 5.0
        for yr in range(n_years):
            year_val = float(yr)
            # seasonal_max: positive trend (higher = more extreme heat)
            hot_value = trend_slope * year_val + loc_offset + 20.0
            noise_hot = float(rng.standard_normal()) * noise_std
            hot_flat_idx = yr * n_metrics * n_locations + 0 * n_locations + loc
            metrics.flat[hot_flat_idx] = hot_value + noise_hot

            # seasonal_min: negative trend (lower = more extreme cold)
            cold_value = -trend_slope * year_val + loc_offset - 20.0
            noise_cold = float(rng.standard_normal()) * noise_std
            cold_flat_idx = yr * n_metrics * n_locations + 1 * n_locations + loc
            metrics.flat[cold_flat_idx] = cold_value + noise_cold

    # Evenly spaced latitudes from -60 to 60
    latitudes: NDArray[np.float64] = np.zeros(n_locations, dtype=np.float64)
    if n_locations == 1:
        latitudes[0] = 0.0
    else:
        for i in range(n_locations):
            latitudes[i] = -60.0 + 120.0 * float(i) / float(n_locations - 1)

    return SyntheticTrendingData(
        metrics=metrics,
        metric_names=metric_names,
        latitudes=latitudes,
        true_slopes=(trend_slope, -trend_slope),
        n_years=n_years,
        n_locations=n_locations,
    )


__all__ = [
    "FakeDatasetLoader",
    "FakeRegressionDatasetLoader",
    "SyntheticTemporalData",
    "SyntheticTrendingData",
    "create_fake_dataset_loader",
    "create_fake_regression_dataset_loader",
    "create_synthetic_daily_timeseries",
    "create_synthetic_trending_metrics",
]
