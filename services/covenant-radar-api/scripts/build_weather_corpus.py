"""Build the ``weather_tmax`` corpus from vendored GHCN-D station files.

The training corpus for the weather domain, built THROUGH the deployed
feature path: every row's features come from
:class:`covenant_radar_api.domains.weather.features.WeatherFeatureExtractor`
over a :class:`TemporalFeatureState` fitted with covenant_ml's McKinnon
machinery — so training and serving cannot disagree about what a feature
means.

Honesty rules, applied by construction:

- The temporal state (Fourier seasonal cycle, tail thresholds, median
  baseline) is fitted per station on the FIT years only (1950-1989 by
  default); corpus rows come exclusively from later years, so no
  evaluation-period data leaks into the state.
- Rows are restricted to the season the thresholds are defined on
  (June-August, McKinnon's summer), and the target is the NEXT day's
  anomaly at the same station — day-ahead regression. A row exists only
  when the three prior calendar days and the next calendar day are all
  present and in-season: gaps and season boundaries drop the row rather
  than imputing anything.
- Quality-flagged observations (nonblank QFLAG) are dropped, never
  patched.
- ``station`` is the group column, never a feature: a station's summer
  days are correlated, so the benchmark split must be by station.

Usage:
    poetry run python -m scripts.build_weather_corpus \
        --raw-dir data/external/weather_tmax/raw \
        --out data/external/weather_tmax/data.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict

import numpy as np
from covenant_ml.datasets.loaders._netcdf_heat_metrics import fit_temporal_features
from covenant_ml.datasets.types_temporal import (
    SeasonalCycleCoefficients,
    TailThresholds,
    TemporalFeatureConfig,
    TemporalFeatureState,
)
from numpy.typing import NDArray

from covenant_radar_api.domains.weather.features import WeatherFeatureExtractor
from covenant_radar_api.domains.weather.schemas import WeatherEventV1

#: Years whose observations fit the temporal state. Nothing from later
#: years touches the state.
FIT_YEARS: tuple[int, int] = (1950, 1989)

#: Years that produce corpus rows.
ROW_YEARS: tuple[int, int] = (1990, 2024)

#: The season the tail thresholds are defined on (McKinnon's summer).
SEASON_MONTHS: tuple[int, ...] = (6, 7, 8)

#: The fitting configuration: 5 Fourier harmonics at annual frequency
#: (McKinnon's choice) and 95th/5th percentile tails.
TEMPORAL_CONFIG = TemporalFeatureConfig(
    n_fourier_harmonics=5,
    hot_cutoff_percentile=95.0,
    cold_cutoff_percentile=5.0,
    season="warm",
    season_months=SEASON_MONTHS,
    compute_ar1=False,
)

#: Completeness gate: a station enters the corpus only with at least
#: this many quality-passed days in the fit window (of 14,610 possible)
#: and this many in-season days in the row window (of ~3,220 possible).
#: The inventory's first/last-year span says nothing about continuity —
#: one vendored station has TMAX 1939-1944 and nothing in the fit window
#: at all — so eligibility is decided from the DATA, mechanically, and
#: skipped stations are counted and printed, never silent.
MIN_FIT_DAYS = 12000
MIN_ROW_SEASON_DAYS = 1500

#: Output column order: group column, features, target.
OUTPUT_HEADER: tuple[str, ...] = (
    "station",
    "day_of_year",
    "anomaly",
    "hot_excess",
    "cold_excess",
    "is_hot_extreme",
    "is_cold_extreme",
    "anomaly_lag1",
    "anomaly_lag2",
    "anomaly_lag3",
    "next_day_anomaly",
)


class StationSeries(TypedDict):
    """One station's quality-filtered daily TMAX series.

    Args:
        station: GHCN station identifier.
        days: Observation dates in ascending order.
        temps_c: TMAX in degrees Celsius, aligned with ``days``.
    """

    station: str
    days: list[date]
    temps_c: list[float]


class CorpusStats(TypedDict):
    """Summary statistics of a built corpus.

    Args:
        n_rows: Total corpus rows.
        n_stations: Stations contributing at least one row.
        skipped: Stations refused by the completeness gate, with the
            reason, in station order.
        target_mean: Mean of the target column.
    """

    n_rows: int
    n_stations: int
    skipped: list[str]
    target_mean: float


def _write(message: str) -> None:
    """Write a message to stdout.

    Args:
        message: Text to emit.
    """
    sys.stdout.write(message)
    sys.stdout.flush()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Build the weather_tmax day-ahead anomaly corpus from GHCN-D raw files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory holding <STATION>.csv.gz files from fetch_ghcnd_weather.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the corpus data.csv.",
    )
    return parser


def parse_station_file(path: Path) -> StationSeries:
    """Parse one GHCN-D by_station file into a daily TMAX series.

    The by_station files carry no header; columns are ID, DATE
    (YYYYMMDD), ELEMENT, VALUE, MFLAG, QFLAG, SFLAG, OBS-TIME. TMAX
    values are tenths of degrees Celsius. Rows with a nonblank QFLAG
    (failed quality assurance) are dropped, never patched.

    Args:
        path: The ``<STATION>.csv.gz`` file.

    Returns:
        The station's quality-filtered series, ascending by date.

    Raises:
        ValueError: If the file holds no usable TMAX rows.
    """
    station = path.name.split(".")[0]
    by_day: dict[date, float] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 6 or row[2] != "TMAX":
                continue
            qflag = row[5].strip()
            if qflag != "":
                continue
            raw_value = row[3].strip()
            if raw_value == "" or raw_value == "-9999":
                continue
            stamp = row[1]
            day = date(int(stamp[0:4]), int(stamp[4:6]), int(stamp[6:8]))
            by_day[day] = float(int(raw_value)) / 10.0
    if not by_day:
        raise ValueError(f"{path.name}: no usable TMAX rows")
    days = sorted(by_day)
    return StationSeries(
        station=station,
        days=days,
        temps_c=[by_day[d] for d in days],
    )


def fit_station_state(
    series: StationSeries,
    fit_years: tuple[int, int],
) -> TemporalFeatureState:
    """Fit one station's temporal state on the fit years only.

    Args:
        series: The station's daily series.
        fit_years: Inclusive (first, last) year bounds for fitting.

    Returns:
        A single-location fitted state.

    Raises:
        ValueError: If the fit years hold no observations (raised by the
            fitting machinery, which also rejects data that cannot
            determine the seasonal cycle).
    """
    values: list[float] = []
    doy: list[int] = []
    months: list[int] = []
    years: list[int] = []
    for day, temp in zip(series["days"], series["temps_c"], strict=True):
        if fit_years[0] <= day.year <= fit_years[1]:
            values.append(temp)
            doy.append(day.timetuple().tm_yday)
            months.append(day.month)
            years.append(day.year)
    daily_values: NDArray[np.float64] = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return fit_temporal_features(
        daily_values,
        np.asarray(doy, dtype=np.int64),
        np.asarray(months, dtype=np.int64),
        np.asarray(years, dtype=np.int64),
        TEMPORAL_CONFIG,
    )


def combine_states(states: list[TemporalFeatureState]) -> TemporalFeatureState:
    """Concatenate single-location states into one multi-location state.

    Per-location parameters are independent throughout the state shape,
    so concatenation in station order is exact — location ``i`` of the
    result is exactly state ``i``'s only location.

    Args:
        states: Single-location states in station order.

    Returns:
        One state whose location axis spans the inputs.

    Raises:
        ValueError: If ``states`` is empty.
    """
    if not states:
        raise ValueError("combine_states needs at least one state")
    first_cycle = states[0]["seasonal_cycle"]
    n_harmonics = first_cycle["n_harmonics"]
    cos_rows: list[tuple[float, ...]] = []
    sin_rows: list[tuple[float, ...]] = []
    for k in range(n_harmonics):
        cos_rows.append(tuple(s["seasonal_cycle"]["cos_coefficients"][k][0] for s in states))
        sin_rows.append(tuple(s["seasonal_cycle"]["sin_coefficients"][k][0] for s in states))
    cycle = SeasonalCycleCoefficients(
        n_harmonics=n_harmonics,
        cos_coefficients=tuple(cos_rows),
        sin_coefficients=tuple(sin_rows),
        mean=tuple(s["seasonal_cycle"]["mean"][0] for s in states),
        n_days_per_year=first_cycle["n_days_per_year"],
    )
    thresholds = TailThresholds(
        hot_threshold=tuple(s["thresholds"]["hot_threshold"][0] for s in states),
        cold_threshold=tuple(s["thresholds"]["cold_threshold"][0] for s in states),
        hot_percentile=states[0]["thresholds"]["hot_percentile"],
        cold_percentile=states[0]["thresholds"]["cold_percentile"],
    )
    return TemporalFeatureState(
        config=states[0]["config"],
        seasonal_cycle=cycle,
        thresholds=thresholds,
        median_baseline=tuple(s["median_baseline"][0] for s in states),
        n_locations=len(states),
    )


def _scalar(values: NDArray[np.float64], index: int) -> float:
    """Read one feature value as a plain float.

    Args:
        values: A feature vector from the extractor.
        index: Position to read.

    Returns:
        The value as a Python float.
    """
    value: np.float64 = values[index]
    return float(value)


def _event_for(station: str, day: date, temp_c: float) -> WeatherEventV1:
    """Build the observation event the deployed extractor consumes.

    Args:
        station: Station identifier.
        day: Observation date.
        temp_c: TMAX in degrees Celsius.

    Returns:
        The event, with corpus-build placeholders for the stream-only
        identity fields (they do not participate in extraction).
    """
    return WeatherEventV1(
        type="weather.observation.v1",
        event_id=f"corpus-{station}-{day.isoformat()}",
        station_id=station,
        day_of_year=day.timetuple().tm_yday,
        temperature=temp_c,
        timestamp=f"{day.isoformat()}T00:00:00+00:00",
    )


def build_station_rows(
    series: StationSeries,
    extractor: WeatherFeatureExtractor,
    row_years: tuple[int, int],
) -> list[list[str]]:
    """Build one station's corpus rows.

    A row exists for day ``t`` only when ``t-3..t-1`` and ``t+1`` are
    all present, calendar-consecutive, in-season and within the row
    years — gaps and season boundaries drop rows rather than impute.

    Args:
        series: The station's daily series.
        extractor: The deployed feature extractor over the fitted state.
        row_years: Inclusive (first, last) year bounds for rows.

    Returns:
        Output rows in ``OUTPUT_HEADER`` order (station first).
    """
    station = series["station"]
    in_window: dict[date, float] = {}
    for day, temp in zip(series["days"], series["temps_c"], strict=True):
        if row_years[0] <= day.year <= row_years[1] and day.month in SEASON_MONTHS:
            in_window[day] = temp

    anomaly_cache: dict[date, NDArray[np.float64]] = {}

    def features_of(day: date) -> NDArray[np.float64]:
        cached = anomaly_cache.get(day)
        if cached is not None:
            return cached
        computed = extractor.extract(_event_for(station, day, in_window[day]))
        anomaly_cache[day] = computed
        return computed

    rows: list[list[str]] = []
    one = timedelta(days=1)
    for day in sorted(in_window):
        needed = (day - 3 * one, day - 2 * one, day - one, day + one)
        if any(other not in in_window for other in needed):
            continue
        feats = features_of(day)
        lags = (
            _scalar(features_of(day - one), 0),
            _scalar(features_of(day - 2 * one), 0),
            _scalar(features_of(day - 3 * one), 0),
        )
        target = _scalar(features_of(day + one), 0)
        rows.append(
            [
                station,
                str(day.timetuple().tm_yday),
                f"{_scalar(feats, 0):.6f}",
                f"{_scalar(feats, 1):.6f}",
                f"{_scalar(feats, 2):.6f}",
                f"{_scalar(feats, 3):.1f}",
                f"{_scalar(feats, 4):.1f}",
                f"{lags[0]:.6f}",
                f"{lags[1]:.6f}",
                f"{lags[2]:.6f}",
                f"{target:.6f}",
            ]
        )
    return rows


def _completeness_verdict(series: StationSeries) -> str | None:
    """Apply the completeness gate to one station.

    Args:
        series: The station's daily series.

    Returns:
        None when the station is eligible, else the refusal reason.
    """
    n_fit = 0
    n_row_season = 0
    for day in series["days"]:
        if FIT_YEARS[0] <= day.year <= FIT_YEARS[1]:
            n_fit += 1
        elif ROW_YEARS[0] <= day.year <= ROW_YEARS[1] and day.month in SEASON_MONTHS:
            n_row_season += 1
    if n_fit < MIN_FIT_DAYS:
        return f"{n_fit} fit-window days < {MIN_FIT_DAYS}"
    if n_row_season < MIN_ROW_SEASON_DAYS:
        return f"{n_row_season} row-season days < {MIN_ROW_SEASON_DAYS}"
    return None


def build_corpus(raw_dir: Path, out_path: Path) -> CorpusStats:
    """Build the corpus from every eligible station in the raw directory.

    Args:
        raw_dir: Directory of ``<STATION>.csv.gz`` files.
        out_path: Destination for the corpus CSV.

    Returns:
        Summary statistics of what was written, including every station
        the completeness gate refused and why.

    Raises:
        ValueError: If the raw directory holds no station files, or the
            gate leaves no eligible station.
    """
    station_paths = sorted(raw_dir.glob("*.csv.gz"))
    if not station_paths:
        raise ValueError(f"no station files under {raw_dir}")

    all_series = [parse_station_file(path) for path in station_paths]
    series_list: list[StationSeries] = []
    skipped: list[str] = []
    for series in all_series:
        verdict = _completeness_verdict(series)
        if verdict is None:
            series_list.append(series)
        else:
            skipped.append(f"{series['station']}: {verdict}")
    if not series_list:
        raise ValueError("the completeness gate left no eligible station")

    states = [fit_station_state(series, FIT_YEARS) for series in series_list]
    combined = combine_states(states)
    station_to_location = {series["station"]: idx for idx, series in enumerate(series_list)}
    extractor = WeatherFeatureExtractor(combined, station_to_location)

    all_rows: list[list[str]] = []
    n_stations = 0
    for series in series_list:
        rows = build_station_rows(series, extractor, ROW_YEARS)
        if rows:
            n_stations += 1
        all_rows.extend(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(OUTPUT_HEADER)
        writer.writerows(all_rows)

    target_sum = 0.0
    for row in all_rows:
        target_sum += float(row[-1])
    n_rows = len(all_rows)
    return CorpusStats(
        n_rows=n_rows,
        n_stations=n_stations,
        skipped=skipped,
        target_mean=target_sum / n_rows if n_rows > 0 else 0.0,
    )


def main(argv: list[str] | None = None) -> int:
    """Build the corpus and report its shape.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parsed = build_parser().parse_args(argv)
    raw_dir: Path = parsed.raw_dir
    out: Path = parsed.out

    stats = build_corpus(raw_dir, out)
    for reason in stats["skipped"]:
        _write(f"  skipped {reason}\n")
    _write(
        f"weather_tmax: {stats['n_rows']} rows across {stats['n_stations']} stations "
        f"({len(stats['skipped'])} skipped by the completeness gate) -> {out}\n"
        f"  next_day_anomaly: mean {stats['target_mean']:.3f} C\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
