"""Tests for the weather_tmax corpus builder.

Drives the real pipeline — GHCN-shaped gzip fixtures through the real
covenant_ml fitting machinery and the real deployed feature extractor —
on synthetic stations small enough to hand-check: a pure sinusoid plus a
known bump, so anomalies and the day-ahead target are predictable.
"""

from __future__ import annotations

import csv
import gzip
import math
import runpy
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest
from scripts.build_weather_corpus import (
    FIT_YEARS,
    MIN_FIT_DAYS,
    MIN_ROW_SEASON_DAYS,
    OUTPUT_HEADER,
    ROW_YEARS,
    build_corpus,
    build_parser,
    combine_states,
    main,
    parse_station_file,
)


def _sinusoid_c(day: date) -> float:
    """The synthetic climate: annual sinusoid in degrees Celsius."""
    doy = day.timetuple().tm_yday
    return 20.0 + 10.0 * math.sin(2.0 * math.pi * doy / 365.0)


def _write_station(
    path: Path,
    station: str,
    first_year: int,
    last_year: int,
    bump_day: date | None = None,
    flagged_day: date | None = None,
    missing_day: date | None = None,
) -> None:
    """Write a GHCN-shaped by_station gzip for the synthetic climate.

    Args:
        path: Destination ``<STATION>.csv.gz``.
        station: Station identifier.
        first_year: First calendar year of observations.
        last_year: Last calendar year, inclusive.
        bump_day: Optional day given a +8C excursion.
        flagged_day: Optional day written with a QFLAG (must be dropped).
        missing_day: Optional day written as GHCN's -9999 sentinel
            (must be dropped like a gap).
    """
    lines: list[str] = []
    day = date(first_year, 1, 1)
    end = date(last_year, 12, 31)
    one = timedelta(days=1)
    while day <= end:
        temp = _sinusoid_c(day)
        if bump_day is not None and day == bump_day:
            temp += 8.0
        tenths = round(temp * 10.0)
        qflag = "X" if flagged_day is not None and day == flagged_day else ""
        stamp = day.strftime("%Y%m%d")
        if missing_day is not None and day == missing_day:
            lines.append(f"{station},{stamp},TMAX,-9999,,{qflag},H,")
        else:
            lines.append(f"{station},{stamp},TMAX,{tenths},,{qflag},H,")
        day += one
    path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode("utf-8")))


def _full_span_station(tmp_path: Path, station: str, **kwargs: date | None) -> Path:
    """Write a station covering the fit and row windows fully."""
    out = tmp_path / f"{station}.csv.gz"
    _write_station(out, station, FIT_YEARS[0], ROW_YEARS[1], **kwargs)
    return out


class TestParseStationFile:
    """GHCN parsing: units, quality flags, ordering."""

    def test_values_are_tenths_of_celsius(self, tmp_path: Path) -> None:
        """A 20.0C sinusoid day round-trips through the tenths encoding."""
        path = tmp_path / "TST00000001.csv.gz"
        _write_station(path, "TST00000001", 2000, 2000)
        series = parse_station_file(path)
        assert series["station"] == "TST00000001"
        assert len(series["days"]) == 366
        first_expected = _sinusoid_c(date(2000, 1, 1))
        assert abs(series["temps_c"][0] - first_expected) < 0.051

    def test_quality_flagged_days_are_dropped(self, tmp_path: Path) -> None:
        """A nonblank QFLAG removes the day rather than patching it."""
        flagged = date(2000, 7, 15)
        path = tmp_path / "TST00000001.csv.gz"
        _write_station(path, "TST00000001", 2000, 2000, flagged_day=flagged)
        series = parse_station_file(path)
        assert flagged not in series["days"]
        assert len(series["days"]) == 365

    def test_an_empty_file_is_refused(self, tmp_path: Path) -> None:
        """A file with no usable TMAX rows is an error, not an empty series."""
        path = tmp_path / "TST00000001.csv.gz"
        path.write_bytes(gzip.compress(b"TST00000001,20000101,PRCP,10,,,H,\n"))
        with pytest.raises(ValueError, match="no usable TMAX rows"):
            parse_station_file(path)


class TestBuildCorpus:
    """The full pipeline over synthetic stations."""

    def test_rows_carry_the_bump_and_its_lags(self, tmp_path: Path) -> None:
        """A +8C day shows up as its own anomaly, the next day's lag1,
        and the PRIOR day's target."""
        bump = date(2000, 7, 15)
        _full_span_station(tmp_path, "TST00000001", bump_day=bump)
        out = tmp_path / "data.csv"
        stats = build_corpus(tmp_path, out)
        assert stats["n_stations"] == 1
        assert stats["skipped"] == []

        with out.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        assert tuple(rows[0]) == OUTPUT_HEADER
        body = rows[1:]
        bump_doy = bump.timetuple().tm_yday
        # Rows repeat each day-of-year across 35 summers, so the bump's
        # rows are identified by their VALUES: exactly one row carries
        # the +8C excursion as its own anomaly, one as its target (the
        # prior day), and one as lag1 (the next day).
        bump_rows = [r for r in body if float(r[2]) > 6.0]
        assert len(bump_rows) == 1
        assert int(bump_rows[0][1]) == bump_doy
        assert float(bump_rows[0][5]) == 1.0  # is_hot_extreme
        target_rows = [r for r in body if float(r[-1]) > 6.0]
        assert len(target_rows) == 1
        assert int(target_rows[0][1]) == bump_doy - 1
        lag1_rows = [r for r in body if float(r[7]) > 6.0]
        assert len(lag1_rows) == 1
        assert int(lag1_rows[0][1]) == bump_doy + 1

    def test_on_a_pure_sinusoid_anomalies_are_small(self, tmp_path: Path) -> None:
        """With the climate exactly the fitted Fourier shape, residual
        anomalies stay within the tenths-rounding noise."""
        _full_span_station(tmp_path, "TST00000001")
        out = tmp_path / "data.csv"
        build_corpus(tmp_path, out)
        with out.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        for row in rows[1:]:
            assert abs(float(row[2])) < 1.0

    def test_incomplete_stations_are_skipped_with_reasons(self, tmp_path: Path) -> None:
        """A station below the fit-window floor is refused by name."""
        _full_span_station(tmp_path, "TST00000001")
        thin = tmp_path / "TST00000002.csv.gz"
        _write_station(thin, "TST00000002", ROW_YEARS[0], ROW_YEARS[1])
        out = tmp_path / "data.csv"
        stats = build_corpus(tmp_path, out)
        assert stats["n_stations"] == 1
        assert len(stats["skipped"]) == 1
        assert stats["skipped"][0].startswith("TST00000002: 0 fit-window days")
        assert str(MIN_FIT_DAYS) in stats["skipped"][0]

    def test_gap_days_drop_rows_rather_than_impute(self, tmp_path: Path) -> None:
        """A quality-dropped day removes every row needing it."""
        flagged = date(2000, 7, 15)
        _full_span_station(tmp_path, "TST00000001", flagged_day=flagged)
        out = tmp_path / "data.csv"
        build_corpus(tmp_path, out)
        with out.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        doy_2000 = {int(r[1]) for r in rows[1:]}
        gap_doy = flagged.timetuple().tm_yday
        # The gap day itself and the 4 rows needing it (t-1 target, t+1..t+3
        # lags) cannot exist for that day-of-year... other years still carry
        # the doy, so assert via row count instead: one station, one flagged
        # day removes exactly 5 rows relative to the clean build.
        clean_dir = tmp_path / "clean"
        clean_dir.mkdir()
        _full_span_station(clean_dir, "TST00000001")
        clean_out = clean_dir / "data.csv"
        build_corpus(clean_dir, clean_out)
        with clean_out.open("r", encoding="utf-8", newline="") as handle:
            clean_rows = list(csv.reader(handle))
        assert len(clean_rows) - len(rows) == 5
        assert gap_doy in doy_2000  # other years still carry that doy

    def test_no_eligible_station_is_an_error(self, tmp_path: Path) -> None:
        """A directory of hollow stations cannot silently build nothing."""
        thin = tmp_path / "TST00000002.csv.gz"
        _write_station(thin, "TST00000002", ROW_YEARS[0], ROW_YEARS[0])
        with pytest.raises(ValueError, match="left no eligible station"):
            build_corpus(tmp_path, tmp_path / "data.csv")

    def test_an_empty_directory_is_an_error(self, tmp_path: Path) -> None:
        """No station files at all is its own refusal."""
        with pytest.raises(ValueError, match="no station files"):
            build_corpus(tmp_path, tmp_path / "data.csv")


class TestCombineStates:
    """State concatenation refusals."""

    def test_empty_input_is_refused(self) -> None:
        """No states cannot silently combine into nothing."""
        with pytest.raises(ValueError, match="at least one state"):
            combine_states([])


class TestSentinelAndSparseStations:
    """GHCN sentinels and hollow-row-window stations."""

    def test_minus_9999_days_are_dropped_like_gaps(self, tmp_path: Path) -> None:
        """The GHCN missing sentinel removes the day, never a zero."""
        missing = date(2000, 7, 15)
        path = tmp_path / "TST00000001.csv.gz"
        _write_station(path, "TST00000001", 2000, 2000, missing_day=missing)
        series = parse_station_file(path)
        assert missing not in series["days"]
        assert len(series["days"]) == 365

    def test_eligible_station_with_no_consecutive_runs_yields_no_rows(self, tmp_path: Path) -> None:
        """A station observing every OTHER summer day passes the
        completeness gate yet can never satisfy the 4-consecutive-day
        row requirement — it contributes zero rows, and the station
        count says so."""
        sparse = tmp_path / "TST00000004.csv.gz"
        lines: list[str] = []
        day = date(FIT_YEARS[0], 1, 1)
        end = date(ROW_YEARS[1], 12, 31)
        one = timedelta(days=1)
        while day <= end:
            in_row_window = day.year >= ROW_YEARS[0]
            if not in_row_window or day.toordinal() % 2 == 0:
                tenths = round(_sinusoid_c(day) * 10.0)
                stamp = day.strftime("%Y%m%d")
                lines.append(f"TST00000004,{stamp},TMAX,{tenths},,,H,")
            day += one
        sparse.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode("utf-8")))
        _full_span_station(tmp_path, "TST00000001")
        out = tmp_path / "data.csv"
        stats = build_corpus(tmp_path, out)
        assert stats["skipped"] == []
        assert stats["n_stations"] == 1

    def test_row_season_floor_is_its_own_refusal(self, tmp_path: Path) -> None:
        """A station complete in the fit window but hollow in the row
        window is refused with the row-season reason."""
        path = tmp_path / "TST00000003.csv.gz"
        _write_station(path, "TST00000003", FIT_YEARS[0], FIT_YEARS[1])
        _full_span_station(tmp_path, "TST00000001")
        out = tmp_path / "data.csv"
        stats = build_corpus(tmp_path, out)
        assert len(stats["skipped"]) == 1
        assert "row-season days" in stats["skipped"][0]
        assert str(MIN_ROW_SEASON_DAYS) in stats["skipped"][0]


class TestMain:
    """The CLI wiring."""

    def test_parser_requires_both_paths(self) -> None:
        """--raw-dir and --out are both mandatory."""
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_main_reports_skips_and_shape(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Stdout names every skip and states the corpus shape."""
        _full_span_station(tmp_path, "TST00000001")
        thin = tmp_path / "TST00000002.csv.gz"
        _write_station(thin, "TST00000002", ROW_YEARS[0], ROW_YEARS[1])
        out = tmp_path / "data.csv"
        exit_code = main(["--raw-dir", str(tmp_path), "--out", str(out)])
        assert exit_code == 0
        captured = capsys.readouterr()
        lines = captured.out.splitlines()
        assert lines[0].startswith("  skipped TST00000002:")
        assert "(1 skipped by the completeness gate)" in lines[1]
        assert out.exists()

    def test_module_entry_point_raises_system_exit(self, tmp_path: Path) -> None:
        """Running as ``__main__`` exits through SystemExit(main())."""
        _full_span_station(tmp_path, "TST00000001")
        out = tmp_path / "data.csv"
        argv = [
            "build_weather_corpus",
            "--raw-dir",
            str(tmp_path),
            "--out",
            str(out),
        ]
        saved = sys.argv
        sys.argv = argv
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module("scripts.build_weather_corpus", run_name="__main__")
            assert excinfo.value.code == 0
        finally:
            sys.argv = saved
