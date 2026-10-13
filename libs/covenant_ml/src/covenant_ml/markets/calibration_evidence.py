"""Offline calibration inputs bound to report provenance and original market metadata."""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import (
    JSONObject,
    narrow_json_to_dict,
    require_dict,
    require_list,
    require_str,
)

from .archive import ResponseArchive
from .collection import MarketCollector
from .forecast_collection import forecast_inventory
from .forecast_inputs import bracket_probability, strike_bounds
from .forecast_model import chronological_split
from .forecast_records import (
    CITY_COORDINATES,
    ComparisonPlan,
    ForecastRow,
    decode_comparison_plan,
    decode_forecast_row,
)
from .reporting import read_verified_market_report


def load_calibration_evidence(
    report_path: Path, archive: ResponseArchive
) -> tuple[
    ComparisonPlan,
    tuple[ForecastRow, ...],
    dict[str, tuple[float | None, float | None]],
    tuple[Path, ...],
    str,
]:
    """Validate report hashes and reconcile each row with its archived native market.

    Args:
        report_path: Original fixed-uncertainty historical report and sidecar.
        archive: Original market archive, accessed exclusively through offline replay.

    Returns:
        Date plan, original rows, native bounds, verified paths and source report digest.

    Raises:
        ValueError: CALIBRATION-004 for provenance mismatch, CALIBRATION-005 for
            an unbound archive request, CALIBRATION-006 for inconsistent observations.
        OSError: Required evidence is missing or unreadable.
        TypeError: Required report or archive fields are malformed.
    """
    report, paths, digest = read_verified_market_report(report_path, "CALIBRATION-004")
    verified = {path.resolve() for path in paths}

    def read(path: str, params: dict[str, str | int]) -> JSONObject:
        """Replay only requests whose files are bound to the original report.

        Args:
            path: Native endpoint suffix.
            params: Complete request query parameters.

        Returns:
            Verified archived native response.

        Raises:
            ValueError: CALIBRATION-005 if the request was not original evidence.
        """
        file = archive.file_for(archive.request_url(path, params)).resolve()
        if file not in verified:
            raise ValueError("[CALIBRATION-005] market request is outside original evidence")
        return archive.replay(path, params)

    plan = decode_comparison_plan(require_dict(report, "plan"))
    rows = tuple(
        decode_forecast_row(narrow_json_to_dict(raw))
        for partition in ("training_rows", "evaluation_rows")
        for raw in require_list(report, partition)
    )
    chronological_split(rows, plan)
    collector = MarketCollector(read)
    bounds: dict[str, tuple[float | None, float | None]] = {}
    for series in CITY_COORDINATES:
        collector.verify_fee(series)
        native, specs = forecast_inventory(plan, collector, series)
        for row in rows:
            if row["series"] != series:
                continue
            ticker = row["ticker"]
            if ticker not in native:
                raise ValueError("[CALIBRATION-006] observation has no native market")
            spec = specs[ticker]
            bracket = strike_bounds(native[ticker])
            if (
                bracket is None
                or row["event_day"] != spec["day"]
                or row["decision_ms"] != spec["decision_ms"]
                or row["settled_ms"] != spec["settled_ms"]
                or row["yes"] * 10000 != spec["settlement"]
                or row["settlement_rule"] != require_str(native[ticker], "rules_primary")
            ):
                raise ValueError("[CALIBRATION-006] observation conflicts with native settlement")
            expected = bracket_probability(row["forecast_f"], bracket, 3.0)
            if abs(expected - row["weather_p"]) > 1e-12:
                raise ValueError("[CALIBRATION-006] source probability is not the fixed model")
            bounds[ticker] = bracket
    return plan, rows, bounds, paths, digest
