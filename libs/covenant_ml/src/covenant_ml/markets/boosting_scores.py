"""Equal-date predictive contrasts and unchanged quoted execution diagnostics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from platform_core.json_utils import JSONObject, JSONValue, require_float

from ..evaluation.codec_records import encode_interval
from ..evaluation.paired import paired_mean_difference
from ..evaluation.resampling import build_resampling
from ..evaluation.types import ResamplingUnit
from ..metrics import brier_per_row, log_loss_per_row
from .forecast_model import float_values
from .forecast_policy import quoted_policy
from .forecast_records import ForecastRow


def daily_means(values: Sequence[float], rows: Sequence[ForecastRow]) -> tuple[float, ...]:
    """Give each represented date equal weight regardless of contract count.

    Args:
        values: Per-contract values in row order.
        rows: Corresponding observations.

    Returns:
        Mean for each distinct date, in chronological order.

    Raises:
        ValueError: Values and observations are not aligned.
    """
    days: dict[str, list[float]] = defaultdict(list)
    for value, row in zip(values, rows, strict=True):
        days[row["event_day"]].append(value)
    return tuple(sum(days[day]) / len(days[day]) for day in sorted(days))


def score_predictions(
    rows: Sequence[ForecastRow], predictions: dict[str, tuple[float, ...]]
) -> JSONObject:
    """Score all declared arms and paired daily contrasts against both references.

    Args:
        rows: Complete held-out observations, with at least nine represented dates.
        predictions: Aligned raw_market, logistic and boosting probabilities.

    Returns:
        Full predictions, losses, costs and equal-date uncertainty diagnostics.

    Raises:
        ValueError: BOOST-003 for too few dates or an undeclared arm set.
    """
    if len({row["event_day"] for row in rows}) < 9 or set(predictions) != {
        "raw_market",
        "logistic",
        "boosting",
    }:
        raise ValueError("[BOOST-003] require nine dates and the three declared arms")
    label_values = [row["yes"] for row in rows]
    labels: NDArray[np.int64] = np.asarray(label_values, dtype=np.int64)
    arms: JSONObject = {}
    losses: dict[str, tuple[float, ...]] = {}
    for name, probabilities in predictions.items():
        policy = quoted_policy(rows, probabilities)
        values: NDArray[np.float64] = np.asarray(probabilities, dtype=np.float64)
        losses[name] = float_values(brier_per_row(labels, values))
        logs = float_values(log_loss_per_row(labels, values))
        days = daily_means(losses[name], rows)
        encoded_predictions: list[JSONValue] = list(probabilities)
        arms[name] = {
            "predictions": encoded_predictions,
            "brier_by_row": list(losses[name]),
            "brier_by_date": list(days),
            "date_mean_brier": sum(days) / len(days),
            "row_mean_brier": sum(losses[name]) / len(rows),
            "row_mean_log_loss": sum(logs) / len(rows),
            "policy": policy,
        }
    comparisons: JSONObject = {}
    for reference in ("logistic", "raw_market"):
        differences = tuple(
            a - b for a, b in zip(losses["boosting"], losses[reference], strict=True)
        )
        daily = daily_means(differences, rows)
        array: NDArray[np.float64] = np.asarray(daily, dtype=np.float64)
        groups: NDArray[np.int64] = np.arange(len(daily), dtype=np.int64)
        cluster = encode_interval(
            paired_mean_difference(
                array, build_resampling(ResamplingUnit.CLUSTER, len(daily), groups, 1), 0.95
            )
        )
        serial = encode_interval(
            paired_mean_difference(
                array, build_resampling(ResamplingUnit.BLOCK, len(daily), groups, 8), 0.95
            )
        )
        comparisons[reference] = {
            "daily_differences": list(daily),
            "mean": sum(daily) / len(daily),
            "date_clustered": cluster,
            "daily_newey_west": serial,
            "both_upper_bounds_negative": (
                require_float(cluster, "ci_high") < 0 and require_float(serial, "ci_high") < 0
            ),
        }
    ordered_dates = sorted({row["event_day"] for row in rows})
    event_dates: list[JSONValue] = list(ordered_dates)
    return {
        "arms": arms,
        "comparisons": comparisons,
        "event_dates": event_dates,
        "primary_reference": "logistic",
    }
