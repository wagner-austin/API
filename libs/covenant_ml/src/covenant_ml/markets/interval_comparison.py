"""Fixed chronological comparison with future labels isolated from fitting and actions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from platform_core.json_utils import JSONObject, JSONValue, require_int

from ..metrics import brier_per_row, log_loss_per_row
from .forecast_model import float_values
from .interval_inputs import (
    interval_observations,
    require_interval_comparison_plan,
    split_interval_rows,
)
from .interval_model import fit_interval_model, predict_interval_model
from .interval_records import encode_interval_fit, encode_interval_row
from .interval_scoring import interval_daily_estimate, interval_quoted_policy


def compare_intervals(cohort: JSONObject, quotes: JSONObject, plan: JSONObject) -> JSONObject:
    """Fit on available earlier labels and score identical later opportunities.

    Args:
        cohort: Source-verified complete inventory.
        quotes: Source-verified prior quote attachment report.
        plan: Frozen development declaration, not a prospective study.

    Returns:
        Full fitted parameters, row predictions, paired scores and cost scenarios.

    Raises:
        ValueError: IFORE-001 through IFORE-007 for invalid evidence or experiment design.
        TypeError: Required source fields are malformed.
    """
    require_interval_comparison_plan(plan, quotes)
    rows, excluded = interval_observations(cohort, quotes)
    train, test, purged = split_interval_rows(rows, require_int(plan, "train_end_ts"))
    model = fit_interval_model(train)
    predictions: dict[str, tuple[float, ...]] = {
        "raw_market": tuple((row["bid"] + row["ask"]) / 20000 for row in test),
        "earlier_asset_base_rate": tuple(
            model["bitcoin_base"] if row["series"] == "KXBTC15M" else model["ether_base"]
            for row in test
        ),
        "logistic_recalibration": tuple(predict_interval_model(model, row) for row in test),
    }
    label_values: list[int] = [row["yes"] for row in test]
    labels: NDArray[np.int64] = np.asarray(label_values, dtype=np.int64)
    scores: dict[str, tuple[float, ...]] = {}
    arms: JSONObject = {}
    for name, probabilities in predictions.items():
        array: NDArray[np.float64] = np.asarray(probabilities, dtype=np.float64)
        loss = float_values(brier_per_row(labels, array))
        scores[name] = loss
        encoded: list[JSONValue] = list(probabilities)
        policies: list[JSONValue] = [
            interval_quoted_policy(test, probabilities, cost) for cost in (0, 100, 200)
        ]
        arms[name] = {
            "brier": sum(loss) / len(test),
            "log_loss": sum(float_values(log_loss_per_row(labels, array))) / len(test),
            "predictions": encoded,
            "cost_scenarios": policies,
        }
    paired: JSONObject = {}
    for name in ("earlier_asset_base_rate", "logistic_recalibration"):
        differences = tuple(a - b for a, b in zip(scores[name], scores["raw_market"], strict=True))
        paired[name] = interval_daily_estimate(differences, test)
    encoded_rows: list[JSONValue] = [encode_interval_row(row) for row in test]
    train_ids: list[JSONValue] = [row["ticker"] for row in train]
    exclusions: list[JSONValue] = list(excluded)
    return {
        "schema_version": 1,
        "plan": plan,
        "model": encode_interval_fit(model),
        "training_count": len(train),
        "evaluation_count": len(test),
        "training_ids": train_ids,
        "purged_training_labels": purged,
        "exclusions": exclusions,
        "evaluation_rows": encoded_rows,
        "arms": arms,
        "paired_daily_brier_minus_market": paired,
        "orders_submitted": 0,
        "interpretation": (
            "Exploratory historical development comparison, not an untouched future test. "
            "No hyperparameter search; logistic C=1, market logit and asset indicator only. "
            "Brier/log loss averages weight opportunities; paired intervals weight dates equally. "
            "Both assets share dates; seven-lag serial uncertainty does not prove independence. "
            "Settlement labels/times are capture-time snapshots; earlier revisions are unverified. "
            "Costs assume a 0.07 quadratic fee with per-order cent rounding and fixed friction. "
            "Hypothetical quote payoffs do not establish executable fills or a profitable edge."
        ),
    }
