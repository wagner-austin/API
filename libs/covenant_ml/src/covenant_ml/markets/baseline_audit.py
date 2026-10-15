"""Offline arithmetic audit of saved baseline forecasts, with no model fitting."""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.json_utils import (
    JSONObject,
    narrow_json_to_dict,
    require_dict,
    require_float,
    require_int,
    require_list,
)

from .boosting_scores import daily_means, paired_daily_contrast, score_predictions
from .forecast_model import decode_fitted_logit, predict_logit
from .forecast_records import ForecastRow, decode_comparison_plan, decode_forecast_row


def _policy_summary(rows: Sequence[ForecastRow], scores: JSONObject) -> JSONObject:
    """Retain the original costs and identify dates with actual simulated decisions.

    Args:
        rows: Aligned verified evaluation observations.
        scores: Recomputed pooled score object.

    Returns:
        Original per-arm policies with distinct traded event dates.

    Raises:
        ValueError: Policy actions and observations have different lengths.
    """
    arms = require_dict(scores, "arms")
    policies: JSONObject = {}
    for name in arms:
        policy = require_dict(require_dict(arms, name), "policy")
        dates = sorted(
            {
                row["event_day"]
                for row, action in zip(rows, require_list(policy, "actions"), strict=True)
                if action != "pass"
            }
        )
        policies[name] = {"policy": policy, "trade_dates": list(dates)}
    return policies


def audit_baseline(document: JSONObject) -> JSONObject:
    """Replay cached forecasts and costs before comparing logistic with market.

    Args:
        document: Source boosting experiment whose provenance was verified at the edge.

    Returns:
        Equal-date paired contrast, verified scores, trade dates and sensitivity limits.

    Raises:
        ValueError: BASELINE-002 for absent folds; BASELINE-003 for inconsistent
            predictions; BASELINE-004 for changed scores; BASELINE-005 for duplicate
            contracts or inconsistent recorded chronological boundaries.
        TypeError: Required source fields are missing or malformed.
    """
    folds = require_list(document, "folds")
    if not folds:
        raise ValueError("[BASELINE-002] require saved chronological evaluation folds")
    rows: list[ForecastRow] = []
    predictions: dict[str, tuple[float, ...]] = {
        "raw_market": (),
        "logistic": (),
        "boosting": (),
    }
    identities: set[str] = set()
    previous_end = ""
    for value in folds:
        fold = narrow_json_to_dict(value)
        plan = decode_comparison_plan(require_dict(fold, "plan"))
        observations = tuple(
            decode_forecast_row(narrow_json_to_dict(row))
            for row in require_list(fold, "evaluation_rows")
        )
        if (
            not observations
            or plan["train_end"] < previous_end
            or any(not plan["train_end"] < row["event_day"] <= plan["end"] for row in observations)
            or require_int(fold, "earliest_evaluation_decision_ms")
            != min(row["decision_ms"] for row in observations)
            or require_int(fold, "latest_training_settlement_ms")
            >= min(row["decision_ms"] for row in observations)
        ):
            raise ValueError("[BASELINE-005] saved fold violates recorded chronological boundaries")
        previous_end = plan["end"]
        for row in observations:
            if row["ticker"] in identities:
                raise ValueError("[BASELINE-005] evaluation contract is repeated")
            identities.add(row["ticker"])
        scores = require_dict(fold, "scores")
        arms = require_dict(scores, "arms")
        saved = {
            name: tuple(
                require_float({"p": probability}, "p")
                for probability in require_list(require_dict(arms, name), "predictions")
            )
            for name in predictions
        }
        if saved["raw_market"] != tuple(
            (row["bid"] + row["ask"]) / 20000 for row in observations
        ) or saved["logistic"] != predict_logit(
            decode_fitted_logit(require_dict(fold, "logistic_model")), observations
        ):
            raise ValueError(
                "[BASELINE-003] saved predictions disagree with quotes or fitted model"
            )
        if score_predictions(observations, saved) != scores:
            raise ValueError("[BASELINE-004] saved fold scores or policy accounting disagree")
        rows.extend(observations)
        for name in predictions:
            predictions[name] += saved[name]
    pooled = score_predictions(rows, predictions)
    if pooled != require_dict(document, "scores"):
        raise ValueError("[BASELINE-004] pooled predictions, scores or policy accounting disagree")
    differences = tuple(
        (logistic - row["yes"]) ** 2 - (market - row["yes"]) ** 2
        for row, logistic, market in zip(
            rows, predictions["logistic"], predictions["raw_market"], strict=True
        )
    )
    comparison = paired_daily_contrast(daily_means(differences, rows))
    interval = require_dict(comparison, "date_clustered")
    return {
        "contracts": len(rows),
        "event_dates": pooled["event_dates"],
        "logistic_minus_raw_market": comparison,
        "verified_scores": pooled,
        "policies": _policy_summary(rows, pooled),
        "date_independence_ci_half_width": (
            require_float(interval, "ci_high") - require_float(interval, "ci_low")
        )
        / 2,
        "interpretation": (
            "Exploratory audit of already examined historical dates, with no fitting or selection. "
            "Negative Brier differences favor logistic. The confidence half-width is a precision "
            "diagnostic, not an 80%-power minimum detectable effect. Dates can be dependent; "
            "the seven-lag sensitivity has only the recorded dates and no fresh holdout. "
            "Quoted taker profits retain original fees and hurdle; actual fills are unverified."
        ),
    }
