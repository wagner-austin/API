"""Earlier-only capacity selection with complete inner validation evidence."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from platform_core.json_utils import JSONObject, JSONValue

from ..backends.lightgbm.backend import _LGBModelProtocol
from .boosting_model import boosting_parameters, feature_matrix, fit_boosting
from .boosting_scores import daily_means
from .forecast_model import fit_logit, float_values, predict_logit
from .forecast_records import ForecastRow


def prediction_diagnostic(
    rows: Sequence[ForecastRow], probabilities: tuple[float, ...]
) -> JSONObject:
    """Retain aligned probabilities and equally weighted daily squared errors.

    Args:
        rows: Nonempty validated observations.
        probabilities: Model predictions in observation order.

    Returns:
        Exact probabilities, date losses, mean loss and probability range.

    Raises:
        ValueError: Prediction and observation counts differ.
    """
    losses = tuple((p - row["yes"]) ** 2 for p, row in zip(probabilities, rows, strict=True))
    daily = daily_means(losses, rows)
    encoded: list[JSONValue] = list(probabilities)
    return {
        "predictions": encoded,
        "date_losses": list(daily),
        "date_mean_brier": sum(daily) / len(daily),
        "minimum_probability": min(probabilities),
        "maximum_probability": max(probabilities),
    }


def select_boosting(
    rows: Sequence[ForecastRow], artifact: Path
) -> tuple[_LGBModelProtocol, bool, JSONObject]:
    """Select on earlier validation dates, then refit using available training rows.

    Args:
        rows: Outer training rows already purged against outer decision time.
        artifact: New selected model path; candidate artifacts share its directory.

    Returns:
        Selected refitted model, weather-feature flag and full selection evidence.

    Raises:
        ValueError: BOOST-008 for fewer than twenty training dates or no inner labels.
        FileExistsError: Any native artifact already exists.
        OSError: Native evidence cannot be written.
    """
    dates = sorted({row["event_day"] for row in rows})
    if len(dates) < 20:
        raise ValueError("[BOOST-008] selection requires twenty earlier event dates")
    validation = tuple(row for row in rows if row["event_day"] >= dates[-10])
    first_decision = min(row["decision_ms"] for row in validation)
    candidates = tuple(row for row in rows if row["event_day"] < dates[-10])
    training = tuple(row for row in candidates if row["settled_ms"] < first_decision)
    if not training:
        raise ValueError("[BOOST-008] no inner training labels available before validation")
    records: list[JSONValue] = []
    ranking: list[tuple[float, int, int, bool]] = []
    for rounds in (20, 100, 400):
        for leaves in (4, 8):
            for weather in (False, True):
                name = f"{artifact.stem}-r{rounds}-l{leaves}-w{int(weather)}.txt"
                model = fit_boosting(training, artifact.parent / name, rounds, leaves, weather)
                train_p = float_values(model.booster_.predict(feature_matrix(training, weather)))
                valid_p = float_values(model.booster_.predict(feature_matrix(validation, weather)))
                daily = daily_means(
                    tuple((p - r["yes"]) ** 2 for p, r in zip(valid_p, validation, strict=True)),
                    validation,
                )
                ranking.append((sum(daily) / len(daily), rounds, leaves, weather))
                records.append(
                    {
                        "artifact": name,
                        "parameters": boosting_parameters(rounds, leaves),
                        "weather": weather,
                        "training": prediction_diagnostic(training, train_p),
                        "validation": prediction_diagnostic(validation, valid_p),
                    }
                )
    _, rounds, leaves, weather = min(ranking)
    logistic: JSONObject = {}
    for augmented in (False, True):
        baseline = fit_logit(training, augmented)
        logistic["weather" if augmented else "market"] = prediction_diagnostic(
            validation, predict_logit(baseline, validation)
        )
    evidence: JSONObject = {
        "training_tickers": [row["ticker"] for row in training],
        "validation_tickers": [row["ticker"] for row in validation],
        "unavailable_training_labels": len(candidates) - len(training),
        "latest_training_settlement_ms": max(row["settled_ms"] for row in training),
        "earliest_validation_decision_ms": first_decision,
        "candidates": records,
        "logistic_validation": logistic,
        "selected_parameters": boosting_parameters(rounds, leaves),
        "selected_weather": weather,
    }
    return fit_boosting(rows, artifact, rounds, leaves, weather), weather, evidence
