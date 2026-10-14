"""Chronological real-learner tests with later-outcome mutation and JSON replay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
)

from covenant_ml.evaluation.warning_experiment import (
    decode_warning_evaluation,
    encode_warning_evaluation,
    evaluate_warning_experiment,
)
from covenant_ml.evaluation.warning_folds import (
    WarningFold,
    decode_warning_fold,
    encode_warning_fold,
    require_warning_study,
)
from covenant_ml.evaluation.warning_learning import (
    WarningLearnerFactory,
    create_warning_logistic,
    predict_warning_probabilities,
    require_warning_distribution,
    require_warning_probabilities,
    warning_features,
    warning_labels,
)
from covenant_ml.evaluation.warning_panel import decode_warning_row, encode_warning_row
from covenant_ml.evaluation.warning_rules import WarningRatioRule
from covenant_ml.evaluation.warning_study import (
    decode_warning_forecast,
    encode_warning_forecast,
    replay_warning_calibration,
    run_warning_study,
)

from .warning_testing import cohort, study


def candidates() -> dict[str, WarningLearnerFactory]:
    """Real fixed ratio rules and two real logistic regularization choices."""
    return {
        "ratio": WarningRatioRule(2.5, 2.0, 1.0),
        "logistic-weak": create_warning_logistic(0.1),
        "logistic-strong": create_warning_logistic(10.0),
    }


def test_real_experiment_replays_and_holdout_outcomes_never_change_fit(tmp_path: Path) -> None:
    folds, final = study()
    selected, scores, results = evaluate_warning_experiment(
        candidates(), folds, final, 1340, 1, 70, 5
    )
    assert set(scores) == set(results) == set(candidates())
    assert selected == min((loss, name) for name, loss in scores.items())[1]
    assert all(0 <= value <= 1 for value in scores.values())
    for result in results.values():
        document = encode_warning_evaluation(result)
        path = tmp_path / "result.json"
        path.write_text(dump_json_str(document), encoding="utf-8")
        assert (
            decode_warning_evaluation(narrow_json_to_dict(load_json_str(path.read_text("utf-8"))))
            == result
        )
        assert replay_warning_calibration(result["forecast"]) == result["forecast"]["calibrated"]
        assert result["raw_alerts"]["incidents"] == result["calibrated_alerts"]["incidents"] == 2
        assert result["raw_alerts"]["opportunities"] == 8
        assert sum(result["raw_reliability"]["counts"]) == 8
    altered = []
    for row in final["validation"]:
        document = encode_warning_row(row)
        require_dict(document, "followup")["first_breach_ms"] = None
        altered.append(decode_warning_row(document))
    changed = WarningFold(
        fitting=final["fitting"], calibration=final["calibration"], validation=tuple(altered)
    )
    chosen_again, scores_again, forecasts_again = run_warning_study(
        candidates(), folds, changed, 1340
    )
    assert (chosen_again, scores_again) == (selected, scores)
    assert forecasts_again == {name: result["forecast"] for name, result in results.items()}
    invalid = encode_warning_evaluation(results[selected])
    require_dict(invalid, "raw_reliability")["counts"] = [1, 1, 1, 1, 1]
    require_dict(invalid, "raw_reliability")["positives"] = [0, 0, 0, 0, 0]
    require_dict(invalid, "raw_reliability")["probability_sums"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="WARN-029"):
        decode_warning_evaluation(invalid)


def test_fold_roundtrip_and_lexical_selection_tie() -> None:
    folds, final = study()
    assert decode_warning_fold(encode_warning_fold(final)) == final
    factories: dict[str, WarningLearnerFactory] = {
        "z": create_warning_logistic(1),
        "a": create_warning_logistic(1),
    }
    chosen, scores, results = run_warning_study(factories, folds, final, 1340)
    assert chosen == "a"
    assert scores["a"] == scores["z"]
    assert results["a"] == results["z"]
    altered = encode_warning_forecast(results["a"])
    altered["calibrated"] = [0.5] * len(results["a"]["raw"])
    with pytest.raises(ValueError, match=r"WARN-019.*disagree"):
        decode_warning_forecast(altered)


def test_calibration_can_harm_and_the_report_exposes_it() -> None:
    folds, final = study()
    inverted = []
    for row in final["calibration"]:
        document = encode_warning_row(row)
        followup = require_dict(document, "followup")
        followup["first_breach_ms"] = (
            row["cutoff_ms"] + 40 if followup["first_breach_ms"] is None else None
        )
        inverted.append(decode_warning_row(document))
    misleading = WarningFold(
        fitting=final["fitting"], calibration=tuple(inverted), validation=final["validation"]
    )
    _, _, results = evaluate_warning_experiment(candidates(), folds, misleading, 1340, 1, 0, 5)
    ratio = results["ratio"]
    assert ratio["raw_reliability"]["brier"] == 0.0
    assert ratio["calibrated_reliability"]["brier"] > 0.9
    assert ratio["raw_alerts"]["detected"] == 1
    assert ratio["calibrated_alerts"]["detected"] == 0
    damaged = encode_warning_evaluation(ratio)
    require_dict(damaged, "raw_alerts")["exposure_ms"] = 1
    with pytest.raises(ValueError, match="WARN-029"):
        decode_warning_evaluation(damaged)


@pytest.mark.parametrize("case", ["empty", "order", "reserved", "late"])
def test_development_boundary_rejections(case: str) -> None:
    folds, final = study()
    if case == "empty":
        folds = ()
    elif case == "order":
        folds = tuple(reversed(folds))
    elif case == "reserved":
        folds = (
            WarningFold(
                fitting=cohort("hold", 0),
                calibration=cohort("cal", 200),
                validation=cohort("val", 400),
            ),
        )
    else:
        folds = (
            WarningFold(
                fitting=cohort("x", 0), calibration=cohort("y", 200), validation=cohort("z", 1100)
            ),
        )
    with pytest.raises(ValueError, match=r"WARN-(008|009|015)"):
        require_warning_study(folds, final, 1340)


@pytest.mark.parametrize("names", [(), ("one",), ("", "valid"), (" bad", "valid")])
def test_candidate_names(names: tuple[str, ...]) -> None:
    folds, final = study()
    with pytest.raises(ValueError, match="WARN-021"):
        run_warning_study({name: create_warning_logistic(1) for name in names}, folds, final, 1340)


def test_one_class_calibration_is_not_silently_replaced() -> None:
    folds, final = study()
    one_class = WarningFold(
        fitting=folds[0]["fitting"],
        calibration=folds[0]["calibration"][:2],
        validation=folds[0]["validation"],
    )
    with pytest.raises(ValueError, match="WARN-020"):
        run_warning_study(candidates(), (one_class,), final, 1340)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_logistic_settings(value: float) -> None:
    with pytest.raises(ValueError, match="WARN-018"):
        create_warning_logistic(value)


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_iteration_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="WARN-018"):
        create_warning_logistic(1, limit)


def test_real_solver_failure_one_class_and_censored_labels() -> None:
    rows = cohort("fit", 0)
    features = warning_features(tuple(row["disclosure"] for row in rows))
    labels = warning_labels(rows)
    with pytest.raises(ValueError, match=r"WARN-018.*both"):
        create_warning_logistic(1)(features[:2], labels[:2])
    with pytest.raises(ValueError, match=r"WARN-018.*exhausted"):
        create_warning_logistic(1, 1)(features, labels)
    document = encode_warning_row(rows[0])
    require_dict(document, "followup")["observed_through_ms"] = 50
    with pytest.raises(ValueError, match="WARN-016"):
        warning_labels((decode_warning_row(document),))
    model = create_warning_logistic(1)(features, labels)
    predicted = predict_warning_probabilities(model, tuple(row["disclosure"] for row in rows))
    assert predicted.shape == (4,)
    assert float(predicted.flat[0]) < float(predicted.flat[3])


@pytest.mark.parametrize(
    "values", [((0.2, 0.2),), ((-0.1, 1.1),), ((float("nan"), 0.5),), ((0.1, 0.2, 0.7),)]
)
def test_invalid_binary_distribution(values: tuple[tuple[float, ...], ...]) -> None:
    raw: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    with pytest.raises(ValueError, match="WARN-017"):
        require_warning_distribution(raw, 1)


def test_probability_length_is_not_broadcast() -> None:
    values: NDArray[np.float64] = np.asarray((0.2,), dtype=np.float64)
    with pytest.raises(ValueError, match="WARN-017"):
        require_warning_probabilities(values, 2)


def test_ratio_rule_uses_all_declared_thresholds_and_validates_shape() -> None:
    rule = WarningRatioRule(4.0, 2.0, 1.0)
    values: NDArray[np.float64] = np.asarray(
        ((4, 2, 1), (5, 3, 2), (3, 1, 2), (3, 3, 0.5)), dtype=np.float64
    )
    expected: NDArray[np.float64] = np.asarray(((1, 0), (0, 1), (0, 1), (0, 1)), dtype=np.float64)
    assert np.array_equal(rule.predict_proba(values), expected)
    with pytest.raises(ValueError, match="WARN-028"):
        WarningRatioRule(float("nan"), 2, 1)
    with pytest.raises(ValueError, match="WARN-028"):
        rule.predict_proba(values[:, :2])


@pytest.mark.parametrize(
    "key,value",
    [("raw", []), ("raw", [0.2]), ("calibrated", [float("nan"), 0.4]), ("raw", [-0.1, 0.4])],
)
def test_forecast_codec_refuses_invalid_vectors(key: str, value: JSONValue) -> None:
    document: JSONObject = {"raw": [0.1, 0.9], "calibrated": [0.2, 0.8], "calibration_json": "{}"}
    document[key] = value
    with pytest.raises(ValueError, match="WARN-019"):
        decode_warning_forecast(document)
