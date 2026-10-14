"""Hand-derived longitudinal counts and adversarial repeated-alert histories."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONValue, require_dict

from covenant_ml.evaluation.warning_alerts import (
    decode_warning_longitudinal,
    encode_warning_longitudinal,
    evaluate_warning_alerts,
    select_warning_alerts,
)
from covenant_ml.evaluation.warning_panel import WarningRow, decode_warning_row, encode_warning_row

from .test_warning_panel import row_document


def opportunity(borrower: str, cutoff: int, breach: int | None) -> WarningRow:
    """Use a common horizon with a fixed absolute incident date across observations."""
    document = row_document(borrower)
    document.update(cutoff_ms=cutoff, horizon_end_ms=cutoff + 70)
    require_dict(document, "followup").update(
        first_breach_ms=breach, observed_through_ms=200, available_ms=210
    )
    return decode_warning_row(document)


def panel() -> tuple[WarningRow, ...]:
    """Two dates, two borrowers, one incident and overlapping surveillance windows."""
    return (
        opportunity("A", 30, 90),
        opportunity("B", 30, None),
        opportunity("A", 60, 90),
        opportunity("B", 60, None),
    )


def test_repeated_true_alerts_do_not_multiply_events_or_exposure() -> None:
    result = evaluate_warning_alerts(panel(), (0.9, 0.1, 0.9, 0.1), 1, 0, 210)
    # A contributes (30,90] = 60; B contributes union (30,130] = 100.
    assert encode_warning_longitudinal(result) == {
        "opportunities": 4,
        "alerts": 2,
        "true_alerts": 2,
        "incidents": 1,
        "detected": 1,
        "exposure_ms": 160,
        "lead_times_ms": [60],
    }
    assert decode_warning_longitudinal(encode_warning_longitudinal(result)) == result
    reversed_ranking = evaluate_warning_alerts(panel(), (0.1, 0.9, 0.1, 0.9), 1, 0, 210)
    assert reversed_ranking["true_alerts"] == reversed_ranking["detected"] == 0
    assert reversed_ranking["incidents"] == 1
    assert reversed_ranking["alerts"] == 2
    suppressed = evaluate_warning_alerts(panel(), (0.9, 0.1, 0.9, 0.1), 1, 70, 210)
    assert suppressed["alerts"] == 2
    assert suppressed["true_alerts"] == suppressed["detected"] == 1
    assert suppressed["lead_times_ms"] == (60,)
    permuted = tuple(reversed(panel()))
    assert evaluate_warning_alerts(permuted, (0.1, 0.9, 0.1, 0.9), 1, 0, 210) == result


def test_cooldown_boundary_and_fully_suppressed_batch() -> None:
    scores = {60: {"A": 0.8}, 30: {"A": 0.9}, 100: {"A": 0.2}}
    assert select_warning_alerts(scores, 1, 70) == ((30, "A"), (100, "A"))
    assert select_warning_alerts(scores, 1, 30) == ((30, "A"), (60, "A"), (100, "A"))
    assert select_warning_alerts({30: {"A": 0.8, "B": 0.8}}, 1, 0) == ((30, "A"),)


def test_disjoint_windows_and_breach_outside_scored_horizon() -> None:
    rows = (opportunity("A", 30, 180), opportunity("A", 110, 180))
    result = evaluate_warning_alerts(rows, (0.9, 0.9), 1, 0, 210)
    assert result["exposure_ms"] == 140
    assert result["incidents"] == result["detected"] == result["true_alerts"] == 1
    assert result["lead_times_ms"] == (70,)
    assert result["alerts"] == 2


@pytest.mark.parametrize(
    "case", ["conflicting", "contradictory", "duplicate", "horizon", "length", "empty"]
)
def test_panel_refusals(case: str) -> None:
    rows = panel()
    scores: tuple[float, ...] = (0.9, 0.1, 0.9, 0.1)
    if case == "conflicting":
        rows = (*rows[:2], opportunity("A", 60, 95), rows[3])
    elif case == "contradictory":
        rows = (*rows[:2], opportunity("A", 60, None), rows[3])
    elif case == "duplicate":
        rows = (*rows[:3], rows[2])
    elif case == "horizon":
        document = encode_warning_row(rows[0])
        document["horizon_end_ms"] = 101
        rows = (decode_warning_row(document), *rows[1:])
    elif case == "length":
        scores = (0.9,)
    else:
        rows = ()
        scores = ()
    with pytest.raises(ValueError, match=r"WARN-(013|024|025)"):
        evaluate_warning_alerts(rows, scores, 1, 0, 210)


@pytest.mark.parametrize("budget,cooldown", [(0, 0), (True, 0), (1, -1), (1, True)])
def test_invalid_policy(budget: int, cooldown: int) -> None:
    with pytest.raises(ValueError, match="WARN-023"):
        select_warning_alerts({30: {"A": 0.1}}, budget, cooldown)


def test_empty_history_and_invalid_time_or_suppressed_score() -> None:
    with pytest.raises(ValueError, match="WARN-023"):
        select_warning_alerts({}, 1, 0)
    with pytest.raises(ValueError, match="WARN-001"):
        select_warning_alerts({-1: {"A": 0.1}}, 1, 0)
    with pytest.raises(ValueError, match="WARN-011"):
        select_warning_alerts({30: {"A": 0.1}, 31: {"A": float("nan")}}, 1, 100)
    with pytest.raises(ValueError, match="WARN-011"):
        select_warning_alerts({30: {"A": True}}, 1, 0)


@pytest.mark.parametrize(
    "key,value",
    [
        ("opportunities", 0),
        ("true_alerts", 3),
        ("detected", 2),
        ("detected", 0),
        ("incidents", -1),
        ("exposure_ms", 0),
        ("lead_times_ms", []),
        ("lead_times_ms", [-1]),
    ],
)
def test_longitudinal_codec_refuses_impossible_counts(key: str, value: JSONValue) -> None:
    document: JSONObject = {
        "opportunities": 4,
        "alerts": 2,
        "true_alerts": 2,
        "incidents": 1,
        "detected": 1,
        "exposure_ms": 160,
        "lead_times_ms": [60],
    }
    document[key] = value
    with pytest.raises(ValueError, match="WARN-022"):
        decode_warning_longitudinal(document)
