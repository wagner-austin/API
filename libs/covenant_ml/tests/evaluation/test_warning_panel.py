"""Real as-of selection, censoring and chronological borrower exclusion tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
)

from covenant_ml.evaluation.warning_panel import (
    WarningRow,
    decode_warning_row,
    encode_warning_row,
    require_warning_split,
    select_warning_disclosure,
    warning_label,
)
from covenant_ml.evaluation.warning_records import (
    decode_warning_disclosure,
    encode_warning_disclosure,
)

from .test_warning_records import disclosure_document, followup_document


def row_document(borrower: str = "A", shift: int = 0) -> JSONObject:
    """Construct explicit synthetic evidence on a translated time axis."""
    disclosure = disclosure_document()
    disclosure.update(borrower=borrower, period_end_ms=10 + shift, available_ms=20 + shift)
    followup = followup_document()
    followup.update(
        borrower=borrower,
        coverage_start_ms=shift,
        observed_through_ms=100 + shift,
        available_ms=110 + shift,
        first_breach_ms=70 + shift,
    )
    return {
        "cutoff_ms": 30 + shift,
        "horizon_end_ms": 100 + shift,
        "disclosure": disclosure,
        "followup": followup,
    }


def row(borrower: str = "A", shift: int = 0) -> WarningRow:
    """Validate an opportunity through the production decoder."""
    return decode_warning_row(row_document(borrower, shift))


def test_row_roundtrip_and_archived_file_replay(tmp_path: Path) -> None:
    training, evaluation = row(), row("B", 200)
    path = tmp_path / "evidence.json"
    path.write_text(dump_json_str(encode_warning_row(training)), encoding="utf-8")
    replay = decode_warning_row(narrow_json_to_dict(load_json_str(path.read_text("utf-8"))))
    assert replay == training
    assert encode_warning_row(replay) == row_document()
    assert warning_label(replay) == 1
    require_warning_split((replay,), (evaluation,), 310)
    assert warning_label(evaluation) == 1
    path.write_text('{"cutoff_ms":30}', encoding="utf-8")
    with pytest.raises(TypeError, match="horizon_end_ms"):
        decode_warning_row(narrow_json_to_dict(load_json_str(path.read_text("utf-8"))))


def test_no_lookahead_future_vintages_and_other_borrowers() -> None:
    original = decode_warning_disclosure(disclosure_document())
    revision_doc = disclosure_document()
    revision_doc.update(available_ms=40, debt_to_ebitda=999.0)
    revision = decode_warning_disclosure(revision_doc)
    future_doc = disclosure_document()
    future_doc.update(period_end_ms=50, available_ms=60, current_ratio=123.0)
    future = decode_warning_disclosure(future_doc)
    other_doc = disclosure_document()
    other_doc.update(borrower="B", available_ms=30)
    other = decode_warning_disclosure(other_doc)
    evidence = (future, original, other, revision)
    assert select_warning_disclosure(evidence, "A", 30) == original
    assert select_warning_disclosure(tuple(reversed(evidence)), "A", 30) == original
    assert select_warning_disclosure(evidence, "A", 40) == revision
    assert select_warning_disclosure(evidence, "A", 60) == future
    assert select_warning_disclosure((original,), "A", 20) == original
    future_doc.update(current_ratio=-999.0)
    altered = decode_warning_disclosure(future_doc)
    assert select_warning_disclosure((original, altered, revision), "A", 30) == original


def test_late_revision_of_older_period_does_not_replace_latest_period() -> None:
    latest_doc = disclosure_document()
    latest_doc.update(period_end_ms=20, available_ms=25)
    old_revision_doc = disclosure_document()
    old_revision_doc.update(available_ms=30)
    latest = decode_warning_disclosure(latest_doc)
    old_revision = decode_warning_disclosure(old_revision_doc)
    assert select_warning_disclosure((latest, old_revision), "A", 30) == latest


def test_selection_rejects_unavailable_and_ambiguous_evidence() -> None:
    disclosure = decode_warning_disclosure(disclosure_document())
    with pytest.raises(ValueError, match=r"WARN-006.*no disclosure"):
        select_warning_disclosure((disclosure,), "A", 19)
    with pytest.raises(ValueError, match=r"WARN-006.*multiple"):
        select_warning_disclosure((disclosure, disclosure), "A", 20)


@pytest.mark.parametrize(
    "breach,through,expected",
    [
        (70, 100, 1),
        (100, 100, 1),
        (None, 100, 0),
        (101, 101, 0),
        (None, 99, None),
        (70, 70, 1),
    ],
)
def test_labels_keep_horizon_boundaries_and_censoring(
    breach: int | None,
    through: int,
    expected: int | None,
) -> None:
    document = row_document()
    followup = require_dict(document, "followup")
    followup.update(first_breach_ms=breach, observed_through_ms=through)
    assert warning_label(decode_warning_row(document)) == expected


def test_outcome_mutation_cannot_change_predictors() -> None:
    original = row()
    document = encode_warning_row(original)
    require_dict(document, "followup")["first_breach_ms"] = None
    changed = decode_warning_row(document)
    assert changed["disclosure"] == original["disclosure"]
    assert encode_warning_disclosure(changed["disclosure"]) == disclosure_document()
    assert warning_label(original) == 1
    assert warning_label(changed) == 0


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("", "cutoff_ms", 19),
        ("", "horizon_end_ms", 30),
        ("followup", "borrower", "B"),
        ("followup", "coverage_start_ms", 31),
        ("followup", "observed_through_ms", 29),
        ("followup", "first_breach_ms", 30),
        ("followup", "first_breach_ms", 20),
    ],
)
def test_rows_refuse_invalid_opportunities(section: str, key: str, value: JSONValue) -> None:
    document = row_document()
    target = require_dict(document, section) if section else document
    if key == "observed_through_ms":
        target["first_breach_ms"] = None
    target[key] = value
    with pytest.raises(ValueError, match="WARN-007"):
        decode_warning_row(document)


@pytest.mark.parametrize(
    "training,evaluation",
    [
        ((), (row("B", 200),)),
        ((row(),), ()),
        ((row(),), (row("A", 200),)),
        ((row(), row()), (row("B", 200),)),
        ((row(),), (row("B", 200), row("B", 200))),
    ],
)
def test_split_refuses_missing_duplicate_or_shared_borrowers(
    training: tuple[WarningRow, ...],
    evaluation: tuple[WarningRow, ...],
) -> None:
    with pytest.raises(ValueError, match="WARN-008"):
        require_warning_split(training, evaluation, 1000)


def test_split_requires_one_predeclared_horizon() -> None:
    document = row_document("B", 200)
    document["horizon_end_ms"] = 301
    with pytest.raises(ValueError, match=r"WARN-008.*horizon"):
        require_warning_split((row(),), (decode_warning_row(document),), 1000)


def test_split_refuses_unmatured_labels_even_for_early_positive() -> None:
    document = row_document()
    require_dict(document, "followup").update(observed_through_ms=70, available_ms=75)
    with pytest.raises(ValueError, match=r"WARN-009.*cutoff"):
        require_warning_split((decode_warning_row(document),), (row("B", 60),), 1000)


@pytest.mark.parametrize("shift", [79, 80])
def test_training_outcome_must_be_known_strictly_before_first_evaluation(shift: int) -> None:
    with pytest.raises(ValueError, match=r"WARN-009.*cutoff"):
        require_warning_split((row(),), (row("B", shift),), 1000)


def test_training_evaluation_and_censoring_boundaries() -> None:
    require_warning_split((row(),), (row("B", 81),), 191)
    assert warning_label(row("B", 81)) == 1
    with pytest.raises(ValueError, match=r"WARN-009.*cutoff"):
        require_warning_split((row(),), (row("B", 81),), 190)
    document = row_document()
    require_dict(document, "followup").update(first_breach_ms=None, observed_through_ms=99)
    with pytest.raises(ValueError, match=r"WARN-009.*censored"):
        require_warning_split((decode_warning_row(document),), (row("B", 200),), 1000)
