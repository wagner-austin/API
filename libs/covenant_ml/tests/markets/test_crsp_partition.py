"""Partition bounds checked on rows decoded by the existing reader."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, require_str

from covenant_ml.markets.crsp_extract import CrspSources, load_crsp_sources
from covenant_ml.markets.crsp_partition import check_crsp_partition
from covenant_ml.markets.crsp_queries import crsp_scopes

from ._crsp_database import STRADDLE, database_protocol, database_rows, partition_rows
from ._crsp_fixtures import write_sources
from ._crsp_study_fixtures import study_protocol


def _rows(partition: int) -> dict[str, list[JSONObject]]:
    protocol = database_protocol()
    return partition_rows(database_rows(), protocol, crsp_scopes(protocol)[partition])


def _load(directory: Path, rows: dict[str, list[JSONObject]]) -> CrspSources:
    directory.mkdir()
    path, _ = write_sources(directory, rows, study_protocol())
    return load_crsp_sources(path)


def test_both_partitions_of_the_synthetic_database_pass(tmp_path: Path) -> None:
    validation, final = crsp_scopes(database_protocol())
    check_crsp_partition(_load(tmp_path / "v", _rows(0)), validation, 20000)
    check_crsp_partition(_load(tmp_path / "f", _rows(1)), final, 20000)


def test_final_evidence_cannot_pass_as_train_validation(tmp_path: Path) -> None:
    validation, _ = crsp_scopes(database_protocol())
    with pytest.raises(ValueError, match=r"\[CRSP-012\] dlycaldt lies outside"):
        check_crsp_partition(_load(tmp_path / "f", _rows(1)), validation, 20000)


def test_an_unclipped_membership_end_is_refused(tmp_path: Path) -> None:
    rows = _rows(0)
    rows["membership"][0] = {**rows["membership"][0], "nameendt": "2024-12-31"}
    with pytest.raises(ValueError, match=r"\[CRSP-012\] nameendt lies outside"):
        check_crsp_partition(
            _load(tmp_path / "v", rows), crsp_scopes(database_protocol())[0], 20000
        )


def test_a_terminal_value_dated_after_the_boundary_is_refused(tmp_path: Path) -> None:
    rows = _rows(0)
    late = [row for row in database_rows()["delists"] if row["permno"] == STRADDLE]
    rows["delists"] += late
    with pytest.raises(ValueError, match=r"\[CRSP-012\] delamtdt lies outside"):
        check_crsp_partition(
            _load(tmp_path / "v", rows), crsp_scopes(database_protocol())[0], 20000
        )


def test_one_row_beyond_the_limit_means_truncation(tmp_path: Path) -> None:
    rows = _rows(0)
    sources = _load(tmp_path / "v", rows)
    validation = crsp_scopes(database_protocol())[0]
    check_crsp_partition(sources, validation, len(rows["returns"]))
    with pytest.raises(ValueError, match=r"\[CRSP-012\] returns exceeds its row limit"):
        check_crsp_partition(sources, validation, len(rows["returns"]) - 1)


def test_empty_required_relations_are_refused_but_terminal_events_may_be_empty(
    tmp_path: Path,
) -> None:
    validation = crsp_scopes(database_protocol())[0]
    rows = _rows(0)
    rows["delists"] = []
    check_crsp_partition(_load(tmp_path / "a", rows), validation, 20000)
    rows["membership"] = []
    with pytest.raises(ValueError, match=r"\[CRSP-012\] membership is empty"):
        check_crsp_partition(_load(tmp_path / "b", rows), validation, 20000)


def test_calendar_must_span_exactly_and_completely(tmp_path: Path) -> None:
    validation = crsp_scopes(database_protocol())[0]
    short = _rows(0)
    short["calendar"] = [
        row for row in short["calendar"] if require_str(row, "caldt") < "2020-02-01"
    ]
    with pytest.raises(ValueError, match=r"\[CRSP-012\] the calendar does not span"):
        check_crsp_partition(_load(tmp_path / "a", short), validation, 20000)
    holed = _rows(0)
    holed["calendar"] = [row for row in holed["calendar"] if row["caldt"] != "2018-05-16"]
    with pytest.raises(ValueError, match=r"\[CRSP-002\] calendar dates must be unique"):
        check_crsp_partition(_load(tmp_path / "b", holed), validation, 20000)
