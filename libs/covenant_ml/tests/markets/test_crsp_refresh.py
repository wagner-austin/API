"""Real encrypted refresh evidence, private differences and refusal before study use."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Literal

import pytest
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    require_dict,
    require_int,
    require_list,
    require_str,
)
from platform_core.testing import sample_run_fingerprint

from covenant_ml.markets.crsp_cohort_records import (
    decode_crsp_cohort_plan,
    encode_crsp_cohort_plan,
)
from covenant_ml.markets.crsp_diagnostic import capture_crsp_diagnostic
from covenant_ml.markets.crsp_extract import load_crsp_sources
from covenant_ml.markets.crsp_refresh_records import (
    decode_crsp_refresh,
    encode_crsp_refresh,
)
from covenant_ml.markets.crsp_rows import encode_crsp_rows
from covenant_ml.markets.encrypted_storage import (
    decrypted_bundle,
    read_encrypted_bundle,
)
from covenant_ml.markets.research_trial import execute_trial
from covenant_ml.markets.research_trial_records import decode_research_trial

from ._crsp_action_fixtures import ActionDatabase, action_rows, capture_actions, installed_actions
from ._crsp_database import database_protocol, relation_columns
from ._crsp_extraction_fixtures import (
    NOW,
    FakeDatabase,
    database_rows,
    installed,
    limits,
)
from ._crsp_refresh_fixtures import RefreshCase, refresh, refresh_case
from ._crsp_study_fixtures import study_protocol
from ._encrypted_fixtures import evaluator_keys
from ._trial_fixtures import frozen_request


def _refused(case: RefreshCase) -> None:
    assert list(case.cohort.extraction.scratch.iterdir()) == []
    assert not case.destination.exists()


def test_refresh_retains_exact_private_evidence_and_never_needs_the_final_key(
    tmp_path: Path,
) -> None:
    case = refresh_case(tmp_path)
    (tmp_path / "keys" / "final.agekey").unlink()
    database = FakeDatabase(database_rows(), securities=(1,))
    with installed(database):
        result = refresh(case, limits())
    assert result["source"] == "crsp_a_stock"
    assert result["started_at"] == result["completed_at"] == NOW
    assert result["plaintext_removed"] is True
    assert result["cohort_bundle_sha256"] == case.digest
    assert database.calls == [(120, 2_000_000)]
    assert "final/" not in database.driver
    public = dump_json_str(result)
    for forbidden in ("permno", "2019-", "SELECT", "dlyret", "AGE-SECRET-KEY"):
        assert forbidden not in public
    digest = require_str(require_dict(result, "bundle"), "sha256")
    bundle = read_encrypted_bundle(case.destination / "bundle.json", digest)
    assert bundle["manifest"]["name"] == "comparison.json"
    assert all(
        path.name == "bundle.json" or path.suffix == ".age" for path in case.destination.iterdir()
    )
    with decrypted_bundle(
        case.destination / "bundle.json",
        digest,
        partition="train-validation",
        identity=evaluator_keys(tmp_path).acquire("train-validation"),
        scratch=case.cohort.extraction.scratch,
        max_plaintext_bytes=4_000_000,
    ) as path:
        document = load_json_str(path.read_text("utf-8"))
        record = decode_crsp_refresh(document)
        assert encode_crsp_refresh(record) == document
        assert record["cohort"]["plan"] == case.cohort.plan
        assert record["setup_sha256"] is None
        assert record["snapshot"] == "750:750:"
        assert record["driver"]["sha256"] == sha256(database.driver.encode()).hexdigest()
        assert (path.parent / "extract.sql").read_text() == database.driver
        assert (path.parent / "snapshot-first.csv").read_bytes() == (
            path.parent / "snapshot-last.csv"
        ).read_bytes()
        assert "AND permno IN (1)" in (path.parent / "returns.sql").read_text()
        details = require_dict(
            {"value": load_json_str((path.parent / "differences.json").read_text())}, "value"
        )
        for role, original in case.cohort.rows.items():
            expected = len(
                [row for row in original if role == "calendar" or require_int(row, "permno") == 1]
            )
            assert require_dict(details, role) == {
                "unchanged": expected,
                "added": [],
                "removed": [],
                "changed": [],
            }
            assert require_dict(require_dict(result, "rows"), role) == {
                "old": expected,
                "fresh": expected,
                "unchanged": expected,
                "added": 0,
                "removed": 0,
                "changed": 0,
            }
        with pytest.raises(ValueError, match=r"CRSP-006.*exact extract fields"):
            load_crsp_sources(path)
    assert list(case.cohort.extraction.scratch.iterdir()) == []


def test_changed_and_absent_vendor_rows_are_retained_as_encrypted_differences(
    tmp_path: Path,
) -> None:
    case = refresh_case(tmp_path)
    rows = database_rows()
    selected = [row for row in rows["returns"] if require_int(row, "permno") == 1]
    before = dict(selected[0])
    selected[0]["dlyret"] = None
    removed = selected[1]
    rows["returns"].remove(removed)
    with installed(FakeDatabase(rows, securities=(1,))):
        result = refresh(case, limits())
    counts = require_dict(require_dict(result, "rows"), "returns")
    assert (counts["changed"], counts["removed"], counts["added"]) == (1, 1, 0)
    digest = require_str(require_dict(result, "bundle"), "sha256")
    with decrypted_bundle(
        case.destination / "bundle.json",
        digest,
        partition="train-validation",
        identity=evaluator_keys(tmp_path).acquire("train-validation"),
        scratch=case.cohort.extraction.scratch,
        max_plaintext_bytes=4_000_000,
    ) as path:
        differences = require_dict(
            {"value": load_json_str((path.parent / "differences.json").read_text())}, "value"
        )
        daily = require_dict(differences, "returns")
        assert require_list(daily, "changed") == [{"old": before, "fresh": selected[0]}]
        assert require_list(daily, "removed") == [removed]
    assert list(case.cohort.extraction.scratch.iterdir()) == []


def test_synthetic_identity_and_definitions_remain_explicit(tmp_path: Path) -> None:
    case = refresh_case(tmp_path)
    setup = tmp_path / "setup.sql"
    setup.write_text("dsenames(permno) AS (SELECT 1)")
    with installed(FakeDatabase(database_rows(), securities=(1,))):
        result = refresh(case, limits(), setup)
    assert result["source"] == "synthetic"
    digest = require_str(require_dict(result, "bundle"), "sha256")
    with decrypted_bundle(
        case.destination / "bundle.json",
        digest,
        partition="train-validation",
        identity=evaluator_keys(tmp_path).acquire("train-validation"),
        scratch=case.cohort.extraction.scratch,
        max_plaintext_bytes=4_000_000,
    ) as path:
        record = decode_crsp_refresh(load_json_str(path.read_text()))
        assert record["setup_sha256"] == sha256(setup.read_bytes()).hexdigest()
        assert (path.parent / "setup.sql").read_bytes() == setup.read_bytes()
        assert (path.parent / "returns.sql").read_text().startswith("COPY (WITH dsenames")


@pytest.mark.parametrize("failure", ["client", "snapshot", "cohort", "rows", "bytes", "scope"])
def test_refused_refresh_publishes_nothing_and_removes_private_storage(
    tmp_path: Path, failure: str
) -> None:
    case = refresh_case(tmp_path)
    database = FakeDatabase(database_rows(), securities=(1,))
    bounds = limits()
    if failure == "client":
        database.code = 3
    elif failure == "snapshot":
        database.snapshots = ("one,t\n", "two,t\n")
    elif failure == "cohort":
        case = case._replace(digest="f" * 64)
    elif failure == "rows":
        bounds = limits(max_rows=1)
    elif failure == "bytes":
        bounds = limits(max_relation_bytes=1, max_partition_bytes=1)
    else:
        database.securities = None
    with (
        installed(database),
        pytest.raises(ValueError, match=r"CRSP-011|CRSP-012|ENCRYPTED_MANIFEST|CRSP_REFRESH_SCOPE"),
    ):
        refresh(case, bounds)
    _refused(case)


def test_changed_plan_is_refused_before_the_client_runs(tmp_path: Path) -> None:
    case = refresh_case(tmp_path)
    document = encode_crsp_cohort_plan(case.cohort.plan)
    document["audit_sha256"] = "f" * 64
    plan = decode_crsp_cohort_plan(document)
    database = FakeDatabase(database_rows(), securities=(1,))
    with installed(database), pytest.raises(ValueError, match="CRSP_COHORT_LINEAGE"):
        capture_crsp_diagnostic(
            case.cohort.destination / "bundle.json",
            case.digest,
            plan,
            evaluator_keys(tmp_path),
            operation="refresh",
            scratch=case.cohort.extraction.scratch,
            destination=case.destination,
            limits=limits(),
            setup=None,
        )
    assert database.calls == []
    _refused(case)


def test_existing_destination_is_not_replaced(tmp_path: Path) -> None:
    case = refresh_case(tmp_path)
    case.destination.mkdir()
    with pytest.raises(FileExistsError, match="CRSP_DIAGNOSTIC_OUTPUT"):
        refresh(case, limits())
    assert list(case.destination.iterdir()) == []
    assert list(case.cohort.extraction.scratch.iterdir()) == []


def test_total_evidence_budget_includes_queries_metadata_and_differences(tmp_path: Path) -> None:
    case = refresh_case(tmp_path)
    columns = relation_columns(database_protocol())
    sizes = [
        len(
            encode_crsp_rows(
                tuple(row for row in rows if role == "calendar" or require_int(row, "permno") == 1),
                columns[role],
            )
        )
        for role, rows in case.cohort.rows.items()
    ]
    with (
        installed(FakeDatabase(database_rows(), securities=(1,))),
        pytest.raises(ValueError, match="CRSP_DIAGNOSTIC_LIMIT"),
    ):
        refresh(case, limits(max_relation_bytes=max(sizes), max_partition_bytes=sum(sizes)))
    _refused(case)


def test_dangling_output_link_is_refused_before_query(tmp_path: Path) -> None:
    case = refresh_case(tmp_path)
    case.destination.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError, match="CRSP_DIAGNOSTIC_OUTPUT"):
        refresh(case, limits())
    assert case.destination.is_symlink()
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("operation", ["refresh", "actions"])
def test_final_bundle_cannot_be_opened_as_a_diagnostic(
    tmp_path: Path, operation: Literal["refresh", "actions"]
) -> None:
    case = refresh_case(tmp_path)
    final = case.cohort.extraction.output / "final" / "bundle.json"
    with pytest.raises(ValueError, match="ENCRYPTED_ACCESS"):
        capture_crsp_diagnostic(
            final,
            sha256(final.read_bytes()).hexdigest(),
            case.cohort.plan,
            evaluator_keys(tmp_path),
            operation=operation,
            scratch=case.cohort.extraction.scratch,
            destination=case.destination,
            limits=limits(),
            setup=None,
        )
    _refused(case)


@pytest.mark.parametrize("operation", ["refresh", "actions"])
def test_actual_study_trial_refuses_the_diagnostic_envelope(tmp_path: Path, operation: str) -> None:
    case = refresh_case(tmp_path)
    if operation == "refresh":
        with installed(FakeDatabase(database_rows(), securities=(1,))):
            refresh(case, limits())
    else:
        with installed_actions(ActionDatabase(action_rows())):
            capture_actions(case)
    evaluation: JSONObject = {
        "design": study_protocol(),
        "stopping_months": 6,
        "fold_months": 12,
        "confidence": 0.9,
    }
    document = frozen_request(
        evaluation, case.destination / "bundle.json", ["momentum", "liquidity"]
    )
    output = tmp_path / "attempt" / "result.json"
    with pytest.raises(ValueError, match=r"CRSP-006.*exact extract fields"):
        execute_trial(
            decode_research_trial(document),
            sample_run_fingerprint(image_digest="d" * 64),
            "c" * 64,
            output,
            keys=evaluator_keys(tmp_path),
            scratch=case.cohort.extraction.scratch,
        )
    assert not output.exists()
    assert list(case.cohort.extraction.scratch.iterdir()) == []
