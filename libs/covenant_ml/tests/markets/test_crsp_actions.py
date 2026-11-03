"""Encrypted diagnostic evidence and adversarial checks on actual returned source bytes."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import (
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_list,
    require_str,
)

from covenant_ml.markets.crsp_action_records import (
    decode_crsp_action_evidence,
    encode_crsp_action_evidence,
)
from covenant_ml.markets.crsp_action_schema import CRSP_ACTIONS
from covenant_ml.markets.crsp_actions import check_crsp_action_rows
from covenant_ml.markets.crsp_extract import load_crsp_sources
from covenant_ml.markets.crsp_queries import crsp_scopes
from covenant_ml.markets.crsp_rows import decode_crsp_rows, encode_crsp_rows
from covenant_ml.markets.crsp_snapshot import PSQL
from covenant_ml.markets.encrypted_storage import decrypted_bundle

from ._crsp_action_fixtures import ActionDatabase, action_rows, capture_actions, installed_actions
from ._crsp_database import database_protocol
from ._crsp_extraction_fixtures import NOW, limits
from ._crsp_refresh_fixtures import refresh_case
from ._encrypted_fixtures import evaluator_keys


@pytest.mark.parametrize("synthetic", [False, True])
def test_private_actions_retain_null_zero_flags_and_linked_ids_without_study_admission(
    tmp_path: Path, synthetic: bool
) -> None:
    case = refresh_case(tmp_path)
    (tmp_path / "keys" / "final.agekey").unlink()
    setup = tmp_path / "setup.sql" if synthetic else None
    if setup is not None:
        setup.write_text("dsenames(permno) AS (SELECT 1)")
    database = ActionDatabase(action_rows())
    with installed_actions(database):
        result = capture_actions(case, setup)
    assert database.calls == 1
    assert result["source"] == ("synthetic" if synthetic else "crsp_a_stock")
    assert result["kind"] == "CRSP_CORPORATE_ACTION_DIAGNOSTIC"
    assert result["plaintext_removed"] is True
    assert result["started_at"] == result["completed_at"] == NOW
    assert result["rows"] == {
        "delist-events": 2,
        "delist-details": 1,
        "distribution-events": 1,
        "distributions": 1,
        "delist-without-details": 1,
        "distribution-without-details": 0,
    }
    for private in ("permno", "2019-", "SELECT", "disdivamt", "AGE-SECRET-KEY"):
        assert private not in dump_json_str(result)
    assert all(
        path.suffix == ".age" or path.name == "bundle.json" for path in case.destination.iterdir()
    )
    assert "final/" not in database.driver
    with decrypted_bundle(
        case.destination / "bundle.json",
        require_str(require_dict(result, "bundle"), "sha256"),
        partition="train-validation",
        identity=evaluator_keys(tmp_path).acquire("train-validation"),
        scratch=case.cohort.extraction.scratch,
        max_plaintext_bytes=4_000_000,
    ) as path:
        document = load_json_str(path.read_text())
        record = decode_crsp_action_evidence(document)
        assert encode_crsp_action_evidence(record) == document
        assert record["cohort"]["plan"] == case.cohort.plan
        assert (record["setup_sha256"] is not None) == synthetic
        assert (path.parent / "extract.sql").read_text() == database.driver
        for relation in CRSP_ACTIONS:
            decoded = decode_crsp_rows(
                (path.parent / (relation.role + ".csv")).read_bytes(),
                relation.role,
                relation.columns,
                NOW,
            )
            assert list(decoded) == database.rows[relation.role]
        with pytest.raises(ValueError, match=r"CRSP-006.*exact extract fields"):
            load_crsp_sources(path)
        original = encode_crsp_action_evidence(record)
        changes: tuple[tuple[str, JSONValue], ...] = (
            ("kind", "study"),
            ("source", "another-library"),
            ("setup_sha256", None if synthetic else "a" * 64),
            ("started_at", "2020-01-01T00:00:00+00:00"),
            ("completed_at", "2020-01-01T00:00:00+00:00"),
            ("transaction_started_at", "2026-10-31T09:00:00+01:00"),
            ("snapshot", " "),
            ("files", []),
            ("extra", 1),
        )
        for field, value in changes:
            with pytest.raises(ValueError, match="CRSP_ACTION_RECORD"):
                decode_crsp_action_evidence({**original, field: value})
        wrong_inventory = narrow_json_to_dict(load_json_str(dump_json_str(original)))
        narrow_json_to_dict(require_list(wrong_inventory, "files")[0])["name"] = "unexpected.csv"
        with pytest.raises(ValueError, match="every exact query and source file once"):
            decode_crsp_action_evidence(wrong_inventory)
        altered = narrow_json_to_dict(load_json_str(dump_json_str(original)))
        for value in require_list(altered, "files"):
            item = narrow_json_to_dict(value)
            if require_str(item, "name") == ("setup.sql" if synthetic else "protocol.json"):
                item["sha256"] = "f" * 64
        with pytest.raises(ValueError, match="protocol or synthetic definitions differ"):
            decode_crsp_action_evidence(altered)
    assert list(case.cohort.extraction.scratch.iterdir()) == []


def test_action_reader_refuses_missing_key_duplicate_projection_and_private_scalar() -> None:
    relation = CRSP_ACTIONS[3]
    original = action_rows()["distributions"][0]
    with pytest.raises(ValueError, match="diagnostic projection differs"):
        decode_crsp_rows(b"", relation.role, relation.columns[:-1], NOW)
    for field in ("disseqnbr", "disdivamt", "dispaydt"):
        row = {**original, field: "PRIVATE_ACTION_CELL"}
        with pytest.raises(ValueError) as stopped:
            decode_crsp_rows(
                encode_crsp_rows([row], relation.columns), relation.role, relation.columns, NOW
            )
        assert str(stopped.value) == f"[CRSP-007] invalid {field} scalar in distributions CSV"
        assert stopped.value.__suppress_context__ is True
    with pytest.raises(ValueError, match="duplicate source key"):
        decode_crsp_rows(
            encode_crsp_rows([original, original], relation.columns),
            relation.role,
            relation.columns,
            NOW,
        )


@pytest.mark.parametrize(
    "failure", ["bytes", "total", "rows", "security", "early", "late", "orphan"]
)
def test_post_query_validation_refuses_actual_out_of_contract_rows(
    tmp_path: Path, failure: str
) -> None:
    rows = action_rows()
    bounds = limits()
    if failure == "bytes":
        bounds = limits(max_relation_bytes=1)
    elif failure == "total":
        sizes = [len(encode_crsp_rows(rows[item.role], item.columns)) for item in CRSP_ACTIONS]
        bounds = limits(max_relation_bytes=max(sizes), max_partition_bytes=max(sizes))
    elif failure == "rows":
        bounds = limits(max_rows=1)
    elif failure == "security":
        rows["distributions"][0]["permno"] = 99
    elif failure == "early":
        rows["distributions"][0]["disexdt"] = "1990-01-01"
    elif failure == "late":
        rows["distributions"][0]["dispaydt"] = "2020-01-01"
    else:
        rows["distribution-events"] = []
    (tmp_path / "train-validation").mkdir()
    (tmp_path / "extract.sql").write_text("")
    ActionDatabase(rows)(PSQL, tmp_path, seconds=120, max_file_bytes=2_000_000)
    with pytest.raises(ValueError, match=r"CRSP_ACTION_LIMIT|CRSP_ACTION_SCOPE|CRSP_ACTION_KEY"):
        check_crsp_action_rows(
            {
                item.role: (tmp_path / "train-validation" / (item.role + ".csv")).read_bytes()
                for item in CRSP_ACTIONS
            },
            crsp_scopes(database_protocol())[0],
            (1,),
            NOW,
            bounds,
        )


def test_scope_refusal_cleans_authenticated_plaintext_and_publishes_nothing(tmp_path: Path) -> None:
    case = refresh_case(tmp_path)
    rows = action_rows()
    rows["distributions"][0]["dispaydt"] = "2020-01-01"
    with (
        installed_actions(ActionDatabase(rows)),
        pytest.raises(ValueError, match="CRSP_ACTION_SCOPE"),
    ):
        capture_actions(case)
    assert not case.destination.exists()
    assert list(case.cohort.extraction.scratch.iterdir()) == []
