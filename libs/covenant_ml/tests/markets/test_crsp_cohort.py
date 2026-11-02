"""Authenticated cohort derivation, private storage and refusal without study admission."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_int,
    require_list,
    require_str,
)
from platform_core.testing import sample_run_fingerprint

from covenant_ml.markets.crsp_cohort import derive_crsp_cohort
from covenant_ml.markets.crsp_cohort_records import (
    CrspCohortPlan,
    decode_crsp_cohort,
    decode_crsp_cohort_plan,
    encode_crsp_cohort,
    encode_crsp_cohort_plan,
)
from covenant_ml.markets.crsp_extract import load_crsp_sources
from covenant_ml.markets.crsp_rows import decode_crsp_rows
from covenant_ml.markets.encrypted_storage import decrypted_bundle, read_encrypted_bundle
from covenant_ml.markets.research_trial import execute_trial
from covenant_ml.markets.research_trial_records import decode_research_trial

from ._crsp_cohort_fixtures import CohortCase, cohort_case
from ._crsp_database import database_protocol, relation_columns
from ._crsp_study_fixtures import study_protocol
from ._encrypted_fixtures import evaluator_keys
from ._trial_fixtures import frozen_request


def _derive(case: CohortCase, plan: CrspCohortPlan) -> JSONObject:
    return derive_crsp_cohort(
        case.extraction.output / "train-validation" / "bundle.json",
        case.audit,
        plan,
        evaluator_keys(case.extraction.root),
        scratch=case.extraction.scratch,
        destination=case.destination,
    )


def _assert_refused(case: CohortCase) -> None:
    assert list(case.extraction.scratch.iterdir()) == []
    assert not case.destination.exists()


def test_real_encrypted_cohort_keeps_lineage_and_cannot_load_as_a_study(tmp_path: Path) -> None:
    case = cohort_case(tmp_path)
    (tmp_path / "keys" / "final.agekey").unlink()
    result = _derive(case, case.plan)
    assert (result["cases"], result["securities"], result["plaintext_removed"]) == (2, 1, True)
    assert result["fresh_vendor_extraction"] is False
    assert result["plan"] == encode_crsp_cohort_plan(case.plan)
    assert "permno" not in dump_json_str(result)
    assert "SELECT" not in dump_json_str(result)
    assert list(case.extraction.scratch.iterdir()) == []
    digest = require_str(require_dict(result, "bundle"), "sha256")
    bundle_path = case.destination / "bundle.json"
    bundle = read_encrypted_bundle(bundle_path, digest)
    assert bundle["manifest"]["name"] == "cohort.json"
    assert {p.name for p in case.destination.iterdir()} == {
        "bundle.json",
        "cohort.json.age",
        "audit.json.age",
        "protocol.json.age",
        *(role + suffix + ".age" for role in case.rows for suffix in (".csv", ".sql")),
    }
    with decrypted_bundle(
        bundle_path,
        digest,
        partition="train-validation",
        identity=evaluator_keys(tmp_path).acquire("train-validation"),
        scratch=case.extraction.scratch,
        max_plaintext_bytes=case.plan["max_cohort_bytes"],
    ) as path:
        with pytest.raises(ValueError, match="CRSP-006.*exact extract fields"):
            load_crsp_sources(path)
        document = load_json_str(path.read_text())
        cohort = decode_crsp_cohort(document)
        assert encode_crsp_cohort(cohort) == document
        assert cohort["derived_at"] == result["derived_at"]
        assert cohort["source"]["protocol"]["sha256"] == case.plan["protocol_sha256"]
        assert cohort["subset"]["extracted_at"] == cohort["source"]["extracted_at"]
        assert (path.parent / "audit.json").read_bytes() == case.audit.read_bytes()
        columns = relation_columns(database_protocol())
        expected_counts: JSONObject = {}
        for role, rows in case.rows.items():
            expected = tuple(
                row for row in rows if role == "calendar" or require_int(row, "permno") == 1
            )
            decoded = decode_crsp_rows(
                (path.parent / (role + ".csv")).read_bytes(),
                role,
                columns[role],
                cohort["source"]["extracted_at"],
            )
            assert decoded == expected
            expected_counts[role] = len(expected)
        assert result["rows"] == expected_counts
    assert list(case.extraction.scratch.iterdir()) == []


@pytest.mark.parametrize("field", ["bundle_sha256", "protocol_sha256", "audit_sha256"])
def test_changed_input_identity_refuses_and_cleans_up(tmp_path: Path, field: str) -> None:
    case = cohort_case(tmp_path)
    plan = encode_crsp_cohort_plan(case.plan)
    plan[field] = "f" * 64
    with pytest.raises(ValueError, match="CRSP_COHORT|WRDS_DATA_DRIFT"):
        _derive(case, decode_crsp_cohort_plan(plan))
    _assert_refused(case)


@pytest.mark.parametrize("field", ["max_cohort_bytes", "max_source_rows"])
def test_evidence_budgets_refuse_before_publication(tmp_path: Path, field: str) -> None:
    case = cohort_case(tmp_path)
    plan = encode_crsp_cohort_plan(case.plan)
    plan[field] = 1
    with pytest.raises(ValueError, match="CRSP_COHORT_LIMIT|CRSP-012"):
        _derive(case, decode_crsp_cohort_plan(plan))
    _assert_refused(case)


def test_metadata_also_counts_toward_plaintext_budget(tmp_path: Path) -> None:
    case = cohort_case(tmp_path)
    result = _derive(case, case.plan)
    bundle = read_encrypted_bundle(
        case.destination / "bundle.json", require_str(require_dict(result, "bundle"), "sha256")
    )
    size = sum(f["plaintext"]["size_bytes"] for f in bundle["files"])
    case = case._replace(destination=tmp_path / "too-small")
    plan = encode_crsp_cohort_plan(case.plan)
    plan["max_cohort_bytes"] = size - 100
    with pytest.raises(ValueError, match="CRSP_COHORT_LIMIT: diagnostic metadata"):
        _derive(case, decode_crsp_cohort_plan(plan))
    _assert_refused(case)


@pytest.mark.parametrize("linked", [False, True])
def test_existing_output_including_dangling_link_is_not_replaced(
    tmp_path: Path, linked: bool
) -> None:
    case = cohort_case(tmp_path)
    if linked:
        case.destination.symlink_to(tmp_path / "absent")
    else:
        case.destination.mkdir()
    with pytest.raises(FileExistsError, match="CRSP_COHORT_OUTPUT"):
        _derive(case, case.plan)
    assert case.destination.is_symlink() is linked
    assert list(case.extraction.scratch.iterdir()) == []


def test_final_bundle_is_refused_without_a_final_key(tmp_path: Path) -> None:
    case = cohort_case(tmp_path)
    (tmp_path / "keys" / "final.agekey").unlink()
    final = case.extraction.output / "final" / "bundle.json"
    plan = encode_crsp_cohort_plan(case.plan)
    plan["bundle_sha256"] = sha256(final.read_bytes()).hexdigest()
    result = narrow_json_to_dict(require_list(case.document, "results")[0])
    result["bundle_sha256"] = plan["bundle_sha256"]
    audit_bytes = dump_json_str(case.document).encode()
    case.audit.write_bytes(audit_bytes)
    plan["audit_sha256"] = sha256(audit_bytes).hexdigest()
    with pytest.raises(ValueError, match="ENCRYPTED_ACCESS"):
        derive_crsp_cohort(
            final,
            case.audit,
            decode_crsp_cohort_plan(plan),
            evaluator_keys(tmp_path),
            scratch=case.extraction.scratch,
            destination=case.destination,
        )
    _assert_refused(case)


def test_changed_ciphertext_is_refused_and_partial_plaintext_is_removed(tmp_path: Path) -> None:
    case = cohort_case(tmp_path)
    path = case.extraction.output / "train-validation" / "returns.csv.age"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="ENCRYPTED_DIGEST"):
        _derive(case, case.plan)
    _assert_refused(case)


def test_actual_trial_refuses_a_cohort_even_when_its_bundle_is_admitted(tmp_path: Path) -> None:
    case = cohort_case(tmp_path)
    _derive(case, case.plan)
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
    with pytest.raises(ValueError, match="CRSP-006.*exact extract fields"):
        execute_trial(
            decode_research_trial(document),
            sample_run_fingerprint(image_digest="d" * 64),
            "c" * 64,
            output,
            keys=evaluator_keys(tmp_path),
            scratch=case.extraction.scratch,
        )
    assert not output.exists()
    assert list(case.extraction.scratch.iterdir()) == []


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("document", "results", []),
        ("result", "kind", "final-audit"),
        ("result", "bundle_sha256", "f" * 64),
        ("result", "all_paths_accounted", False),
        ("result", "refused_cases", []),
        ("result", "refused_paths", 3),
        ("result", "accepted_paths", 19),
        ("case", "period", "final"),
        ("entry", "present", False),
        ("case", "path_ordinal", 21),
        ("case", "path_ordinal", 3),
        ("second_entry", "row_sha256", "duplicate"),
        ("entry", "row_sha256", "f" * 64),
    ],
)
def test_inconsistent_or_unresolved_audit_refuses_without_plaintext(
    tmp_path: Path, target: str, field: str, value: JSONValue
) -> None:
    case = cohort_case(tmp_path)
    result = narrow_json_to_dict(require_list(case.document, "results")[0])
    cases = require_list(result, "refused_cases")
    first = narrow_json_to_dict(cases[0])
    entry = require_dict(first, "entry")
    targets = {
        "document": case.document,
        "result": result,
        "case": first,
        "entry": entry,
        "second_entry": require_dict(narrow_json_to_dict(cases[1]), "entry"),
    }
    targets[target][field] = entry["row_sha256"] if value == "duplicate" else value
    payload = dump_json_str(case.document).encode()
    case.audit.write_bytes(payload)
    plan = encode_crsp_cohort_plan(case.plan)
    plan["audit_sha256"] = sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="CRSP_COHORT_|MARKET-001"):
        _derive(case, decode_crsp_cohort_plan(plan))
    _assert_refused(case)
