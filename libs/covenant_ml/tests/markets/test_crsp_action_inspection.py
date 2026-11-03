"""Actual encrypted lineage, process execution and failure cleanup for the case join."""

from __future__ import annotations

import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.config import config_test_hooks
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_list,
    require_str,
)
from scripts import crsp_extraction

from covenant_ml.markets.crsp_action_cases import inspect_crsp_actions
from covenant_ml.markets.encrypted_storage import decrypted_bundle, staged_file_record

from ._crsp_action_fixtures import ActionDatabase, action_rows, capture_actions, installed_actions
from ._crsp_diagnostic_process import prepare_allocation
from ._crsp_refresh_fixtures import RefreshCase, refresh_case
from ._encrypted_fixtures import evaluator_keys, seal_extract


def _capture(root: Path) -> tuple[RefreshCase, str]:
    """Prepare real encrypted inputs through the already-tested extraction boundary."""
    case = refresh_case(root)
    with installed_actions(ActionDatabase(action_rows())):
        result = capture_actions(case)
    return case, require_str(require_dict(result, "bundle"), "sha256")


def _inspect(case: RefreshCase, path: Path, digest: str) -> JSONObject:
    """Use actual fixture custody and scratch for a pinned inspection input."""
    return inspect_crsp_actions(
        case.cohort.destination / "bundle.json",
        case.digest,
        path,
        digest,
        case.cohort.plan,
        evaluator_keys(case.cohort.extraction.root),
        scratch=case.cohort.extraction.scratch,
    )


def _arguments(case: RefreshCase, digest: str) -> list[str]:
    """Bind the native CLI to the fixture's two independent ciphertext identities."""
    root = case.cohort.extraction.root
    return [
        "inspect-actions",
        "--cohort",
        str(case.cohort.destination / "bundle.json"),
        "--cohort-sha256",
        case.digest,
        "--actions",
        str(case.destination / "bundle.json"),
        "--actions-sha256",
        digest,
        "--plan",
        str(root / "plan.json"),
        "--access",
        str(root / "access.json"),
        "--scratch-env",
        "TMPDIR",
        "--output-env",
        "WRDS_OUTPUT_DIR",
    ]


def test_real_encrypted_case_inspection_needs_no_final_key(tmp_path: Path) -> None:
    case, digest = _capture(tmp_path)
    (tmp_path / "keys" / "final.agekey").unlink()
    result = _inspect(case, case.destination / "bundle.json", digest)
    assert result["kind"] == "CRSP_ACTION_CASE_DIAGNOSTIC"
    assert result["plaintext_removed"] is True
    assert result["outcomes_qualified"] is False
    assert result["cohort_bundle_sha256"] == case.digest
    assert result["action_bundle_sha256"] == digest
    assert result["audit_sha256"] == case.cohort.plan["audit_sha256"]
    cases = [narrow_json_to_dict(item) for item in require_list(result, "cases")]
    assert [item["exit_day_offset"] for item in cases] == [28, 33]
    assert [item["path_ordinal"] for item in cases] == [1, 2]
    assert result["rows"] == {
        "delist-events": 2,
        "delist-details": 1,
        "distribution-events": 1,
        "distributions": 1,
        "delist-without-details": 1,
        "distribution-without-details": 0,
    }
    for secret in ("2017-", "2019-", "SELECT", "AGE-SECRET-KEY"):
        assert secret not in dump_json_str(result)
    assert list(case.cohort.extraction.scratch.iterdir()) == []


@pytest.mark.parametrize("failure", ["bundle", "cohort", "extra", "size", "limit", "scope"])
def test_authenticated_but_inconsistent_capture_refuses(tmp_path: Path, failure: str) -> None:
    case, digest = _capture(tmp_path)
    keys = evaluator_keys(tmp_path)
    with decrypted_bundle(
        case.destination / "bundle.json",
        digest,
        partition="train-validation",
        identity=keys.acquire("train-validation"),
        scratch=case.cohort.extraction.scratch,
        max_plaintext_bytes=4_000_000,
    ) as manifest:
        document = narrow_json_to_dict(load_json_str(manifest.read_text()))
        if failure == "bundle":
            document["cohort_bundle_sha256"] = "f" * 64
        elif failure == "cohort":
            require_dict(require_dict(document, "cohort"), "plan")["max_source_rows"] = 20001
        elif failure == "extra":
            (manifest.parent / "extra.txt").write_text("private extra")
        elif failure == "limit":
            bounds = require_dict(document, "limits")
            bounds["max_relation_bytes"] = bounds["max_partition_bytes"] = 1
        else:
            for value in require_list(document, "files"):
                record = narrow_json_to_dict(value)
                if record["name"] == "distribution-events.csv":
                    if failure == "size":
                        record["size_bytes"] = 1
                    else:
                        file = manifest.parent / "distribution-events.csv"
                        file.write_text(file.read_text().replace("1,2019", "99,2019"))
                        actual = staged_file_record(file)
                        record.update(sha256=actual["sha256"], size_bytes=actual["size_bytes"])
        manifest.write_text(dump_json_str(document))
        bundle = seal_extract(manifest, tmp_path / "altered", keys.access, "train-validation")
    with pytest.raises(
        ValueError, match=r"CRSP_ACTION_LINEAGE|CRSP_ACTION_LIMIT|CRSP_ACTION_SCOPE|CRSP-006"
    ):
        _inspect(case, bundle, sha256(bundle.read_bytes()).hexdigest())
    assert list(case.cohort.extraction.scratch.iterdir()) == []


def test_combined_plaintext_budget_refuses_before_opening_either_bundle(tmp_path: Path) -> None:
    case, digest = _capture(tmp_path)
    access = narrow_json_to_dict(load_json_str((tmp_path / "access.json").read_text()))
    access["max_plaintext_bytes"] = 1
    (tmp_path / "access.json").write_text(dump_json_str(access))
    with pytest.raises(ValueError, match="CRSP_ACTION_LIMIT: both bundles"):
        _inspect(case, case.destination / "bundle.json", digest)
    assert list(case.cohort.extraction.scratch.iterdir()) == []


@pytest.mark.parametrize("failure", [False, True])
def test_real_inspection_child_emits_metrics_or_cleans_ciphertext_failure(
    tmp_path: Path, failure: bool
) -> None:
    case, digest = _capture(tmp_path)
    (tmp_path / "keys" / "final.agekey").unlink()
    environment = dict(config_test_hooks.get_environment())
    environment.update(prepare_allocation(case))
    if failure:
        with (case.destination / "distributions.csv.age").open("ab") as stream:
            stream.write(b"PRIVATE_CIPHERTEXT_SENTINEL")
    result = subprocess.run(
        [sys.executable, "-m", "scripts.crsp_extraction", *_arguments(case, digest)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = tmp_path / "native-output"
    assert result.stdout == ""
    assert list(case.cohort.extraction.scratch.iterdir()) == []
    if failure:
        assert result.returncode == 1
        assert (
            result.stderr.splitlines()[-1]
            == "ValueError: ENCRYPTED_DIGEST: file differs from its frozen identity"
        )
        assert "PRIVATE_CIPHERTEXT_SENTINEL" not in result.stderr
        assert list(output.iterdir()) == []
    else:
        assert (result.returncode, result.stderr) == (0, "")
        assert load_json_str((output / "metrics.json").read_text()) == [
            {"name": "cases", "value": 2.0}
        ]
        document = narrow_json_to_dict(load_json_str((output / "action-cases.json").read_text()))
        assert document["plaintext_removed"] is True
        assert len(require_list(document, "cases")) == 2


@pytest.mark.parametrize("occupied", ["", "action-cases.json", "metrics.json", "dangling"])
def test_inspection_cli_uses_native_metrics_and_refuses_existing_outputs(
    tmp_path: Path, occupied: str
) -> None:
    case, digest = _capture(tmp_path)
    previous = config_test_hooks.get_environment
    environment = dict(previous())
    environment.update(prepare_allocation(case))
    config_test_hooks.get_environment = lambda: environment
    output = tmp_path / "native-output"
    if occupied == "dangling":
        (output / "metrics.json").symlink_to(tmp_path / "absent")
    elif occupied:
        (output / occupied).write_text("preserve")
    if occupied:
        with pytest.raises(FileExistsError, match="CRSP_ACTION_OUTPUT"):
            crsp_extraction.main(_arguments(case, digest))
        if occupied != "dangling":
            assert (output / occupied).read_text() == "preserve"
    else:
        assert crsp_extraction.main(_arguments(case, digest)) == 0
        assert load_json_str((output / "metrics.json").read_text()) == [
            {"name": "cases", "value": 2.0}
        ]
    config_test_hooks.get_environment = previous
    assert list(case.cohort.extraction.scratch.iterdir()) == []
