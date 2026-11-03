"""Real action-capture children prove private errors and ciphertext-only publication."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.config import config_test_hooks
from platform_core.json_utils import load_json_str, narrow_json_to_dict, require_dict
from scripts import crsp_extraction

from covenant_ml.markets.crsp_snapshot import PSQL

from ._crsp_action_fixtures import ActionDatabase, action_rows, installed_actions
from ._crsp_diagnostic_process import (
    diagnostic_arguments,
    prepare_allocation,
    run_diagnostic_process,
)
from ._crsp_refresh_fixtures import refresh_case


def test_action_cli_metrics_use_the_existing_native_observation_contract(tmp_path: Path) -> None:
    case = refresh_case(tmp_path)
    previous = config_test_hooks.get_environment
    environment = dict(previous())
    environment.update(prepare_allocation(case))
    config_test_hooks.get_environment = lambda: environment
    with installed_actions(ActionDatabase(action_rows())):
        assert crsp_extraction.main(diagnostic_arguments(case, 120, 2_000_000, "actions")) == 0
    config_test_hooks.get_environment = previous
    output = tmp_path / "native-output"
    result = narrow_json_to_dict(load_json_str((output / "actions.json").read_text()))
    assert result["kind"] == "CRSP_CORPORATE_ACTION_DIAGNOSTIC"
    assert load_json_str((output / "metrics.json").read_text()) == [
        {"name": "delist-events", "value": 2.0},
        {"name": "delist-details", "value": 1.0},
        {"name": "distribution-events", "value": 1.0},
        {"name": "distributions", "value": 1.0},
        {"name": "delist-without-details", "value": 1.0},
        {"name": "distribution-without-details", "value": 0.0},
    ]
    assert list(case.cohort.extraction.scratch.iterdir()) == []


@pytest.mark.parametrize("field", ["", "disseqnbr", "disdivamt", "dispaydt"])
def test_real_action_process_succeeds_or_refuses_private_scalar_without_leak(
    tmp_path: Path, field: str
) -> None:
    case = refresh_case(tmp_path)
    (tmp_path / "keys" / "final.agekey").unlink()
    rows = action_rows()
    sentinel = "PRIVATE_ACTION_SCALAR_2468"
    if field:
        rows["distributions"][0][field] = sentinel
    source = tmp_path / "client-output"
    (source / "train-validation").mkdir(parents=True)
    (source / "extract.sql").write_text("")
    ActionDatabase(rows)(PSQL, source, seconds=120, max_file_bytes=2_000_000)
    (source / "extract.sql").unlink()
    result = run_diagnostic_process(case, f'exec cp -R "{source}/." .\n', 120, 2_000_000, "actions")
    output = tmp_path / "native-output"
    assert result.stdout == ""
    assert list(case.cohort.extraction.scratch.iterdir()) == []
    if field:
        assert result.returncode == 1
        assert result.stderr.splitlines()[-1] == (
            f"ValueError: [CRSP-007] invalid {field} scalar in distributions CSV"
        )
        assert sentinel not in result.stderr
        assert list(output.iterdir()) == []
    else:
        assert (result.returncode, result.stderr) == (0, "")
        public = narrow_json_to_dict(load_json_str((output / "actions.json").read_text()))
        assert require_dict(public, "rows")["delist-without-details"] == 1
        assert public["plaintext_removed"] is True
        assert (output / "actions" / "bundle.json").is_file()


def test_real_action_client_failure_removes_plaintext_without_printing_sql(tmp_path: Path) -> None:
    case = refresh_case(tmp_path)
    result = run_diagnostic_process(
        case, "echo 'private SQL' >&2\nexit 3\n", 120, 2_000_000, "actions"
    )
    assert (result.returncode, result.stdout) == (1, "")
    assert result.stderr.splitlines()[-1] == (
        "ValueError: [CRSP-011] psql exited 3; no bundle was published"
    )
    assert "private SQL" not in result.stderr
    assert list(case.cohort.extraction.scratch.iterdir()) == []
    assert list((tmp_path / "native-output").iterdir()) == []
