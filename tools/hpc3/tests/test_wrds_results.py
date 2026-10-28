"""Successful accounting alone cannot authorize collecting another run's bytes."""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
)

from hpc3.contracts.wrds_run import WrdsRun
from hpc3.core.wrds_results import collect, verified_completion
from hpc3.wrds import _test_hooks as hooks
from hpc3.wrds.worker import execute_run
from tests.conftest import FakeRun
from tests.test_wrds_scheduler import HELD
from tests.test_wrds_worker import job_environment
from tests.wrds_support import example
from tests.wrds_transport import NOW, CopyTransport, completed


def actual_result(directory: pathlib.Path) -> tuple[WrdsRun, JSONObject]:
    """Execute the real payload and decode its actual attestation.

    Args:
        directory: Temporary attempt directory.

    Returns:
        Run and completion metadata.
    """
    run = example(directory)
    hooks.environment = job_environment
    execute_run(directory / "run.json", "one")
    hooks.reset_hooks()
    return run, narrow_json_to_dict(load_json_str((directory / "completion.json").read_text()))


def test_collect_verifies_copied_files_and_writes_completion_last(tmp_path: pathlib.Path) -> None:
    run, result = actual_result(tmp_path)
    transport = CopyTransport(tmp_path / "outputs")
    completed(transport, run, result)
    destination = tmp_path / "collected"
    assert collect(run, "one", destination, tmp_path / "ledger", NOW) == result
    assert (destination / "outputs/metrics.json").read_bytes() == (
        tmp_path / "outputs/metrics.json"
    ).read_bytes()
    assert (
        narrow_json_to_dict(load_json_str((destination / "results.runrecord.json").read_text()))
        == result["record"]
    )
    assert (destination / "completion.json").exists()
    with pytest.raises(FileExistsError):
        collect(run, "one", destination, tmp_path / "ledger", NOW)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("identity", "wrong", "WRDS_RESULT_IDENTITY"),
        ("attempt", "wrong", "WRDS_RESULT_IDENTITY"),
        ("job_id", "456", "WRDS_RESULT_IDENTITY"),
        ("worker_sha256", "a" * 64, "WRDS_WORKER_DRIFT"),
        ("outputs", {}, "WRDS_OUTPUTS"),
        ("outputs", {"metrics.json": "a" * 64}, "WRDS_RESULT_DIGEST"),
    ],
)
def test_completion_mismatch(
    tmp_path: pathlib.Path, field: str, value: JSONValue, error: str
) -> None:
    run, result = actual_result(tmp_path)
    result[field] = value
    completed(FakeRun(), run, result)
    with pytest.raises(ValueError, match=error):
        verified_completion(run, "one")


def test_record_and_runtime_drift(tmp_path: pathlib.Path) -> None:
    run, result = actual_result(tmp_path)
    runtime = require_dict(result, "runtime")
    runtime["version"] = "changed"
    completed(FakeRun(), run, result)
    with pytest.raises(ValueError, match="WRDS_RUNTIME_DRIFT"):
        verified_completion(run, "one")
    runtime["version"] = run["runtime"]["version"]
    record = require_dict(result, "record")
    record["experiment"] = "different"
    completed(FakeRun(), run, result)
    with pytest.raises(ValueError, match="WRDS_RECORD_IDENTITY"):
        verified_completion(run, "one")


def test_active_job_cannot_be_collected(tmp_path: pathlib.Path) -> None:
    run, result = actual_result(tmp_path)
    transport = FakeRun()
    transport.add("qstat", stdout=HELD)
    completed(transport, run, result)
    with pytest.raises(ValueError, match="WRDS_NOT_COMPLETE"):
        verified_completion(run, "one")


@pytest.mark.parametrize("failure", ["corrupt", "transfer"])
def test_incomplete_transfer_never_writes_success(tmp_path: pathlib.Path, failure: str) -> None:
    run, result = actual_result(tmp_path)
    transport = CopyTransport(tmp_path / "outputs")
    if failure == "corrupt":
        (tmp_path / "outputs/metrics.json").write_bytes(b"changed after completion")
        error = "WRDS_RESULT_DIGEST"
    else:
        transport.add("metrics.json", returncode=1, stderr="interrupted transfer")
        error = "WRDS_RESULT_TRANSFER"
    completed(transport, run, result)
    destination = tmp_path / "partial"
    with pytest.raises(ValueError, match=error):
        collect(run, "one", destination, tmp_path / "ledger", NOW)
    assert not (destination / "completion.json").exists()
    assert not (tmp_path / "ledger").exists()
