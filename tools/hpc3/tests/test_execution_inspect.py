"""Single-attempt observation uses shared ledger, accounting and artifact checks."""

import runpy
from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import load_json_str, narrow_json_to_dict, require_dict
from platform_core.mcp_testing import sent_arguments

from hpc3.cli import execution_inspect
from hpc3.core import ledger
from hpc3.core.logs import log_path
from tests.conftest import FakeRun, write_workspace
from tests.execution_fixtures import IMAGE, accounting_row, documents, entry, serve, serve_text
from tests.wrds_board import TASK, accepting_board


def arguments(root: Path, copies: int = 1) -> list[str]:
    workspace = write_workspace(root / "workspace.json")
    for _ in range(copies):
        ledger.append(root / "ledger.jsonl", entry(experiment={"research_attempt": "study:1"}))
    checksums = root / "checksums.txt"
    checksums.write_text(f"{IMAGE}  /pub/runtime.sif\n")
    return [
        "--config",
        workspace,
        "--attempt",
        "study:1",
        "--artifact",
        "/pub/result-101.json",
        "--checksums",
        str(checksums),
        "--report-task",
        TASK,
    ]


@pytest.mark.parametrize("state", ["PENDING", "RUNNING", "FAILED", "COMPLETED"])
def test_native_state_and_verified_completed_record(
    tmp_path: Path, fake_run: FakeRun, emitted: list[str], state: str
) -> None:
    args = arguments(tmp_path)
    board = accepting_board(2)
    value = entry(experiment={"research_attempt": "study:1"})
    fake_run.add("-j 101 ", stdout=accounting_row(value, state=state))
    serve(fake_run, value, *documents())
    assert execution_inspect.main(args) == 0
    report = narrow_json_to_dict(load_json_str(emitted[0]))
    assert require_dict(require_dict(report, "accounting"), "job")["state"] == state
    assert ("record" in report, "measurement" in report) == (
        state == "COMPLETED",
        state == "COMPLETED",
    )
    assert sent_arguments(board.bodies[-1])["body"] == "HPC3 ATTEMPT LOAD-REPORT\n" + emitted[0]


@pytest.mark.parametrize("copies", [0, 2])
def test_ambiguous_recovery_never_guesses(tmp_path: Path, fake_run: FakeRun, copies: int) -> None:
    with pytest.raises(ValueError, match="HPC_RESEARCH_RECOVERY"):
        execution_inspect.main(arguments(tmp_path, copies))
    assert fake_run.commands() == []


def test_repeated_policy_in_another_study_or_host_cannot_shadow_this_attempt(
    tmp_path: Path, fake_run: FakeRun, emitted: list[str]
) -> None:
    args = arguments(tmp_path)
    ledger.append(
        tmp_path / "ledger.jsonl", entry("102", experiment={"research_attempt": "study:1"})
    )
    ledger.append(
        tmp_path / "ledger.jsonl",
        entry(
            "103",
            experiment={"research_attempt": "study:1"},
            artifact="/pub/result-101.json",
            host="another-host",
        ),
    )
    ledger.append(
        tmp_path / "ledger.jsonl", entry("104", experiment={"research_attempt": "other:1"})
    )
    accepting_board(2)
    value = entry(experiment={"research_attempt": "study:1"})
    fake_run.add("-j 101 ", stdout=accounting_row(value))
    serve(fake_run, value, *documents())
    assert execution_inspect.main(args) == 0
    report = narrow_json_to_dict(load_json_str(emitted[0]))
    assert require_dict(report, "ledger")["job_id"] == "101"
    assert require_dict(report, "ledger")["artifact"] == "/pub/result-101.json"


def test_entrypoint_reads_arguments(tmp_path: Path, fake_run: FakeRun, argv: list[str]) -> None:
    argv[:] = ["inspect", *arguments(tmp_path)]
    accepting_board(2)
    value = entry(experiment={"research_attempt": "study:1"})
    fake_run.add("-j 101 ", stdout=accounting_row(value, state="RUNNING"))
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("hpc3.cli.execution_inspect", run_name="__main__")
    assert caught.value.code == 0


@pytest.mark.parametrize("state", ["FAILED", "COMPLETED"])
def test_input_proof_never_reads_or_claims_a_result(
    tmp_path: Path, fake_run: FakeRun, emitted: list[str], state: str
) -> None:
    args = arguments(tmp_path)
    board = accepting_board(2)
    value = entry(experiment={"research_attempt": "study:1"})
    fake_run.add("-j 101 ", stdout=accounting_row(value, state=state))
    serve_text(fake_run, log_path(value), "/pub/runtime.sif: OK\n")
    fake_run.add("sha256sum '/pub/runtime.sif'", stdout=f"{IMAGE}  /pub/runtime.sif\n")
    assert execution_inspect.main([*args, "--scope", "inputs"]) == 0
    report = narrow_json_to_dict(load_json_str(emitted[0]))
    assert report["kind"] == "HPC3_INPUT_EVIDENCE"
    assert require_dict(report, "inputs")["verified_files"] == {"/pub/runtime.sif": IMAGE}
    assert require_dict(report, "inputs")["compute_checksum_lines"] == 1
    assert "record" not in report and "measurement" not in report
    assert all("/pub/result-101.json" not in command for command in fake_run.commands())
    assert sent_arguments(board.bodies[-1])["body"] == "HPC3 ATTEMPT LOAD-REPORT\n" + emitted[0]


@pytest.mark.parametrize("state", ["PENDING", "RUNNING"])
def test_live_allocation_cannot_supply_terminal_input_proof(
    tmp_path: Path, fake_run: FakeRun, state: str
) -> None:
    args = arguments(tmp_path)
    board = accepting_board(2)
    fake_run.add("-j 101 ", stdout=accounting_row(entry(), state=state))
    with pytest.raises(ValueError, match="HPC_EVIDENCE_STATE"):
        execution_inspect.main([*args, "--scope", "inputs"])
    assert board.bodies == []


def test_unknown_evidence_scope_is_refused_before_remote_reads(
    tmp_path: Path, fake_run: FakeRun
) -> None:
    with pytest.raises(ValueError, match="HPC_EVIDENCE_SCOPE"):
        execution_inspect.main([*arguments(tmp_path), "--scope", "everything"])
    assert fake_run.commands() == []


@pytest.mark.parametrize("fault", ["changed-input", "missing-receipt"])
def test_failed_job_input_proof_requires_bytes_and_compute_receipt(
    tmp_path: Path, fake_run: FakeRun, fault: str
) -> None:
    args = arguments(tmp_path)
    board = accepting_board(2)
    value = entry()
    fake_run.add("-j 101 ", stdout=accounting_row(value, state="FAILED"))
    receipt = "/pub/runtime.sif: OK\n" if fault == "changed-input" else ""
    serve_text(fake_run, log_path(value), receipt)
    digest = "b" * 64 if fault == "changed-input" else IMAGE
    fake_run.add("sha256sum '/pub/runtime.sif'", stdout=f"{digest}  /pub/runtime.sif\n")
    if fault == "changed-input":
        with pytest.raises(AppError, match="digest"):
            execution_inspect.main([*args, "--scope", "inputs"])
    else:
        with pytest.raises(ValueError, match="HPC_EVIDENCE_INPUTS"):
            execution_inspect.main([*args, "--scope", "inputs"])
    assert board.bodies == []


def test_failed_job_remains_observable_without_checksum_receipts(
    tmp_path: Path, fake_run: FakeRun, emitted: list[str]
) -> None:
    args = arguments(tmp_path)
    accepting_board(2)
    fake_run.add("-j 101 ", stdout=accounting_row(entry(), state="FAILED"))
    assert execution_inspect.main(args) == 0
    report = narrow_json_to_dict(load_json_str(emitted[0]))
    assert report["kind"] == "HPC3_ATTEMPT_EVIDENCE"
    assert "inputs" not in report and "record" not in report
