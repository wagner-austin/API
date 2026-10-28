"""Single-attempt observation uses shared ledger, accounting and artifact checks."""

import runpy
from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict, require_dict
from platform_core.mcp_testing import sent_arguments

from hpc3.cli import execution_inspect
from hpc3.core import ledger
from tests.conftest import FakeRun, write_workspace
from tests.execution_fixtures import IMAGE, accounting_row, documents, entry, serve
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


def test_entrypoint_reads_arguments(tmp_path: Path, fake_run: FakeRun, argv: list[str]) -> None:
    argv[:] = ["inspect", *arguments(tmp_path)]
    accepting_board(2)
    value = entry(experiment={"research_attempt": "study:1"})
    fake_run.add("-j 101 ", stdout=accounting_row(value, state="RUNNING"))
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("hpc3.cli.execution_inspect", run_name="__main__")
    assert caught.value.code == 0
