"""The artifact report posts only an array every member of which verified."""

import runpy
from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import load_json_str, narrow_json_to_dict, require_list
from platform_core.mcp_testing import sent_arguments

from hpc3.cli import artifact_report
from hpc3.core import ledger
from hpc3.core.digest import sha256_hex
from tests.conftest import FakeRun, write_workspace
from tests.execution_fixtures import ARRAY_NAME, accounting_row, array_member, serve_member
from tests.wrds_board import TASK, accepting_board


def arguments(root: Path, fake: FakeRun, cluster_bytes: tuple[bytes, bytes]) -> list[str]:
    """Record a two-member array, keep local copies and serve the cluster's side.

    Args:
        root: Where the workspace, ledger and local copies go.
        fake: The remote-command seam.
        cluster_bytes: What each member's artifact holds on the cluster.

    Returns:
        The command's arguments.
    """
    workspace = write_workspace(root / "workspace.json")
    local = root / "local"
    local.mkdir()
    for index, data in enumerate(cluster_bytes):
        value = array_member(index)
        ledger.append(root / "ledger.jsonl", value)
        (local / f"shard-{index}.json").write_bytes(f"shard {index}".encode())
        serve_member(fake, value, data)
    return [
        "--config",
        workspace,
        "--array",
        "101",
        "--local-dir",
        str(local),
        "--report-task",
        TASK,
    ]


def test_the_posted_report_is_the_measured_array(
    tmp_path: Path, fake_run: FakeRun, emitted: list[str]
) -> None:
    board = accepting_board(2)
    assert artifact_report.main(arguments(tmp_path, fake_run, (b"shard 0", b"shard 1"))) == 0
    report = narrow_json_to_dict(load_json_str(emitted[0]))
    assert (report["arrays"], report["local_dir"]) == (["101"], str(tmp_path / "local"))
    members = [narrow_json_to_dict(item) for item in require_list(report, "members")]
    assert [member["job_id"] for member in members] == ["101_0", "101_1"]
    assert [member["artifact_sha256"] for member in members] == [
        sha256_hex(b"shard 0"),
        sha256_hex(b"shard 1"),
    ]
    assert sent_arguments(board.bodies[-1])["body"] == "HPC3 ARTIFACT LOAD-REPORT\n" + emitted[0]


def test_a_campaigns_arrays_are_named_together(
    tmp_path: Path, fake_run: FakeRun, emitted: list[str]
) -> None:
    """Shard 1 failed in array 101 and its campaign resubmission in 202 completed it."""
    accepting_board(2)
    # Ahead of the helper's completed row for 101_1: the fake answers with the
    # first rule a command matches.
    failed = accounting_row(array_member(1), name=ARRAY_NAME, state="FAILED", exit="1:0")
    fake_run.add("-j 101_1 ", stdout=failed)
    argv = arguments(tmp_path, fake_run, (b"shard 0", b"shard 1"))
    rerun = array_member(1, array="202")
    ledger.append(tmp_path / "ledger.jsonl", rerun)
    serve_member(fake_run, rerun, b"shard 1")
    argv[argv.index("101")] = "101,202"
    assert artifact_report.main(argv) == 0
    report = narrow_json_to_dict(load_json_str(emitted[0]))
    assert report["arrays"] == ["101", "202"]
    members = [narrow_json_to_dict(item) for item in require_list(report, "members")]
    assert [member["job_id"] for member in members] == ["101_0", "202_1"]


def test_a_differing_copy_posts_nothing(tmp_path: Path, fake_run: FakeRun) -> None:
    board = accepting_board(2)
    with pytest.raises(AppError, match=r"shard-1\.json"):
        artifact_report.main(arguments(tmp_path, fake_run, (b"shard 0", b"edited")))
    assert board.bodies == []


def test_entrypoint_reads_process_arguments(
    tmp_path: Path, fake_run: FakeRun, argv: list[str]
) -> None:
    argv[:] = ["report", *arguments(tmp_path, fake_run, (b"shard 0", b"shard 1"))]
    accepting_board(2)
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("hpc3.cli.artifact_report", run_name="__main__")
    assert caught.value.code == 0
