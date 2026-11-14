"""The artifact report posts only an array every member of which verified."""

import runpy
from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.mcp_testing import sent_arguments

from hpc3.cli import artifact_report
from hpc3.core import ledger
from hpc3.core.digest import sha256_hex
from tests.conftest import FakeRun, write_workspace
from tests.execution_fixtures import INSTANT, accounting_row, entry
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
        value = entry(f"101_{index}", artifact=f"/pub/out/shard-{index}.json")
        ledger.append(root / "ledger.jsonl", value)
        (local / f"shard-{index}.json").write_bytes(f"shard {index}".encode())
        fake.add(f"-j {value['job_id']} ", stdout=accounting_row(value))
        fake.add(f"sha256sum '/pub/out/shard-{index}.json'", stdout=f"{sha256_hex(data)}  x\n")
        fake.add(f"stat -c %Y -- /pub/out/shard-{index}.json", stdout=str(INSTANT))
    return ["--config", workspace, "--array", "101", "--local-dir", str(local), "--report-task", TASK]


def test_the_posted_report_is_the_measured_array(tmp_path: Path, fake_run: FakeRun, emitted: list[str]) -> None:
    board = accepting_board(2)
    assert artifact_report.main(arguments(tmp_path, fake_run, (b"shard 0", b"shard 1"))) == 0
    report = narrow_json_to_dict(load_json_str(emitted[0]))
    assert (report["array"], report["local_dir"]) == ("101", str(tmp_path / "local"))
    members = report["members"]
    assert isinstance(members, list)
    assert len(members) == 2
    assert sent_arguments(board.bodies[-1])["body"] == "HPC3 ARTIFACT LOAD-REPORT\n" + emitted[0]


def test_a_differing_copy_posts_nothing(tmp_path: Path, fake_run: FakeRun) -> None:
    board = accepting_board(2)
    with pytest.raises(AppError, match="shard-1.json"):
        artifact_report.main(arguments(tmp_path, fake_run, (b"shard 0", b"edited")))
    assert board.bodies == []


def test_entrypoint_reads_process_arguments(tmp_path: Path, fake_run: FakeRun, argv: list[str]) -> None:
    argv[:] = ["report", *arguments(tmp_path, fake_run, (b"shard 0", b"shard 1"))]
    accepting_board(2)
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("hpc3.cli.artifact_report", run_name="__main__")
    assert caught.value.code == 0
