"""Build inspection publishes only fully verified native evidence."""

import runpy
from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.mcp_testing import sent_arguments

from hpc3.cli import image_inspect
from hpc3.core import ledger
from tests.conftest import FakeRun, write_workspace
from tests.image_evidence_fixtures import LOG, build_entry, pins, serve_build
from tests.wrds_board import TASK, accepting_board


def arguments(root: Path) -> list[str]:
    """Prepare an existing workspace, ledger and frozen checksum manifest."""
    workspace = write_workspace(root / "workspace.json")
    ledger.append(root / "ledger.jsonl", build_entry())
    checksums = root / "checksums.txt"
    checksums.write_text("".join(f"{digest}  {path}\n" for path, digest in pins().items()))
    return [
        "--config",
        workspace,
        "--job",
        "101",
        "--checksums",
        str(checksums),
        "--report-task",
        TASK,
    ]


def test_build_report_is_the_measured_metadata(
    tmp_path: Path, fake_run: FakeRun, emitted: list[str]
) -> None:
    board = accepting_board(2)
    serve_build(fake_run)
    assert image_inspect.main(arguments(tmp_path)) == 0
    report = narrow_json_to_dict(load_json_str(emitted[0]))
    assert report["kind"] == "HPC3_IMAGE_BUILD_EVIDENCE"
    assert sent_arguments(board.bodies[-1])["body"] == "HPC3 IMAGE BUILD LOAD-REPORT\n" + emitted[0]


def test_failed_verification_posts_nothing(tmp_path: Path, fake_run: FakeRun) -> None:
    board = accepting_board(2)
    serve_build(fake_run, LOG.replace("smoke 1 OK", "smoke 1 FAILED"))
    with pytest.raises(AppError, match="digests to"):
        image_inspect.main(arguments(tmp_path))
    assert board.bodies == []


def test_entrypoint_reads_process_arguments(
    tmp_path: Path, fake_run: FakeRun, argv: list[str]
) -> None:
    argv[:] = ["inspect", *arguments(tmp_path)]
    accepting_board(2)
    serve_build(fake_run)
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("hpc3.cli.image_inspect", run_name="__main__")
    assert caught.value.code == 0
