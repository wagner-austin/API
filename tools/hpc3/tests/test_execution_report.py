"""Replay reporting resolves recorded jobs and publishes as the measuring program."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str, require_dict
from platform_core.mcp_testing import sent_arguments

from hpc3.cli import execution_report
from hpc3.cli._load_report import report_identity
from hpc3.core import ledger
from tests.conftest import FakeRun, write_workspace
from tests.execution_fixtures import IMAGE, documents, entry, serve
from tests.wrds_board import TASK, accepting_board


def arguments(root: Path, *, host: str = "hpc3", copies: int = 1) -> list[str]:
    """Write an actual workspace, ledger and frozen input manifest.

    Args:
        root: Test directory.
        host: Host recorded on first submission.
        copies: Count of first submission records, to test ambiguity.

    Returns:
        Complete report arguments.
    """
    workspace = write_workspace(root / "workspace.json")
    for _ in range(copies):
        ledger.append(root / "ledger.jsonl", entry(host=host))
    ledger.append(root / "ledger.jsonl", entry("102"))
    checksum = root / "checksums.txt"
    checksum.write_text(f"{IMAGE}  /pub/runtime.sif\n")
    return [
        "--config",
        workspace,
        "--job",
        "101",
        "--repeat",
        "102",
        "--checksums",
        str(checksum),
        "--report-task",
        TASK,
    ]


@pytest.mark.parametrize("score", [0.25, 0.5])
def test_command_posts_its_measured_report(
    tmp_path: Path, fake_run: FakeRun, emitted: list[str], score: float
) -> None:
    args = arguments(tmp_path)
    board = accepting_board(2)
    serve(fake_run, entry(), *documents())
    serve(fake_run, entry("102"), *documents(score=score))
    assert execution_report.main(args) == (0 if score == 0.25 else 2)
    identity = report_identity(TASK, "hpc")
    assert identity["agent"] == "hpc-report-c4522545"
    checkin, note = (sent_arguments(body) for body in board.bodies)
    assert checkin["harness"] == "hpc3-hpc-evidence"
    assert note["agent"] == identity["agent"]
    assert note["sessionId"] == identity["session_id"]
    assert note["taskId"] == TASK
    assert note["body"] == emitted[0]
    assert emitted[0].startswith("HPC3 REPLAY LOAD-REPORT\n")


@pytest.mark.parametrize(("host", "copies"), [("other", 1), ("hpc3", 0), ("hpc3", 2)])
def test_missing_duplicate_or_foreign_job_is_refused_before_ssh(
    tmp_path: Path, fake_run: FakeRun, host: str, copies: int
) -> None:
    with pytest.raises(ValueError, match="HPC_EVIDENCE_LEDGER"):
        execution_report.main(arguments(tmp_path, host=host, copies=copies))
    assert fake_run.commands() == []


def test_entrypoint_reads_process_arguments(
    tmp_path: Path, fake_run: FakeRun, argv: list[str], emitted: list[str]
) -> None:
    argv[:] = ["report", *arguments(tmp_path)]
    accepting_board(2)
    serve(fake_run, entry(), *documents())
    serve(fake_run, entry("102"), *documents())
    with pytest.raises(SystemExit) as result:
        runpy.run_module("hpc3.cli.execution_report", run_name="__main__")
    assert result.value.code == 0
    assert len(emitted) == 1


def test_publication_failure_is_not_reported_as_success(
    tmp_path: Path, fake_run: FakeRun, emitted: list[str]
) -> None:
    from platform_core.errors import AppError
    from platform_core.mcp_client import EVENT_STREAM_MEDIA_TYPE, McpHttpResponse
    from platform_core.mcp_testing import FakeHttpPost, posted_ok

    from hpc3.cli import _test_hooks

    args = arguments(tmp_path)
    accepting_board(0)
    _test_hooks.http_post = FakeHttpPost(
        [
            posted_ok(),
            McpHttpResponse(status=503, body="unavailable", content_type=EVENT_STREAM_MEDIA_TYPE),
        ]
    )
    serve(fake_run, entry(), *documents())
    serve(fake_run, entry("102"), *documents())
    with pytest.raises(AppError, match="HTTP 503"):
        execution_report.main(args)
    assert emitted == []


def test_missing_remote_artifact_propagates(tmp_path: Path, fake_run: FakeRun) -> None:
    from platform_core.errors import AppError

    args = arguments(tmp_path)
    fake_run.add("cat --", stderr="No such file", returncode=1)
    serve(fake_run, entry(), *documents())
    with pytest.raises(AppError, match="No such file"):
        execution_report.main(args)


def test_altered_record_is_refused_even_if_json_is_valid(fake_run: FakeRun) -> None:
    from platform_core.errors import AppError

    from hpc3.core.execution_evidence import measure
    from tests.conftest import cluster
    from tests.execution_fixtures import CHECKSUMS

    _manifest, record = documents()
    require_dict(record, "fingerprint")["driver_version"] = "changed-in-transit"
    fake_run.add("cat -- /pub/result-101.json.runrecord.json", stdout=dump_json_str(record))
    serve(fake_run, entry(), *documents())
    with pytest.raises(AppError, match="digest"):
        measure(entry(), cluster(), CHECKSUMS)
