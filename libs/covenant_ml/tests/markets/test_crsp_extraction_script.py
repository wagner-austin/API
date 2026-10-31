"""The extraction entrypoint in process and as a real process with a stand-in client.

The subprocess cases put an executable named ``psql`` first on ``PATH``: it
copies the COPY files a database would write, exits nonzero, or writes past
the file-size limit and is killed by the kernel, so the real bounded runner,
writer and encryption run end to end.
"""

from __future__ import annotations

import os
import runpy
import signal
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.config import config_test_hooks
from platform_core.errors import AppError
from platform_core.json_utils import load_json_str
from platform_core.load_report import report_identity
from platform_core.mcp_client import EVENT_STREAM_MEDIA_TYPE, McpHttpResponse
from platform_core.mcp_testing import FakeHttpPost, posted_ok, sent_arguments, stack_endpoint_files
from platform_core.session_label import API_KEY_NAME, STACK_ENV_PATH, TENANT_ID_NAME
from scripts import crsp_extraction

from covenant_ml.markets import _test_hooks as markets_hooks
from covenant_ml.markets.crsp_verify import verify_crsp_extraction
from covenant_ml.markets.crsp_writer import PSQL

from ._crsp_extraction_fixtures import (
    Extraction,
    FakeDatabase,
    database_rows,
    extraction,
    installed,
)
from ._encrypted_fixtures import evaluator_keys

TASK = "1f60a20d-d860-43b3-9616-68f192bcedd9"
ENV_TEXT = f"{API_KEY_NAME}=internal-key\n{TENANT_ID_NAME}=2e137b5f-0000-4000-8000-0000aa\n"


def _arguments(case: Extraction, *, relation_bytes: int = 2_000_000) -> list[str]:
    return [
        "extract",
        "--protocol",
        str(case.protocol),
        "--access",
        str(case.root / "access.json"),
        "--source",
        "crsp_a_stock",
        "--max-rows",
        "20000",
        "--max-relation-bytes",
        str(relation_bytes),
        "--max-partition-bytes",
        "4000000",
        "--seconds",
        "60",
        "--scratch-env",
        "EXTRACT_SCRATCH",
        "--output-env",
        "EXTRACT_OUTPUT",
    ]


def _environment(case: Extraction) -> dict[str, str]:
    environment = dict(config_test_hooks.get_environment())
    environment.update(EXTRACT_SCRATCH=str(case.scratch), EXTRACT_OUTPUT=str(case.output))
    return environment


@pytest.fixture()
def allocation(tmp_path: Path) -> Generator[Extraction, None, None]:
    case = extraction(tmp_path)
    environment = _environment(case)
    previous = config_test_hooks.get_environment
    config_test_hooks.get_environment = lambda: environment
    yield case
    config_test_hooks.get_environment = previous


@pytest.fixture()
def database() -> Generator[FakeDatabase, None, None]:
    with installed(FakeDatabase(database_rows())) as fake:
        yield fake


@pytest.fixture()
def board() -> Generator[None, None, None]:
    """Answer the stack's files from memory; each test binds its own poster."""
    files = stack_endpoint_files()
    files[STACK_ENV_PATH] = ENV_TEXT
    previous = (markets_hooks.http_post, markets_hooks.stack_text)
    markets_hooks.stack_text = files.__getitem__
    yield
    markets_hooks.http_post, markets_hooks.stack_text = previous


def _verify_command(case: Extraction, task: str = TASK) -> list[str]:
    scratch = case.root / "verify-scratch"
    scratch.mkdir(mode=0o700, exist_ok=True)
    command = ["verify", "--collected", str(case.root), "--access"]
    command += [str(case.root / "access.json"), "--scratch", str(scratch)]
    return [*command, "--report-task", task]


def test_extract_then_verify_in_process_posts_the_reading_before_printing_it(
    allocation: Extraction,
    database: FakeDatabase,
    board: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert crsp_extraction.main(_arguments(allocation)) == 0
    assert len(database.calls) == 1
    post = FakeHttpPost([posted_ok(), posted_ok()])
    markets_hooks.http_post = post
    assert crsp_extraction.main(_verify_command(allocation)) == 0
    printed = capsys.readouterr().out
    expected = verify_crsp_extraction(
        allocation.root,
        evaluator_keys(allocation.root),
        scratch=allocation.root / "verify-scratch",
        open_final=False,
    )
    assert load_json_str(printed) == expected
    checkin, note = (sent_arguments(body) for body in post.bodies)
    identity = report_identity(TASK, "wrds")
    assert (checkin["agent"], checkin["sessionId"]) == (
        identity["agent"],
        identity["session_id"],
    )
    assert note["agent"] == "load-report-1f60a20d"
    assert note["taskId"] == TASK
    assert note["body"] == "WRDS EXTRACTION VERIFY LOAD-REPORT\n" + printed.removesuffix("\n")


def test_a_refused_report_fails_verification_without_printing(
    allocation: Extraction,
    database: FakeDatabase,
    board: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert crsp_extraction.main(_arguments(allocation)) == 0
    refusal = McpHttpResponse(status=401, body="bad key", content_type=EVENT_STREAM_MEDIA_TYPE)
    markets_hooks.http_post = FakeHttpPost([posted_ok(), refusal])
    with pytest.raises(AppError, match="HTTP 401"):
        crsp_extraction.main(_verify_command(allocation))
    assert capsys.readouterr().out == ""


def test_a_malformed_report_task_is_refused_before_any_bundle_is_opened(
    allocation: Extraction, database: FakeDatabase, board: None
) -> None:
    assert crsp_extraction.main(_arguments(allocation)) == 0
    post = FakeHttpPost([])
    markets_hooks.http_post = post
    with pytest.raises(ValueError, match="WRDS_REPORT_TASK"):
        crsp_extraction.main(_verify_command(allocation, task="1f60a20d"))
    assert post.bodies == []
    assert list((allocation.root / "verify-scratch").iterdir()) == []


def test_the_stack_reader_reads_the_file_as_text(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(ENV_TEXT, encoding="utf-8")
    assert markets_hooks.read_stack_text(path) == ENV_TEXT


def _client(root: Path, body: str) -> dict[str, str]:
    """Install an executable ``psql`` first on PATH and return the process environment."""
    directory = root / "bin"
    directory.mkdir()
    client = directory / "psql"
    client.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    client.chmod(0o755)
    environment = dict(config_test_hooks.get_environment())
    environment["PATH"] = f"{directory}{os.pathsep}{environment['PATH']}"
    return environment


def _run(
    case: Extraction, environment: dict[str, str], arguments: list[str]
) -> subprocess.CompletedProcess[str]:
    environment.update(EXTRACT_SCRATCH=str(case.scratch), EXTRACT_OUTPUT=str(case.output))
    command = [sys.executable, "-m", "scripts.crsp_extraction", *arguments]
    return subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=120, check=False
    )


def test_real_process_extracts_from_a_client_on_path(tmp_path: Path) -> None:
    case = extraction(tmp_path)
    source = tmp_path / "copy-source"
    for name in ("train-validation", "final"):
        (source / name).mkdir(parents=True)
    (source / "extract.sql").write_text("", encoding="utf-8")
    FakeDatabase(database_rows())(PSQL, source, seconds=1, max_file_bytes=1)
    (source / "extract.sql").unlink()
    environment = _client(tmp_path, f'exec cp -R "{source}/." .\n')
    result = _run(case, environment, _arguments(case))
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    assert sorted(path.name for path in case.output.iterdir()) == [
        "extraction.json",
        "final",
        "metrics.json",
        "train-validation",
    ]
    assert list(case.scratch.iterdir()) == []


def test_real_process_refuses_a_failed_client(tmp_path: Path) -> None:
    case = extraction(tmp_path)
    result = _run(case, _client(tmp_path, "exit 3\n"), _arguments(case))
    assert result.returncode == 1
    assert result.stderr.splitlines()[-1] == (
        "ValueError: [CRSP-011] psql exited 3; no bundle was published"
    )
    assert (list(case.output.iterdir()), list(case.scratch.iterdir())) == ([], [])


def test_real_process_kills_a_client_writing_past_the_file_limit(tmp_path: Path) -> None:
    case = extraction(tmp_path)
    body = "exec dd if=/dev/zero of=train-validation/returns.csv bs=1000 count=10 status=none\n"
    result = _run(case, _client(tmp_path, body), _arguments(case, relation_bytes=5000))
    assert result.returncode == 1
    assert result.stderr.splitlines()[-1] == (
        f"ValueError: [CRSP-011] psql exited {-signal.SIGXFSZ}; no bundle was published"
    )
    assert (list(case.output.iterdir()), list(case.scratch.iterdir())) == ([], [])


def test_entrypoint_requires_a_command() -> None:
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("scripts.crsp_extraction", run_name="__main__")
    assert stopped.value.code == 2
