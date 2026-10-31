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
from platform_core.json_utils import load_json_str
from scripts import crsp_extraction

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


def _arguments(case: Extraction, *, relation_bytes: int = 2_000_000) -> list[str]:
    return [
        "extract",
        "--protocol",
        str(case.protocol),
        "--access",
        str(case.root / "access.json"),
        "--schema",
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


def test_extract_then_verify_in_process(
    allocation: Extraction, database: FakeDatabase, capsys: pytest.CaptureFixture[str]
) -> None:
    assert crsp_extraction.main(_arguments(allocation)) == 0
    assert len(database.calls) == 1
    scratch = allocation.root / "verify-scratch"
    scratch.mkdir(mode=0o700)
    command = ["verify", "--collected", str(allocation.root), "--access"]
    command += [str(allocation.root / "access.json"), "--scratch", str(scratch)]
    assert crsp_extraction.main(command) == 0
    printed = load_json_str(capsys.readouterr().out)
    expected = verify_crsp_extraction(
        allocation.root, evaluator_keys(allocation.root), scratch=scratch, open_final=False
    )
    assert printed == expected


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
