"""Bounded extraction, including actual process failures through the shared seam."""

from __future__ import annotations

import datetime
import hashlib
import pathlib
import subprocess
import sys
from collections.abc import Generator, Mapping, Sequence

import pytest
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str

from hpc3.cli import wrds
from hpc3.contracts.wrds_extract import (
    CrspSelection,
    decode_crsp_selection,
    encode_crsp_selection,
    extraction_sql,
)
from hpc3.contracts.wrds_sql import psql_argv
from hpc3.core.wrds_bundle import worker_bundle
from hpc3.wrds import _test_hooks
from hpc3.wrds.extract import extract
from hpc3.wrds.worker import main

SCOPE = CrspSelection(
    source_table="crsp.dsf",
    security_id=10001,
    start_date="2020-01-02",
    end_date="2020-01-07",
    max_rows=3,
)
CSV = "permno,date,ret\n10001,2020-01-02,0.01\n10001,2020-01-03,\n10001,2020-01-06,-0.02\n"
NOW = "2026-10-28T16:00:00+00:00"


@pytest.fixture(autouse=True)
def effects(tmp_path: pathlib.Path) -> Generator[None, None, None]:
    """Use a private output directory and restore worker effects."""
    _test_hooks.reset_hooks()
    _test_hooks.environment = lambda: {"WRDS_OUTPUT_DIR": str(tmp_path)}
    _test_hooks.now_iso = lambda: NOW
    yield
    _test_hooks.reset_hooks()


def test_selection_roundtrip_and_exact_query() -> None:
    assert decode_crsp_selection(encode_crsp_selection(SCOPE)) == SCOPE
    assert extraction_sql(SCOPE) == (
        "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;\n"
        "SET LOCAL statement_timeout = '120s';\n"
        "SET LOCAL DateStyle = 'ISO, YMD';\n"
        "COPY (SELECT permno, date, ret FROM crsp.dsf WHERE permno = 10001"
        " AND date BETWEEN DATE '2020-01-02' AND DATE '2020-01-07'"
        " ORDER BY date LIMIT 4) TO STDOUT WITH (FORMAT CSV, HEADER);\nCOMMIT;\n"
    )


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("extra", 1, "FIELDS"),
        ("source_table", "crsp.dsf; DROP TABLE x", "TABLE"),
        ("security_id", 0, "BOUND"),
        ("max_rows", 2, "BOUND"),
        ("max_rows", 10001, "BOUND"),
        ("start_date", "20200102", "DATES"),
        ("end_date", "20200107", "DATES"),
        ("start_date", "2020-02-01", "DATES"),
    ],
)
def test_selection_refusals(key: str, value: JSONValue, error: str) -> None:
    document = encode_crsp_selection(SCOPE)
    document[key] = value
    with pytest.raises(ValueError, match="WRDS_EXTRACT_" + error):
        decode_crsp_selection(document)


def query_result(text: str, exit_code: int = 0) -> _test_hooks.Execute:
    """Inject SQL output; real process nonzero and timeout twins follow below.

    Args:
        text: COPY output to validate.
        exit_code: Simulated psql status.

    Returns:
        The process seam, asserting the entire actual invocation.
    """

    def run(
        argv: Sequence[str],
        directory: pathlib.Path,
        environment: Mapping[str, str],
        seconds: int,
    ) -> int:
        assert list(argv) == psql_argv(str(directory / "extract.sql"), str(directory / "data.csv"))
        assert environment["WRDS_OUTPUT_DIR"] == str(directory)
        assert seconds == 150
        assert (directory / "extract.sql").read_text() == extraction_sql(SCOPE)
        (directory / "data.csv").write_text(text)
        return exit_code

    return run


def test_extract_preserves_rows_query_timestamps_and_digests(tmp_path: pathlib.Path) -> None:
    _test_hooks.execute = query_result(CSV)
    assert main(["extract", dump_json_str(encode_crsp_selection(SCOPE))]) == 0
    assert (tmp_path / "data.csv").read_text() == CSV
    assert load_json_str((tmp_path / "extraction.json").read_text()) == {
        "selection": encode_crsp_selection(SCOPE),
        "rows": 3,
        "started_at": NOW,
        "extracted_at": NOW,
        "csv_file": "data.csv",
        "csv_sha256": hashlib.sha256(CSV.encode()).hexdigest(),
        "query_file": "extract.sql",
        "query_sha256": hashlib.sha256(extraction_sql(SCOPE).encode()).hexdigest(),
    }
    assert load_json_str((tmp_path / "metrics.json").read_text()) == [
        {"name": "extracted_rows", "value": 3}
    ]
    with pytest.raises(FileExistsError):
        extract(SCOPE)
    assert (tmp_path / "data.csv").read_text() == CSV


@pytest.mark.parametrize(
    ("text", "code", "error"),
    [
        ("", 3, "QUERY"),
        ("", 0, "HEADER"),
        ("wrong\n", 0, "HEADER"),
        ("permno,date,ret\n10001,2020-01-02\n", 0, "ROW"),
        (CSV.replace("10001", "10002"), 0, "ROW"),
        (CSV.replace("2020-01-02", "2019-12-31"), 0, "ROW"),
        (CSV.replace("2020-01-06", "2020-01-08"), 0, "ROW"),
        (CSV.replace("2020-01-03", "2020-01-02"), 0, "ROW"),
        ("permno,date,ret\n", 0, "ROWS"),
        (CSV + "10001,2020-01-07,0.02\n", 0, "ROWS"),
    ],
)
def test_bad_query_or_rows_never_write_success(
    tmp_path: pathlib.Path,
    text: str,
    code: int,
    error: str,
) -> None:
    _test_hooks.execute = query_result(text, code)
    with pytest.raises(ValueError, match="WRDS_EXTRACT_" + error):
        extract(SCOPE)
    assert not (tmp_path / "extraction.json").exists()
    assert not (tmp_path / "metrics.json").exists()


def test_future_scope_refused_before_process_or_files(tmp_path: pathlib.Path) -> None:
    _test_hooks.now_iso = lambda: "2020-01-07T00:00:00+00:00"
    with pytest.raises(ValueError, match="WRDS_EXTRACT_FUTURE"):
        extract(SCOPE)
    assert list(tmp_path.iterdir()) == []


def test_real_process_failure_and_timeout_leave_logs(tmp_path: pathlib.Path) -> None:
    assert (
        _test_hooks._default_execute(
            [sys.executable, "-c", "raise SystemExit(3)"],
            tmp_path,
            {},
            5,
        )
        == 3
    )
    with pytest.raises(subprocess.TimeoutExpired):
        _test_hooks._default_execute(
            [sys.executable, "-c", "import time; time.sleep(20)"],
            tmp_path,
            {},
            1,
        )
    assert (tmp_path / "payload.out").read_bytes() == b""
    assert (tmp_path / "payload.err").read_bytes() == b""
    assert not (tmp_path / "extraction.json").exists()


def test_bundle_command_reserves_and_verifies_bytes(
    tmp_path: pathlib.Path,
    emitted: list[str],
) -> None:
    archive = tmp_path / "extract.pyz"
    assert wrds.main(["bundle", "--output", str(archive)]) == 0
    assert archive.read_bytes() == worker_bundle()
    assert load_json_str(emitted[-1]) == {
        "output": str(archive),
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    with pytest.raises(FileExistsError):
        wrds.main(["bundle", "--output", str(archive)])
    _test_hooks.reset_hooks()
    instant = datetime.datetime.fromisoformat(_test_hooks.now_iso())
    assert instant.utcoffset() == datetime.timedelta(0)
