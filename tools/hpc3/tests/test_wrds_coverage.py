"""The OptionMetrics coverage measurement: its query, its rows and its one job."""

from __future__ import annotations

import hashlib
import pathlib

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONTypeError, JSONValue, load_json_str
from platform_core.mcp_testing import sent_arguments

from hpc3.cli import wrds_coverage as cli
from hpc3.contracts.wrds_coverage import (
    decode_coverage_receipt,
    decode_coverage_row,
    decode_saved_row,
    encode_coverage_receipt,
    encode_coverage_row,
)
from hpc3.core.wrds_coverage import (
    SLICE,
    VOLATILITY_SURFACES,
    Surface,
    coverage_query,
    option_coverage,
    read_coverage,
    surface_slice,
)
from tests.conftest import FakeRun
from tests.wrds_board import TASK, accepting_board

COVERAGE = (
    "optionmsamp_us,vsurfd2014,1200,10,2014-01-02,2014-12-31,250,1\n"
    "omtrial,vsurfd2014,0,0,,,0,0\n"
    "omtrial,volatility_surface_2013,900,5,2013-01-02,2013-12-31,240,2\n"
)
REMOTE = "/home/uci/coverage"
QUERY = coverage_query(VOLATILITY_SURFACES)


def _sha(text: str) -> str:
    """The sha256 the transport records for a text."""
    return hashlib.sha256(text.encode()).hexdigest()


def prepare(fake_run: FakeRun, *, job_exit: int = 0, coverage: str = COVERAGE) -> None:
    """Script the upload, the job and the download.

    Args:
        fake_run: Process seam fixture.
        job_exit: What the job's ``qsub -sync y`` exits.
        coverage: What the job wrote.
    """
    fake_run.add(f"sha256sum '{REMOTE}/coverage.sql'", stdout=f"{_sha(QUERY)}  sql\n")
    fake_run.add(f"sha256sum '{REMOTE}/coverage.csv'", stdout=f"{_sha(coverage)}  csv\n")
    fake_run.add("cat -- ", stdout=coverage)
    fake_run.add(
        "wrds_option_coverage",
        stdout="31\n",
        stderr="" if job_exit == 0 else "Job 31 exited with exit code 3.",
        returncode=job_exit,
    )


def test_the_query_is_read_only_aggregates_over_each_surface() -> None:
    assert QUERY.startswith("BEGIN READ ONLY;\nSET LOCAL statement_timeout = '1800s';\nCOPY (\n")
    assert QUERY.endswith("\n) TO STDOUT WITH (FORMAT CSV);\nCOMMIT;\n")
    assert QUERY.count("\nUNION ALL\n") == len(VOLATILITY_SURFACES) - 1
    assert 'FROM "optionmsamp_us"."vsurfd2014"' in QUERY
    assert 'FROM "omtrial"."volatility_surface_2013"' in QUERY
    for surface in VOLATILITY_SURFACES:
        sharing = sum(other.put_flag == surface.put_flag for other in VOLATILITY_SURFACES)
        assert QUERY.count(f"FILTER (WHERE {surface_slice(surface)})") == 2 * sharing
    assert 'count(DISTINCT "securityid")' in QUERY
    assert "\"callput\" = 'P'" in QUERY
    assert "SELECT *" not in QUERY
    odd = Surface("o'k", 'we"ird', "secid", "cp_flag")
    assert coverage_query((odd,)).count("SELECT 'o''k', 'we\"ird', ") == 1


def test_an_empty_relation_list_is_refused() -> None:
    with pytest.raises(ValueError, match="WRDS_COVERAGE_EMPTY"):
        coverage_query(())


def test_rows_decode_in_order_and_roundtrip() -> None:
    rows = read_coverage(COVERAGE, VOLATILITY_SURFACES)
    assert rows[0]["securities"] == 10
    assert rows[1]["first_date"] == ""
    assert rows[2]["slice_securities"] == 2
    for row in rows:
        assert decode_saved_row(encode_coverage_row(row)) == row


def test_rows_must_name_exactly_the_relations_asked_for() -> None:
    with pytest.raises(ValueError, match="WRDS_COVERAGE_RELATIONS"):
        read_coverage(COVERAGE, VOLATILITY_SURFACES[:2])


@pytest.mark.parametrize(
    ("fields", "code"),
    [
        (["a"] * 7, "WRDS_COVERAGE_RECORD"),
        (["", "b", "1", "1", "2014-01-02", "2014-01-02", "0", "0"], "WRDS_COVERAGE_RECORD"),
        (["a", "", "1", "1", "2014-01-02", "2014-01-02", "0", "0"], "WRDS_COVERAGE_RECORD"),
        (["a", "b", "1", "2", "2014-01-02", "2014-01-02", "0", "0"], "WRDS_COVERAGE_RECORD"),
        (["a", "b", "5", "2", "2014-01-02", "2014-01-02", "1", "2"], "WRDS_COVERAGE_RECORD"),
        (["a", "b", "5", "2", "2014-01-02", "2014-01-02", "6", "1"], "WRDS_COVERAGE_RECORD"),
        (["a", "b", "5", "2", "2014-01-02", "2014-01-02", "5", "3"], "WRDS_COVERAGE_RECORD"),
        (["a", "b", "5", "2", "", "", "0", "0"], "WRDS_COVERAGE_RECORD"),
        (["a", "b", "0", "0", "2014-01-02", "", "0", "0"], "WRDS_COVERAGE_RECORD"),
        (["a", "b", "5", "2", "2014-02-01", "2014-01-02", "0", "0"], "WRDS_COVERAGE_RECORD"),
        (["a", "b", "x", "2", "2014-01-02", "2014-01-02", "0", "0"], "WRDS_COVERAGE_VALUE"),
        (["a", "b", "5", "2", "2014/01/02", "2014-01-02", "0", "0"], "WRDS_COVERAGE_VALUE"),
        (["a", "b", "5", "2", "2014-13-02", "2014-12-02", "0", "0"], "WRDS_COVERAGE_VALUE"),
        (["a", "b", "5", "2", "2014-02-30", "2014-12-02", "0", "0"], "WRDS_COVERAGE_VALUE"),
    ],
)
def test_malformed_rows_are_refused(fields: list[str], code: str) -> None:
    with pytest.raises(ValueError, match=code):
        decode_coverage_row(fields)


def test_a_saved_row_of_the_wrong_type_is_refused() -> None:
    row: dict[str, JSONValue] = dict(
        encode_coverage_row(read_coverage(COVERAGE, VOLATILITY_SURFACES)[0])
    )
    row["rows"] = "1200"
    with pytest.raises(JSONTypeError):
        decode_saved_row(row)


def test_coverage_runs_one_job_and_saves_verified_aggregates(
    tmp_path: pathlib.Path, fake_run: FakeRun
) -> None:
    prepare(fake_run)
    output = tmp_path / "coverage.csv"
    receipt = option_coverage("wrds", REMOTE, output)
    assert receipt["job_id"] == "31"
    assert receipt["query_sha256"] == _sha(QUERY)
    assert receipt["coverage_sha256"] == _sha(COVERAGE)
    assert receipt["slice"] == SLICE
    assert receipt["rows"] == read_coverage(COVERAGE, VOLATILITY_SURFACES)
    assert output.read_text() == COVERAGE
    saved = load_json_str(pathlib.Path(f"{output}.receipt.json").read_text())
    assert decode_coverage_receipt(saved) == receipt
    assert decode_coverage_receipt(encode_coverage_receipt(receipt)) == receipt
    jobs = [command for command in fake_run.commands() if command.startswith("qsub ")]
    assert len(jobs) == 1
    assert " -V " in jobs[0]
    assert jobs[0].endswith(f"-f {REMOTE}/coverage.sql -o {REMOTE}/coverage.csv")
    uploads = [call.stdin_bytes for call in fake_run.calls if call.stdin_bytes is not None]
    assert uploads == [QUERY.encode()]


def test_a_failed_job_saves_nothing(tmp_path: pathlib.Path, fake_run: FakeRun) -> None:
    prepare(fake_run, job_exit=1)
    with pytest.raises(AppError, match="exit code 3"):
        option_coverage("wrds", REMOTE, tmp_path / "coverage.csv")
    assert list(tmp_path.iterdir()) == []


def test_a_mismatched_artifact_saves_nothing(tmp_path: pathlib.Path, fake_run: FakeRun) -> None:
    prepare(fake_run, coverage=COVERAGE.splitlines(keepends=True)[0])
    with pytest.raises(ValueError, match="WRDS_COVERAGE_RELATIONS"):
        option_coverage("wrds", REMOTE, tmp_path / "coverage.csv")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("existing", ["out", "out.receipt.json"])
def test_existing_outputs_are_not_overwritten(
    existing: str, tmp_path: pathlib.Path, fake_run: FakeRun
) -> None:
    (tmp_path / existing).write_text("keep")
    with pytest.raises(FileExistsError):
        option_coverage("wrds", REMOTE, tmp_path / "out")
    assert fake_run.commands() == []


def test_cli_posts_its_own_load_report(
    tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
) -> None:
    prepare(fake_run)
    post = accepting_board(2)
    output = tmp_path / "coverage.csv"
    argv = ["--host", "wrds", "--remote-dir", REMOTE, "--output", str(output)]
    assert cli.main([*argv, "--report-task", TASK]) == 0
    assert emitted[0] == (
        f"WRDS OPTION COVERAGE LOAD-REPORT job 31 on wrds (grid-engine): coverage sha256 "
        f"{_sha(COVERAGE)}, query sha256 {_sha(QUERY)}; slice {SLICE}; "
        "optionmsamp_us.vsurfd2014: 1200 rows, 10 securities, 2014-01-02 to 2014-12-31, "
        "slice 250 rows over 1 securities; omtrial.vsurfd2014: 0 rows, 0 securities, "
        "no dates to no dates, slice 0 rows over 0 securities; "
        "omtrial.volatility_surface_2013: 900 rows, 5 securities, 2013-01-02 to 2013-12-31, "
        "slice 240 rows over 2 securities. Aggregate counts and bounding dates only; "
        "no dataset rows."
    )
    assert emitted[1] == f"Coverage CSV: {output}; scheduler receipt: {output}.receipt.json"
    note = sent_arguments(post.bodies[1])
    assert note["taskId"] == TASK
    assert note["body"] == emitted[0]


def test_cli_refuses_a_malformed_task_before_submitting(
    tmp_path: pathlib.Path, fake_run: FakeRun
) -> None:
    argv = ["--host", "wrds", "--remote-dir", REMOTE, "--output", str(tmp_path / "out")]
    with pytest.raises(ValueError, match="WRDS_REPORT_TASK"):
        cli.main([*argv, "--report-task", TASK.upper()])
    assert fake_run.commands() == []
