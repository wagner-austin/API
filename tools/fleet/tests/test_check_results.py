"""A check's structured results, kept beside its transcript (MCPs board task 25ab85cf).

The copy the build makes is ``sh`` lines, so they are run for real under
this host's bash against a real recipe directory, the way a node runs them;
the isolated variant needs root on the node and is held to its exact text,
never run here (a run as root left files the suite could not remove).
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess

import pytest
from platform_core.json_utils import InvalidJsonError, JSONTypeError

from fleet.core import dialect, names
from fleet.core.check_results import (
    FAILURES_FILE_NAME,
    FAILURES_MARKER,
    RECIPE_RESULTS_NAME,
    RESULTS_DIGEST_MARKER,
    RUN_RESULTS_NAME,
    TOTALS_FILE_NAME,
    TOTALS_KEYS,
    TOTALS_MARKER,
    ReportTotals,
    read_failed_cases,
    read_results_digest,
    read_test_totals,
    retained_results_path,
    run_results_path,
    sh_failures_line,
    sh_isolated_results_lines,
    sh_results_digest_line,
    sh_results_lines,
    sh_retire_lines,
    sh_totals_line,
)
from tests._host_bash import host_bash

#: What MCPs' maketools ``write_failures`` wrote for a check whose suite
#: failed one case and whose mypy refused one line: ``json.dumps`` of the
#: sorted names, then a newline.
WRITTEN_FAILURES = (
    '["packages/db/src/a.py:3 [arg-type]", "packages/db/tests/test_a.py::test_bad \\"quoted\\""]\n'
)

#: What MCPs' maketools ``write_summaries`` wrote for fleet job ea1dd2c0's
#: run (MCPs board task 000f9302): ``json.dumps`` of the totals, a newline.
WRITTEN_TOTALS = '{"passed": 5024, "failed": 105, "skipped": 24}\n'


def test_the_names_are_one_spelling() -> None:
    assert FAILURES_FILE_NAME == "failures.json"
    assert FAILURES_MARKER == "FLEET_FAILURES"
    assert TOTALS_FILE_NAME == "totals.json"
    assert TOTALS_MARKER == "FLEET_TOTALS"
    assert TOTALS_KEYS == ("failed", "passed", "skipped")
    assert RECIPE_RESULTS_NAME == ".check-results"
    assert RUN_RESULTS_NAME == "check-results"
    assert run_results_path("/s/run-1/run-1") == "/s/run-1/run-1/check-results"
    assert retained_results_path("/s", "run-1") == "/s/logs/run-1.results"


def run_lines(lines: list[str], cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    """Run ``sh`` lines under the build's prologue, as a node runs its script.

    Args:
        lines: The lines.
        cwd: Where they run.

    Returns:
        The finished process.
    """
    script = cwd / "lines.sh"
    script.write_text(dialect.PROLOGUE + "\n".join(lines) + "\n", encoding="utf-8")
    return subprocess.run(
        [host_bash(), script.as_posix()],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_a_build_copies_the_recipe_s_results_beside_its_transcript(
    tmp_path: pathlib.Path,
) -> None:
    target = tmp_path / "run-1"
    recipe = target / "packages" / "db"
    written = recipe / ".check-results"
    written.mkdir(parents=True)
    (written / "db.vitest.junit.xml").write_text("<testsuites/>\n", encoding="utf-8")
    (written / "db.mypy.json").write_text("\n", encoding="utf-8")

    ran = run_lines(sh_results_lines(recipe=recipe.as_posix(), target=target.as_posix()), tmp_path)

    assert ran.returncode == 0, ran.stderr
    kept = target / "check-results"
    assert sorted(entry.name for entry in kept.iterdir()) == ["db.mypy.json", "db.vitest.junit.xml"]
    assert (kept / "db.vitest.junit.xml").read_text(encoding="utf-8") == "<testsuites/>\n"


def test_a_recipe_that_wrote_no_results_leaves_nothing_to_keep(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "run-1"
    (target / "packages" / "db").mkdir(parents=True)

    recipe = (target / "packages" / "db").as_posix()
    ran = run_lines(sh_results_lines(recipe=recipe, target=target.as_posix()), tmp_path)

    assert ran.returncode == 0, ran.stderr
    assert sorted(entry.name for entry in target.iterdir()) == ["packages"]


def test_an_isolated_build_copies_as_root_and_gives_the_copy_to_the_runner() -> None:
    assert sh_isolated_results_lines(recipe="$exec_root/p", target="/s/run-1") == [
        'if sudo -n test -d "$exec_root/p/.check-results"; then '
        'sudo -n cp -R "$exec_root/p/.check-results" "/s/run-1/check-results"; '
        'sudo -n chown -R "$(id -u):$(id -g)" "/s/run-1/check-results"; fi'
    ]


def test_a_retire_moves_the_kept_results_and_a_second_retire_finds_none(
    tmp_path: pathlib.Path,
) -> None:
    target = tmp_path / "run-1"
    (target / "check-results").mkdir(parents=True)
    (target / "check-results" / "a.pytest.junit.xml").write_text("<x/>", encoding="utf-8")
    (tmp_path / "logs").mkdir()
    retained = (tmp_path / "logs" / "run-1.results").as_posix()
    lines = sh_retire_lines(target=target.as_posix(), retained=retained)

    assert run_lines(lines, tmp_path).returncode == 0
    again = run_lines(lines, tmp_path)

    assert again.returncode == 0, again.stderr
    assert not (target / "check-results").exists()
    assert (tmp_path / "logs" / "run-1.results" / "a.pytest.junit.xml").read_text(
        encoding="utf-8"
    ) == "<x/>"


def test_the_build_copies_after_the_recipe_and_before_its_last_two_lines() -> None:
    lines = dialect.build_script(
        target="/s/run-1/run-1",
        path="packages/db",
        workers=2,
        install=(),
        cache_root="/s/cache",
        isolated_docker=False,
        agent="opus-demo-0929",
        loopback_unit=None,
        artifact=None,
    ).splitlines()

    assert lines[-3] == (
        'if test -d "/s/run-1/run-1/packages/db/.check-results"; then '
        'cp -R "/s/run-1/run-1/packages/db/.check-results" "/s/run-1/run-1/check-results"; fi'
    )
    assert lines[-1] == f"printf '%s\\n' \"$status\" > '/s/run-1/run-1/{names.RESULT_NAME}'"


def expected_digest(kept: pathlib.Path) -> str:
    """The kept reports' digest by its stated definition, computed here without ``sh``.

    Args:
        kept: The copy.

    Returns:
        SHA-256 of one ``<hex>  ./<path>`` line per file, sorted bytewise by name.
    """
    names_sorted = sorted(
        (f"./{path.relative_to(kept).as_posix()}" for path in kept.rglob("*") if path.is_file()),
        key=str.encode,
    )
    listing = "".join(
        f"{hashlib.sha256((kept / name[2:]).read_bytes()).hexdigest()}  {name}\n"
        for name in names_sorted
    )
    return hashlib.sha256(listing.encode("utf-8")).hexdigest()


def test_the_tail_read_prints_the_kept_reports_digest_by_its_definition(
    tmp_path: pathlib.Path,
) -> None:
    target = tmp_path / "run-1"
    kept = pathlib.Path(run_results_path(target.as_posix()))
    (kept / "packages__db").mkdir(parents=True)
    (kept / "packages__db.vitest.junit.xml").write_text("<testsuites/>\n", encoding="utf-8")
    (kept / "Z.mypy.json").write_text("\n", encoding="utf-8")
    (kept / "packages__db" / "nested.json").write_text("{}", encoding="utf-8")

    completed = run_lines([sh_results_digest_line(target.as_posix())], tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == f"{RESULTS_DIGEST_MARKER}: {expected_digest(kept)}\n"
    assert read_results_digest(completed.stdout) == expected_digest(kept)


def test_a_run_that_kept_no_report_prints_no_digest(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "run-1"
    empty = tmp_path / "run-2"
    pathlib.Path(run_results_path(empty.as_posix())).mkdir(parents=True)

    lines = [sh_results_digest_line(target.as_posix()), sh_results_digest_line(empty.as_posix())]
    completed = run_lines(lines, tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert read_results_digest(completed.stdout) is None


def test_the_last_digest_line_is_the_one_read() -> None:
    tail = f"{RESULTS_DIGEST_MARKER}: {'a' * 64}\nx\n{RESULTS_DIGEST_MARKER}: {'b' * 64}\n"

    assert read_results_digest(tail) == "b" * 64
    assert read_results_digest(f"{RESULTS_DIGEST_MARKER}: {'a' * 63}\n") is None


def test_the_tail_read_prints_the_names_maketools_wrote(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "run-1"
    kept = pathlib.Path(run_results_path(target.as_posix()))
    kept.mkdir(parents=True)
    (kept / FAILURES_FILE_NAME).write_text(WRITTEN_FAILURES, encoding="utf-8")

    completed = run_lines([sh_failures_line(target.as_posix())], tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == f"{FAILURES_MARKER}: {WRITTEN_FAILURES}"
    assert read_failed_cases(completed.stdout) == [
        "packages/db/src/a.py:3 [arg-type]",
        'packages/db/tests/test_a.py::test_bad "quoted"',
    ]


def test_a_file_naming_none_reads_as_none_and_no_file_as_unread(
    tmp_path: pathlib.Path,
) -> None:
    named_none = tmp_path / "run-1"
    unread = tmp_path / "run-2"
    kept = pathlib.Path(run_results_path(named_none.as_posix()))
    kept.mkdir(parents=True)
    (kept / FAILURES_FILE_NAME).write_text("[]", encoding="utf-8")
    pathlib.Path(run_results_path(unread.as_posix())).mkdir(parents=True)

    none_named = run_lines([sh_failures_line(named_none.as_posix())], tmp_path)
    none_read = run_lines([sh_failures_line(unread.as_posix())], tmp_path)

    assert none_named.stdout == f"{FAILURES_MARKER}: []\n", none_named.stderr
    assert read_failed_cases(none_named.stdout) == []
    assert none_read.returncode == 0, none_read.stderr
    assert none_read.stdout == ""
    assert read_failed_cases(none_read.stdout) is None


def test_the_last_failures_line_is_the_one_read() -> None:
    tail = f'{FAILURES_MARKER}: ["a"]\n{FAILURES_MARKER}: ["b"]\nx {FAILURES_MARKER}: ["c"]\n'

    assert read_failed_cases(tail) == ["b"]


def test_a_failures_line_that_is_no_array_of_names_is_refused() -> None:
    with pytest.raises(InvalidJsonError):
        read_failed_cases(f"{FAILURES_MARKER}: [\n")
    with pytest.raises(InvalidJsonError):
        read_failed_cases(f"{FAILURES_MARKER}: \n")
    with pytest.raises(JSONTypeError):
        read_failed_cases(f'{FAILURES_MARKER}: {{"a": 1}}\n')
    with pytest.raises(JSONTypeError):
        read_failed_cases(f'{FAILURES_MARKER}: ["a", 1]\n')


def test_the_tail_read_prints_the_totals_maketools_counted(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "run-1"
    kept = pathlib.Path(run_results_path(target.as_posix()))
    kept.mkdir(parents=True)
    (kept / TOTALS_FILE_NAME).write_text(WRITTEN_TOTALS, encoding="utf-8")
    unread = tmp_path / "run-2"
    pathlib.Path(run_results_path(unread.as_posix())).mkdir(parents=True)

    completed = run_lines([sh_totals_line(target.as_posix())], tmp_path)
    none_read = run_lines([sh_totals_line(unread.as_posix())], tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == f"{TOTALS_MARKER}: {WRITTEN_TOTALS}"
    assert read_test_totals(completed.stdout) == ReportTotals(passed=5024, failed=105, skipped=24)
    assert none_read.returncode == 0, none_read.stderr
    assert none_read.stdout == ""
    assert read_test_totals(none_read.stdout) is None


def test_the_totals_are_the_reports_never_a_printed_summary() -> None:
    tail = (
        "FAILED tests/test_a.py::test_wording[ALL CHECKS PASSED, 22 passed in mcp-shared]\n"
        "105 failed, 5024 passed, 24 skipped in 119.27s\n"
        f'{TOTALS_MARKER}: {{"passed": 1, "failed": 0, "skipped": 0}}\n'
        f"{TOTALS_MARKER}: {WRITTEN_TOTALS}"
        f'x {TOTALS_MARKER}: {{"passed": 9, "failed": 9, "skipped": 9}}\n'
    )

    assert read_test_totals(tail) == ReportTotals(passed=5024, failed=105, skipped=24)
    assert read_test_totals("105 failed, 5024 passed, 24 skipped in 119.27s\n") is None


@pytest.mark.parametrize(
    ("line", "refusal"),
    [
        ("{", InvalidJsonError),
        ("[1, 2, 3]", JSONTypeError),
        ('{"passed": 1, "failed": 0}', JSONTypeError),
        ('{"passed": 1, "failed": 0, "skipped": 0, "errors": 0}', JSONTypeError),
        ('{"passed": 1, "failed": "0", "skipped": 0}', JSONTypeError),
        ('{"passed": true, "failed": 0, "skipped": 0}', JSONTypeError),
        ('{"passed": 1, "failed": 0, "skipped": -1}', JSONTypeError),
        ('{"passed": -1, "failed": 0, "skipped": 0}', JSONTypeError),
        ('{"passed": 0, "failed": -1, "skipped": 0}', JSONTypeError),
    ],
    ids=[
        "no json",
        "an array",
        "a key short",
        "a key more",
        "a word for a count",
        "a boolean",
        "a negative skipped",
        "a negative passed",
        "a negative failed",
    ],
)
def test_a_totals_line_that_is_no_object_of_three_counts_is_refused(
    line: str, refusal: type[Exception]
) -> None:
    with pytest.raises(refusal):
        read_test_totals(f"{TOTALS_MARKER}: {line}\n")
