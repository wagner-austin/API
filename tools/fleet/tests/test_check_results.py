"""A check's structured results, kept beside its transcript (MCPs board task 25ab85cf).

The copy the build makes is ``sh`` lines, so they are run for real under
this host's bash against a real recipe directory, the way a node runs them;
the isolated variant needs root on the node and is held to its exact text.
"""

from __future__ import annotations

import pathlib
import subprocess

from fleet.core import dialect, names
from fleet.core.check_results import (
    RECIPE_RESULTS_NAME,
    RUN_RESULTS_NAME,
    retained_results_path,
    run_results_path,
    sh_results_lines,
    sh_retire_lines,
)
from tests._host_bash import host_bash


def test_the_names_are_one_spelling() -> None:
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

    ran = run_lines(
        sh_results_lines(recipe=recipe.as_posix(), target=target.as_posix(), isolated=False),
        tmp_path,
    )

    assert ran.returncode == 0, ran.stderr
    kept = target / "check-results"
    assert sorted(entry.name for entry in kept.iterdir()) == ["db.mypy.json", "db.vitest.junit.xml"]
    assert (kept / "db.vitest.junit.xml").read_text(encoding="utf-8") == "<testsuites/>\n"


def test_a_recipe_that_wrote_no_results_leaves_nothing_to_keep(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "run-1"
    (target / "packages" / "db").mkdir(parents=True)

    ran = run_lines(
        sh_results_lines(
            recipe=(target / "packages" / "db").as_posix(), target=target.as_posix(), isolated=False
        ),
        tmp_path,
    )

    assert ran.returncode == 0, ran.stderr
    assert sorted(entry.name for entry in target.iterdir()) == ["packages"]


def test_an_isolated_build_copies_as_root_and_gives_the_copy_to_the_runner() -> None:
    assert sh_results_lines(recipe="$exec_root/p", target="/s/run-1", isolated=True) == [
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
