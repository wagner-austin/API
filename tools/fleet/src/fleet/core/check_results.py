"""A check's structured results, kept beside its transcript (MCPs board task 25ab85cf).

WHY. MCPs read which pushed test files passed on a fleet run, and whether a
failed check's failures were its task's own, by regular expressions over the
retained transcript. A vitest, pytest or mypy release that rewords a line
turned every file to not-run or charged a task with an inherited failure.
So each MCPs check now writes what its runners say in their own formats,
JUnit XML from vitest and pytest and mypy's JSON records, into
:data:`RECIPE_RESULTS_NAME` in the directory its recipe ran in (MCPs
``packages/maketools/src/maketools/check_results.py``, whose
``RESULTS_DIRECTORY_NAME`` this name copies: the two repositories share no
package, so this is the one deliberate copy).

WHAT THE RUN DOES WITH THEM. Once the recipe has ended, and before the tree
it ran in is removed, the build copies that directory to
:func:`run_results_path` beside the transcript (:func:`sh_results_lines`,
:func:`sh_isolated_results_lines` from an isolated build's copy).
The retire moves it to :func:`retained_results_path`, next to the retained
transcript under the stage root's logs directory (:func:`sh_retire_lines`),
and the verdict line names that path as ``results=<node>:<path>``
(:mod:`fleet.core.verdict`), which the readers in MCPs copy back. A recipe
that wrote none (a project whose check runs no such runner, a build stopped
before its recipe) leaves nothing, and the path the verdict names is absent,
which a reader reports as results missing rather than as a clean run.
"""

from __future__ import annotations

import re
from typing import Final

from platform_core.json_utils import load_json_str, narrow_json_to_list, narrow_json_to_str

from fleet.core.names import LOGS_DIRECTORY

#: The opening token of the line the tail read prints for the kept reports'
#: digest (:func:`sh_results_digest_line`).
RESULTS_DIGEST_MARKER: Final = "FLEET_RESULTS"

#: That line, as :func:`read_results_digest` reads it back.
RESULTS_DIGEST_LINE: Final[re.Pattern[str]] = re.compile(
    rf"^{RESULTS_DIGEST_MARKER}: ([0-9a-f]{{64}})$", re.MULTILINE
)

#: The opening token of the line the tail read prints for the failures the
#: check's own reports name (:func:`sh_failures_line`).
FAILURES_MARKER: Final = "FLEET_FAILURES"

#: That line, as :func:`read_failed_cases` reads it back.
FAILURES_LINE: Final[re.Pattern[str]] = re.compile(rf"^{FAILURES_MARKER}: (.*)", re.MULTILINE)

#: The file MCPs' check lock writes into :data:`RECIPE_RESULTS_NAME` once its
#: runners return: a JSON array of every failure the reports name, possibly
#: none, written only when the directory holds a report (MCPs
#: ``packages/maketools/src/maketools/check_reports.py``, ``write_failures``;
#: the name copies ``check_results.FAILURES_FILE_NAME`` for the reason
#: :data:`RECIPE_RESULTS_NAME` does).
FAILURES_FILE_NAME: Final = "failures.json"

#: The directory a check's runners write their reports to, in the directory
#: its recipe runs in.
RECIPE_RESULTS_NAME: Final = ".check-results"

#: The copy's name under the run's own directory, beside its transcript.
RUN_RESULTS_NAME: Final = "check-results"


def run_results_path(target: str) -> str:
    """Where a build keeps its check's results until the run is retired.

    Args:
        target: The dispatch's absolute remote directory.

    Returns:
        ``<target>/check-results``.
    """
    return f"{target}/{RUN_RESULTS_NAME}"


def retained_results_path(stage_root: str, run_id: str) -> str:
    """Where a retired run's check results are kept on its node.

    Args:
        stage_root: The node's declared stage root.
        run_id: The dispatch.

    Returns:
        ``<stage_root>/logs/<run_id>.results``, beside its transcript.
    """
    return f"{stage_root}/{LOGS_DIRECTORY}/{run_id}.results"


def sh_results_lines(*, recipe: str, target: str) -> list[str]:
    """The ``sh`` lines that copy the recipe's results beside the transcript, as the runner.

    Args:
        recipe: The directory the recipe ran in.
        target: The dispatch's absolute remote directory.

    Returns:
        One line copying the directory when the recipe wrote it.
    """
    source = f'"{recipe}/{RECIPE_RESULTS_NAME}"'
    kept = f'"{run_results_path(target)}"'
    return [f"if test -d {source}; then cp -R {source} {kept}; fi"]


def sh_isolated_results_lines(*, recipe: str, target: str) -> list[str]:
    """The same copy out of an isolated build's copy, whose files only root may read.

    A function of its own rather than a flag of :func:`sh_results_lines`:
    the runner's copy is run for real by the suite, and a mutant that
    flipped such a flag made that run copy as root on a node whose runner
    may ``sudo``, leaving files the suite's own cleanup could not remove
    (fleet job a810c1fb on lavender, 2026-11-08).

    Args:
        recipe: The directory the recipe ran in, double-quoted where it is
            used, so the shell variable in it (``$exec_root``) expands.
        target: The dispatch's absolute remote directory.

    Returns:
        One line copying the directory as root when the recipe wrote it and
        giving the copy to the runner's own user, so its retire can move it.
    """
    source = f'"{recipe}/{RECIPE_RESULTS_NAME}"'
    kept = f'"{run_results_path(target)}"'
    return [
        f"if sudo -n test -d {source}; then sudo -n cp -R {source} {kept}; "
        f'sudo -n chown -R "$(id -u):$(id -g)" {kept}; fi'
    ]


def sh_results_digest_line(target: str) -> str:
    """The ``sh`` line that prints the digest of a run's kept reports, before its retire.

    The digest is what MCPs stores beside the check's result (MCPs board task
    601eef87, migration 957's ``results_digest``), so a reader that copies the
    reports back can tell it holds the bytes the run kept. It is the SHA-256
    of ``sha256sum``'s output over every file under the copy, each named
    ``./<path>`` relative to it and sorted bytewise by name: one line per
    file, ``<hex>  ./<path>``. A copy holding no file prints no line, as a
    run whose recipe wrote none keeps none.

    Args:
        target: The dispatch's absolute remote directory.

    Returns:
        One line, run by the tail read (:func:`fleet.core.run_registry.log_tail_script`).
    """
    kept = f'"{run_results_path(target)}"'
    return (
        f'if [ -d {kept} ] && [ -n "$(find {kept} -type f -print -quit)" ]; then '
        f'digest="$(cd {kept} && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum '
        f"| sha256sum | cut -d ' ' -f 1)\"; printf '{RESULTS_DIGEST_MARKER}: %s\\n' \"$digest\"; fi"
    )


def read_results_digest(tail: str) -> str | None:
    """Read the kept reports' digest off the tail read's answer.

    Args:
        tail: What the tail read printed.

    Returns:
        The last digest line's SHA-256, or None when the run kept no report.
    """
    last: re.Match[str] | None = None
    for match in RESULTS_DIGEST_LINE.finditer(tail):
        last = match
    return None if last is None else last.group(1)


def sh_failures_line(target: str) -> str:
    """The ``sh`` line that prints the failures a run's kept reports name, before its retire.

    The names are what MCPs stores as the check's failed cases (MCPs board
    task d26709c0, migration 959): maketools parses the JUnit and mypy
    reports it kept and writes :data:`FAILURES_FILE_NAME` beside them, and
    this repository shares no package with it, so the runner sends what that
    file holds rather than parsing the reports a second time. A run whose
    check wrote no such file prints no line, and its close sends the names
    as unread. An assignment, so a ``cat`` that fails ends the read under
    ``set -e`` rather than printing a line with no names.

    Args:
        target: The dispatch's absolute remote directory.

    Returns:
        One line, run by the tail read (:func:`fleet.core.run_registry.log_tail_script`).
    """
    names = f'"{run_results_path(target)}/{FAILURES_FILE_NAME}"'
    return (
        f'if [ -f {names} ]; then failures="$(cat {names})"; '
        f"printf '{FAILURES_MARKER}: %s\\n' \"$failures\"; fi"
    )


def read_failed_cases(tail: str) -> list[str] | None:
    """Read the failures the kept reports name off the tail read's answer.

    Args:
        tail: What the tail read printed.

    Returns:
        The last failures line's names, possibly none, or None when the run
        kept no :data:`FAILURES_FILE_NAME`, so its names were never read.

    Raises:
        platform_core.json_utils.InvalidJsonError: When the line is not JSON.
        platform_core.json_utils.JSONTypeError: When it is not an array of strings.
    """
    last: re.Match[str] | None = None
    for match in FAILURES_LINE.finditer(tail):
        last = match
    if last is None:
        return None
    return [narrow_json_to_str(name) for name in narrow_json_to_list(load_json_str(last.group(1)))]


def sh_retire_lines(*, target: str, retained: str) -> list[str]:
    """The ``sh`` lines a retire moves the run's results to where they are kept.

    Args:
        target: The dispatch's absolute remote directory.
        retained: :func:`retained_results_path` for the run.

    Returns:
        One line moving the run's copy there when the run kept one; a retire
        sent again after it moved finds none and moves nothing, as the
        transcript's own move does.
    """
    kept = f'"{run_results_path(target)}"'
    return [f'if [ -d {kept} ]; then mv -f {kept} "{retained}"; fi']


__all__ = [
    "FAILURES_FILE_NAME",
    "FAILURES_LINE",
    "FAILURES_MARKER",
    "RECIPE_RESULTS_NAME",
    "RESULTS_DIGEST_LINE",
    "RESULTS_DIGEST_MARKER",
    "RUN_RESULTS_NAME",
    "read_failed_cases",
    "read_results_digest",
    "retained_results_path",
    "run_results_path",
    "sh_failures_line",
    "sh_isolated_results_lines",
    "sh_results_digest_line",
    "sh_results_lines",
    "sh_retire_lines",
]
