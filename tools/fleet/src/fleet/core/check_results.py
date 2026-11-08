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

from typing import Final

from fleet.core.names import LOGS_DIRECTORY

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
    "RECIPE_RESULTS_NAME",
    "RUN_RESULTS_NAME",
    "retained_results_path",
    "run_results_path",
    "sh_isolated_results_lines",
    "sh_results_lines",
    "sh_retire_lines",
]
