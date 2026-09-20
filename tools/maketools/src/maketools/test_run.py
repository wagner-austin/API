"""The shared pytest launcher every project's ``make test`` calls.

Each project's Makefile used to inline this recipe, and 35 of the 36 copies
were byte-identical; the duplication is why a lifecycle bug had to be fixed
36 times instead of once. It then lived in ``scripts/run-tests.ps1``, which
only Windows could run, and moved here on 2026-09-20 (board task 33bb86ce)
so a Linux node can run the same ``make check``.

WHAT THIS ADDS OVER THE INLINE RECIPE

1. KILL-ON-CLOSE. On Windows the launcher joins a job object before it
   spawns anything (:mod:`maketools.job`), so the whole tree dies with it.
   On POSIX the suite starts in its own session and the launcher reaps its
   descendants on the way out; a launcher killed with ``SIGKILL`` cannot,
   and the pre-run sweep is what cleans that up next time.

2. PRE-RUN SWEEP. Reaps a previous run's wreckage before starting, gated on
   age AND on an aggregate CPU idle check so it can never kill a live run
   (:mod:`maketools.reap`).

3. ``--max-worker-restart=0``. xdist otherwise defaults to ``numprocesses *
   4`` and CLONES a replacement for every crashed worker, each re-importing
   torch; the crash loop manufactures the memory pressure that causes more
   crashes. Zero turns the first hard-exited worker into an immediate, loud
   failure.

4. SCOPED COVERAGE CLEANUP. The per-run ``COVERAGE_FILE`` token exists so
   two concurrent runs cannot collide, and the cleanup matches THIS run's
   token only: deleting the whole ``.coverage-*`` glob removed the other
   run's data mid-write and reported a nonsense number on a green suite
   (Model-Trainer, 2026-08-27: 1937 passed, coverage 53.03%).

5. ONE BLAS THREAD PER WORKER. ``-n auto`` is 24 workers on the hub and
   every one that imports torch gets a BLAS that sizes its own pool from the
   CPU count, so 576 threads contend for 24 cores and the CPU-bound tests
   with a per-test timeout lose. Measured on covenant-radar-api, 2637
   tests: ``-n auto`` uncapped 206 s with two crashed workers; capped 94 s
   green. SET HERE, not in a conftest: a BLAS reads these when it LOADS.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

from maketools import _test_hooks, reap
from maketools.job import join_kill_on_close_job
from maketools.venvs import venv_executable

Runner = Literal["poetry", "venv"]

#: The directories coverage measures when they exist.
COVERAGE_ROOTS: Final[Sequence[str]] = ("src", "scripts")

#: Where the per-run coverage data lands; gitignored (``**/runs/``).
RUNS_DIRECTORY: Final[str] = "runs"

#: The prefix every run's coverage files share, followed by the run's token.
COVERAGE_PREFIX: Final[str] = ".coverage-"

#: The BLAS thread caps, item 5 above.
THREAD_CAPS: Final[Sequence[tuple[str, str]]] = (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
)

#: The invocation through poetry, before the parallelism, coverage and
#: caller arguments.
PYTEST_ARGV: Final[Sequence[str]] = ("poetry", "run", "pytest")

#: xdist across every core, and item 3 above. Omitted for a SERIAL suite:
#: OrderedKernels' workers would each spin a CUDA context on the one GPU.
PARALLEL_ARGV: Final[Sequence[str]] = ("-n", "auto", "--max-worker-restart=0")

#: Item 3 above, kept by name for the recipes' documentation.
NO_WORKER_RESTART: Final[str] = PARALLEL_ARGV[2]


def pytest_argv(project: Path, runner: Runner) -> list[str]:
    """How pytest is reached for this project.

    Args:
        project: The package directory.
        runner: ``poetry`` for ``poetry run pytest``; ``venv`` for the
            project's own ``.venv`` interpreter, which is how a ``uv``
            managed package (``libs/cleargbm_rs``) runs it.

    Returns:
        The leading argv.
    """
    if runner == "venv":
        return [str(venv_executable(project, "python")), "-m", "pytest"]
    return list(PYTEST_ARGV)


def coverage_arguments(project: Path) -> list[str]:
    """The ``--cov`` arguments for the roots this project has.

    Args:
        project: The project directory.

    Returns:
        ``--cov-branch``, the terminal report, and one ``--cov=<root>`` per
        existing root.
    """
    arguments = ["--cov-branch", "--cov-report=term-missing"]
    arguments.extend(f"--cov={root}" for root in COVERAGE_ROOTS if (project / root).is_dir())
    return arguments


def remove_run_files(runs: Path, token: str) -> int:
    """Delete this run's coverage files and nothing else's.

    Args:
        runs: The runs directory.
        token: This run's coverage token.

    Returns:
        How many files were removed.
    """
    removed = 0
    for path in sorted(runs.glob(f"{token}*")):
        _test_hooks.remove_file(path)
        removed += 1
    return removed


def join_job_or_refuse() -> bool:
    """Item 1 on Windows: join the job object, fatally if it cannot be joined.

    There is no degraded mode. The post-run reap cannot cover a killed
    launcher, which is the exact case the job object exists for, so
    continuing would run the suite with the protection silently absent; a
    warning scrolls past, a non-zero exit does not.

    Returns:
        True when joined.
    """
    problem = join_kill_on_close_job(_test_hooks.job_api())
    if problem != "":
        _test_hooks.write_error(f"run-tests: could not join a kill-on-close job object: {problem}")
        return False
    _test_hooks.write_line(
        "run-tests: joined kill-on-close job object (tree dies with this process)"
    )
    return True


def run_tests(
    project: Path,
    pytest_arguments: Sequence[str],
    *,
    sweep: bool,
    serial: bool = False,
    runner: Runner = "poetry",
) -> int:
    """Run the project's suite under the launcher's protections.

    Args:
        project: The project directory, which is the recipe's cwd.
        pytest_arguments: Extra arguments appended to the invocation.
        sweep: Whether to run the pre-run sweep; off for debugging only.
        serial: Run without xdist (see :data:`PARALLEL_ARGV`).
        runner: How pytest is reached (see :func:`pytest_argv`).

    Returns:
        pytest's exit status, or 1 when the launcher refused to start it.
    """
    windows = _test_hooks.platform() == "win32"
    if windows:
        if not join_job_or_refuse():
            return 1
    else:
        _test_hooks.write_line("run-tests: the suite runs in its own session and is reaped on exit")
    if sweep:
        reap.sweep_stale(project)
    runs = project / RUNS_DIRECTORY
    runs.mkdir(exist_ok=True)
    token = COVERAGE_PREFIX + _test_hooks.token()
    environment = _test_hooks.environ()
    environment["COVERAGE_FILE"] = str(runs.resolve() / token)
    for name, value in THREAD_CAPS:
        environment[name] = value
    parallel = () if serial else PARALLEL_ARGV
    argv = [
        *pytest_argv(project, runner),
        *parallel,
        "-v",
        *coverage_arguments(project),
        *pytest_arguments,
    ]
    try:
        return _test_hooks.run_inheriting(
            argv, cwd=project, env=environment, new_session=not windows
        )
    finally:
        remove_run_files(runs, token)
        # Belt and braces. The job object already covers the case where this
        # process is killed; this covers the ordinary path, and on POSIX it
        # is the only teardown there is.
        reap.reap_descendants(_test_hooks.process_id())


__all__ = [
    "COVERAGE_PREFIX",
    "COVERAGE_ROOTS",
    "NO_WORKER_RESTART",
    "PARALLEL_ARGV",
    "PYTEST_ARGV",
    "RUNS_DIRECTORY",
    "THREAD_CAPS",
    "Runner",
    "coverage_arguments",
    "join_job_or_refuse",
    "pytest_argv",
    "remove_run_files",
    "run_tests",
]
