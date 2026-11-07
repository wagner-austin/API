"""The shared pytest launcher every project's ``make test`` calls.

Each project's Makefile used to inline this recipe, and 35 of the 36 copies
were byte-identical; the duplication is why a lifecycle bug had to be fixed
36 times instead of once. It then lived in ``scripts/run-tests.ps1``, which
only Windows could run, and moved here on 2026-09-20 (board task 33bb86ce)
so a Linux node can run the same ``make check``.

WHAT THIS ADDS OVER THE INLINE RECIPE

1. PRE-RUN SWEEP. Reaps a previous run's wreckage before starting, gated on
   age AND on an aggregate CPU idle check so it can never kill a live run
   (:mod:`maketools.reap`).

2. ``--max-worker-restart=0``. xdist otherwise defaults to ``numprocesses *
   4`` and CLONES a replacement for every crashed worker, each re-importing
   torch; the crash loop manufactures the memory pressure that causes more
   crashes. Zero turns the first hard-exited worker into an immediate, loud
   failure.

3. SCOPED COVERAGE CLEANUP. The per-run ``COVERAGE_FILE`` token exists so
   two concurrent runs cannot collide, and the cleanup matches THIS run's
   token only: deleting the whole ``.coverage-*`` glob removed the other
   run's data mid-write and reported a nonsense number on a green suite
   (Model-Trainer, 2026-08-27: 1937 passed, coverage 53.03%).

4. ONE BLAS THREAD PER WORKER. ``-n auto`` is 24 workers on the hub and
   every one that imports torch gets a BLAS that sizes its own pool from the
   CPU count, so 576 threads contend for 24 cores and the CPU-bound tests
   with a per-test timeout lose. Measured on covenant-radar-api, 2637
   tests: ``-n auto`` uncapped 206 s with two crashed workers; capped 94 s
   green. SET HERE, not in a conftest: a BLAS reads these when it LOADS.

5. THE RUN'S OWN MODEL CACHES. Hugging Face and Triton cache under the
   user's home by default, so a suite that loads ``sshleifer/tiny-gpt2`` or
   compiles a Triton helper wrote outside its own temporary directories:
   Model-Trainer's check on lavender (fleet job 58471c54, 2026-10-31) left
   ``~/.triton/cache/.../cuda_utils.so`` and a ``.no_exist`` marker under
   ``~/.cache/huggingface/hub`` (MCPs board task bcb7c6ef). ``HF_HOME`` and
   ``TRITON_HOME`` name a directory under ``runs/`` that is this run's alone
   and is removed with its coverage files. Set here for item 4's reason:
   ``huggingface_hub`` reads ``HF_HOME`` when it is imported.

6. A HOME OF THE RUN'S OWN, when the recipe asks (``--own-home``). Item 5
   moves the two caches that were measured, and nothing would notice a
   third. Asked, the launcher points ``HOME`` at a directory under
   ``runs/`` that is this run's alone, and after the suite fails the run
   naming every file left there: each is a write the suite would have made
   under the user's home (MCPs board task bcb7c6ef). pytest is then reached
   through the venv's interpreter, named by ``poetry env info --executable``
   before ``HOME`` moves, since Poetry under a fresh ``HOME`` reads no
   configuration and finds no venv. Opt-in, because only the suites whose
   recipes pass it have been measured under it.

7. THE HUB REFUSED, when the recipe asks (``--hub-offline``). Item 5's
   ``HF_HOME`` lies beside item 6's ``HOME``, not inside it, so a test that
   resolves a hub id downloads into the run's own cache, which the launcher
   removes, and item 6 sees nothing: two platform_langid tests pulled the
   3.86 GB MMS-LID-126 checkpoint that way on pendragon, 240 s of fleet job
   f2af5334 (MCPs board task abb0a10e). Asked, the launcher sets
   ``HF_HUB_OFFLINE``, under which ``huggingface_hub`` and ``transformers``
   resolve a hub id from the empty cache alone and raise, so such a test
   fails by name. Opt-in, because Model-Trainer's suite loads
   ``sshleifer/tiny-gpt2`` from the hub by design.

8. MUTATION TESTING, when the recipe asks (``--mutation``, MCPs board task
   e52e5148). The suite runs with ``--cov-context=test``, so its coverage
   file records which test ran each line, and once it passes
   :mod:`maketools.mutation` mutates the src lines the commit adds against
   ``origin/main`` with cosmic-ray, runs each mutant by the tests that ran
   its line, and fails naming every mutant they leave alive. It runs
   before the run's coverage file is removed, which it reads. Opt-in per
   package, because each package that asks carries cosmic-ray as a dev
   dependency and its own changes must then leave no mutant alive.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Final

from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError

from maketools import _test_hooks, reap
from maketools.mutation.launch import mutate_after_suite
from maketools.venvs import venv_executable


class Runner(StrEnum):
    """How pytest is reached; each member is the ``--runner`` word naming it."""

    POETRY = "poetry"
    VENV = "venv"


#: The directories coverage measures when they exist.
COVERAGE_ROOTS: Final[Sequence[str]] = ("src", "scripts")

#: Where the per-run coverage data lands; gitignored (``**/runs/``).
RUNS_DIRECTORY: Final[str] = "runs"

#: The prefix every run's coverage files share, followed by the run's token.
COVERAGE_PREFIX: Final[str] = ".coverage-"

#: The prefix of the run's cache directory, item 5 above, followed by the
#: same token as its coverage files.
CACHES_PREFIX: Final[str] = ".caches-"

#: Each variable naming a cache's home, and its directory under the run's.
CACHE_HOMES: Final[Sequence[tuple[str, str]]] = (
    ("HF_HOME", "huggingface"),
    ("TRITON_HOME", "triton"),
)

#: The prefix of the run's own ``HOME``, item 6 above, followed by the same
#: token as its coverage files.
HOME_PREFIX: Final[str] = ".home-"

#: What refuses the Hugging Face hub to a run that asks, item 7 above.
HUB_OFFLINE: Final[tuple[str, str]] = ("HF_HUB_OFFLINE", "1")

#: The prefix of the mutation step's scratch directory, item 8 above,
#: followed by the same token as the run's coverage files.
MUTATION_PREFIX: Final[str] = ".mutation-"

#: What makes coverage record which test ran each line, item 8 above.
TEST_CONTEXTS: Final[str] = "--cov-context=test"

#: The question that names the venv interpreter Poetry runs a project in.
POETRY_EXECUTABLE_ARGV: Final[Sequence[str]] = ("poetry", "env", "info", "--executable")

#: The BLAS thread caps, item 4 above.
THREAD_CAPS: Final[Sequence[tuple[str, str]]] = (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
)

#: The invocation through poetry, before the parallelism, coverage and
#: caller arguments.
PYTEST_ARGV: Final[Sequence[str]] = ("poetry", "run", "pytest")

#: xdist across every core, and item 2 above. Omitted for a SERIAL suite:
#: OrderedKernels' workers would each spin a CUDA context on the one GPU.
PARALLEL_ARGV: Final[Sequence[str]] = ("-n", "auto", "--max-worker-restart=0")

#: Item 2 above, kept by name for the recipes' documentation.
NO_WORKER_RESTART: Final[str] = PARALLEL_ARGV[2]

#: Wall clock on a package's test suite. Four hours, and it is deliberately
#: the largest bound in this package: the slowest suite measured here is
#: TankpitBot at 350 s and covenant-radar-api at 154 s, so four hours is two
#: orders of magnitude past any honest run. It exists for the run that is not
#: honest. A suite that hangs on a socket or a lock is the shape that drained
#: the fleet's queue for three days, and under a scheduled task there is
#: nobody at the terminal to notice the silence (board task 0d891468).
SUITE_WALL_SECONDS: Final[int] = 14400


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
    if runner is Runner.VENV:
        return [str(venv_executable(project, "python")), "-m", "pytest"]
    return list(PYTEST_ARGV)


def package_python(project: Path, runner: Runner) -> str:
    """The interpreter of the project's own environment.

    Args:
        project: The package directory.
        runner: ``venv`` for the project's own ``.venv``; ``poetry`` asks
            Poetry, which must happen while ``HOME`` is still the user's.

    Returns:
        The interpreter's path.

    Raises:
        AppError: ``MAKETOOLS_ARTIFACT_MISSING`` when Poetry names no
            interpreter for the project.
    """
    if runner is Runner.VENV:
        return str(venv_executable(project, "python"))
    answer = _test_hooks.run_capturing(POETRY_EXECUTABLE_ARGV, cwd=project)
    executable = answer["stdout"].strip()
    if answer["returncode"] != 0 or not executable:
        raise AppError(
            MaketoolsErrorCode.ARTIFACT_MISSING,
            f"poetry names no venv interpreter for {project}: {answer['stderr'].strip()}",
        )
    return executable


def home_writes(home: Path) -> list[Path]:
    """Every file a suite left under its own ``HOME``.

    Args:
        home: The run's ``HOME``.

    Returns:
        Each file, relative to ``home``, in sorted order; directories alone
        are not writes of anything.
    """
    return sorted(path.relative_to(home) for path in home.rglob("*") if not path.is_dir())


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


def run_tests(
    project: Path,
    pytest_arguments: Sequence[str],
    *,
    sweep: bool,
    serial: bool = False,
    runner: Runner = Runner.POETRY,
    own_home: bool = False,
    hub_offline: bool = False,
    mutation: bool = False,
) -> int:
    """Run the project's suite under the launcher's protections.

    Args:
        project: The project directory, which is the recipe's cwd.
        pytest_arguments: Extra arguments appended to the invocation.
        sweep: Whether to run the pre-run sweep; off for debugging only.
        serial: Run without xdist (see :data:`PARALLEL_ARGV`).
        runner: How pytest is reached (see :func:`pytest_argv`).
        own_home: Run under a ``HOME`` of the run's own and fail on any
            file left there (item 6 above).
        hub_offline: Refuse the Hugging Face hub to the suite (item 7
            above).
        mutation: Mutation test the commit's added src lines after a
            passing suite (item 8 above).

    Returns:
        pytest's exit status, 1 when it passed but left a file under its own
        ``HOME``, the mutation step's code after a passing suite that asked
        for it, or 1 when the launcher refused to start it.
    """
    _test_hooks.write_line("run-tests: the suite runs in its own session and is reaped on exit")
    if sweep:
        reap.sweep_stale(project)
    # The interpreter is named before HOME moves (item 6) and once for both
    # items that need it: the own-HOME suite and the mutation step.
    python = package_python(project, runner) if own_home or mutation else ""
    leading = [python, "-m", "pytest"] if own_home else pytest_argv(project, runner)
    runs = project / RUNS_DIRECTORY
    runs.mkdir(exist_ok=True)
    run_token = _test_hooks.token()
    token = COVERAGE_PREFIX + run_token
    caches = runs.resolve() / (CACHES_PREFIX + run_token)
    caches.mkdir(exist_ok=True)
    home = runs.resolve() / (HOME_PREFIX + run_token)
    environment = _test_hooks.environ()
    environment["COVERAGE_FILE"] = str(runs.resolve() / token)
    for name, value in THREAD_CAPS:
        environment[name] = value
    for name, directory in CACHE_HOMES:
        environment[name] = str(caches / directory)
    if own_home:
        home.mkdir(exist_ok=True)
        environment["HOME"] = str(home)
    if hub_offline:
        environment[HUB_OFFLINE[0]] = HUB_OFFLINE[1]
    parallel = () if serial else PARALLEL_ARGV
    argv = [
        *leading,
        *parallel,
        "-v",
        *coverage_arguments(project),
        *([TEST_CONTEXTS] if mutation else []),
        *pytest_arguments,
    ]
    try:
        code = _test_hooks.run_inheriting(
            argv,
            cwd=project,
            env=environment,
            new_session=True,
            timeout_seconds=SUITE_WALL_SECONDS,
        )
        written = home_writes(home) if own_home else []
        for path in written:
            _test_hooks.write_error(f"run-tests: the suite wrote {path} under its own HOME, {home}")
        if written and code == 0:
            return 1
        if mutation and code == 0:
            scratch = runs.resolve() / (MUTATION_PREFIX + run_token)
            return mutate_after_suite(project, python, environment, scratch)
        return code
    finally:
        remove_run_files(runs, token)
        _test_hooks.remove_tree(caches)
        if own_home:
            _test_hooks.remove_tree(home)
        # Reap the child tree on normal exit and propagated failures. A
        # SIGKILL cannot run this cleanup; the next pre-run sweep covers it.
        reap.reap_descendants(_test_hooks.process_id())


__all__ = [
    "CACHES_PREFIX",
    "CACHE_HOMES",
    "COVERAGE_PREFIX",
    "COVERAGE_ROOTS",
    "HOME_PREFIX",
    "HUB_OFFLINE",
    "MUTATION_PREFIX",
    "NO_WORKER_RESTART",
    "PARALLEL_ARGV",
    "POETRY_EXECUTABLE_ARGV",
    "PYTEST_ARGV",
    "RUNS_DIRECTORY",
    "TEST_CONTEXTS",
    "THREAD_CAPS",
    "Runner",
    "coverage_arguments",
    "home_writes",
    "package_python",
    "pytest_argv",
    "remove_run_files",
    "run_tests",
]
