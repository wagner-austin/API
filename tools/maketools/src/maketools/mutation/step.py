"""The mutation step an API package's ``make test`` runs after a passing suite (MCPs task e52e5148).

The API half of MCPs' ``mcp_shared_py.mutation.step`` (MCPs board task
fec5851c): the same scope, the same refusals and the same lines.

WHAT IT MUTATES. The lines the checked commit adds to ``src`` against the
merge base of ``HEAD`` and ``origin/main`` (:mod:`.changes`, lifted), less
the places that are not code (:mod:`.sites`, lifted). cosmic-ray
(:mod:`.cosmic`) enumerates the mutants of the changed files and applies
each in place.

WHICH TESTS RUN A MUTANT. Those the suite's coverage contexts name for its
line: with ``--mutation`` the launcher runs the suite with
``--cov-context=test``, so the run's coverage file records which test ran
every line, and a mutant runs only those. A line run only at import (a
decorator, a module constant, a default) was run by no test in particular,
so its mutant runs the whole suite. A line no test runs is a surviving
mutant without a run. Each set of tests is first run unmutated, which must
pass, and its time sets the mutant's timeout (:mod:`.trials`).

IN PLACE, NEVER OVER UNCOMMITTED WORK. The step refuses a changed file whose
bytes differ from ``HEAD``, and after the run verifies each holds ``HEAD``'s
bytes, restoring them and failing when cosmic-ray did not.
"""

from __future__ import annotations

from platform_core.error_codes_tooling import MutationErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
    narrow_json_to_str,
    require_dict,
)

from maketools import _test_hooks
from maketools.mutation.changes import BASE_REF, added_lines, diff_arguments
from maketools.mutation.context import MutationRun, run_command, run_or_refuse
from maketools.mutation.cosmic import Site, command, decode_sites, session_config
from maketools.mutation.report import Tally, Uncaught, UncaughtMutant, render_verdict
from maketools.mutation.sites import SourceFacts, coverage_line, is_declaration, read_source_facts
from maketools.mutation.trials import run_baseline, run_mutant


def changed_sources(run: MutationRun) -> tuple[str, dict[str, frozenset[int]]]:
    """The lines the checked commit adds to source files against ``origin/main``.

    Args:
        run: The step's run.

    Returns:
        The merge base, and each changed ``.py`` file's added lines.

    Raises:
        AppError: ``MUTATION_BASE_UNAVAILABLE`` when
            :data:`~.changes.BASE_REF` names no commit, and
            ``MUTATION_GIT_FAILED`` when git fails after.
    """
    env = run.environment
    question = ["git", "rev-parse", "--verify", "--quiet", f"{BASE_REF}^{{commit}}"]
    tip = run_command(run, question, env)
    if tip["returncode"]:
        raise AppError(
            MutationErrorCode.BASE_UNAVAILABLE,
            f"{BASE_REF} names no commit here (git rev-parse exited {tip['returncode']}), so "
            "nothing says what this commit changes against main. A fleet check has it when the "
            "package's entry in tools/fleet/registry/projects declares base_branch main; a "
            "checkout has it after git fetch.",
        )
    git = MutationErrorCode.GIT_FAILED
    base = run_or_refuse(run, ["git", "merge-base", "HEAD", BASE_REF], env, git).strip()
    return base, added_lines(run_or_refuse(run, ["git", *diff_arguments(base)], env, git))


def differing_from_head(run: MutationRun, files: list[str]) -> list[str]:
    """The files among ``files`` whose working-tree bytes differ from ``HEAD``.

    Args:
        run: The step's run.
        files: Tracked files, relative to the package.

    Returns:
        Each differing file, as git names it relative to the repository.
    """
    argv = ["git", "status", "--porcelain", "--untracked-files=no", "--", *files]
    status = run_or_refuse(run, argv, run.environment, MutationErrorCode.GIT_FAILED)
    return [line[3:] for line in status.splitlines()]


def verify_restored(run: MutationRun, files: list[str]) -> None:
    """Verify every mutated file holds ``HEAD``'s bytes again, restoring any that does not.

    Args:
        run: The step's run.
        files: The files the run mutated in place.

    Raises:
        AppError: ``MUTATION_TREE_NOT_RESTORED`` naming each file cosmic-ray
            left differing, once ``HEAD``'s bytes are back in it.
    """
    left = differing_from_head(run, files)
    if left:
        argv = ["git", "restore", "--source=HEAD", "--worktree", "--", *files]
        run_or_refuse(run, argv, run.environment, MutationErrorCode.GIT_FAILED)
        raise AppError(
            MutationErrorCode.TREE_NOT_RESTORED,
            f"cosmic-ray left {', '.join(left)} differing from HEAD after mutating in place; "
            "HEAD's bytes were put back",
        )


def enumerate_sites(run: MutationRun, files: list[str]) -> list[Site]:
    """Every mutant cosmic-ray can make in the changed files.

    Args:
        run: The step's run.
        files: The changed files.

    Returns:
        The mutants, in cosmic-ray's order.

    Raises:
        AppError: ``MUTATION_COSMIC_RAY_FAILED`` when ``init`` or ``dump``
            fails.
    """
    config = run.run_dir / "cosmic-ray.toml"
    session = str(run.run_dir / "session.sqlite")
    config.write_text(session_config(files), encoding="utf-8")
    program = command(run.python)
    failed = MutationErrorCode.COSMIC_RAY_FAILED
    run_or_refuse(run, [program, "init", str(config), session], run.environment, failed)
    return decode_sites(run_or_refuse(run, [program, "dump", session], run.environment, failed))


def read_contexts(run: MutationRun, files: list[str]) -> dict[str, dict[int, frozenset[str]]]:
    """Which tests the suite ran each line of the changed files in.

    Args:
        run: The step's run.
        files: The changed files.

    Returns:
        Each file coverage measured, to each line it recorded, to the
        contexts it ran in: ``<test node id>|<phase>``, or empty for a line
        run outside any test.

    Raises:
        AppError: ``MUTATION_COVERAGE_UNAVAILABLE`` when coverage has no
            data to report: the suite did not run with coverage here.
    """
    report = run.run_dir / "coverage.json"
    argv = [run.python, "-m", "coverage", "json", "--show-contexts"]
    argv += [f"--include={','.join(files)}", "-o", str(report)]
    run_or_refuse(run, argv, run.environment, MutationErrorCode.COVERAGE_UNAVAILABLE)
    measured = require_dict(narrow_json_to_dict(load_json_str(report.read_text("utf-8"))), "files")
    contexts: dict[str, dict[int, frozenset[str]]] = {}
    for path, entry in measured.items():
        lines = require_dict(narrow_json_to_dict(entry), "contexts")
        contexts[path] = {
            int(line): frozenset(narrow_json_to_str(name) for name in narrow_json_to_list(names))
            for line, names in lines.items()
        }
    return contexts


def selection(contexts: frozenset[str] | None) -> tuple[str, ...] | None:
    """The tests a mutant runs, from its line's coverage contexts.

    Args:
        contexts: The contexts coverage recorded for the line, or None when
            it recorded none.

    Returns:
        None when no test runs the line; the empty tuple for a line run
        only outside a test, which the whole suite runs; else the node ids
        of the tests that ran it, sorted.
    """
    if contexts is None:
        return None
    return tuple(sorted({name.rsplit("|", 1)[0] for name in contexts if name}))


def _uncaught(site: Site, shown: str, why: Uncaught) -> UncaughtMutant:
    """Name a mutant that fails the step.

    Args:
        site: The mutant.
        shown: The line its report shows.
        why: Why it fails the step.

    Returns:
        The report's record of it.
    """
    return UncaughtMutant(
        path=site.path,
        line=site.line,
        column=site.column,
        operator=site.operator,
        mutated=shown,
        why=why,
    )


def judge(
    run: MutationRun,
    chosen: list[tuple[Site, tuple[str, ...] | None]],
    sources: dict[str, list[str]],
) -> list[UncaughtMutant]:
    """Run every chosen mutant and keep those that fail the step.

    Args:
        run: The step's run.
        chosen: Each mutant with its tests (:func:`selection`): node ids,
            the empty tuple for the whole suite, or None when no test runs
            its line.
        sources: Each changed file's lines, for a mutant no test ran.

    Returns:
        The surviving mutants and those no test runs.
    """
    timeouts: dict[tuple[str, ...], int] = {}
    uncaught: list[UncaughtMutant] = []
    for site, tests in chosen:
        if tests is None:
            original = sources[site.path][site.line - 1].strip()
            uncaught.append(_uncaught(site, original, Uncaught.NO_TEST))
            continue
        if tests not in timeouts:
            timeouts[tests] = run_baseline(run, tests)
        mutated = run_mutant(run, site, tests, timeouts[tests])
        if mutated is not None:
            uncaught.append(_uncaught(site, mutated, Uncaught.SURVIVED))
    return uncaught


def choose(
    run: MutationRun, sites: list[Site], added: dict[str, frozenset[int]], files: list[str]
) -> tuple[list[tuple[Site, tuple[str, ...] | None]], int, dict[str, list[str]]]:
    """The mutants on added lines that are code, each with its tests.

    Args:
        run: The step's run.
        sites: Every mutant of the changed files.
        added: Each changed file's added lines.
        files: The changed files.

    Returns:
        The chosen mutants with their tests, how many places were
        declarations, and each file's lines.
    """
    texts = {path: (run.package_dir / path).read_text(encoding="utf-8") for path in files}
    facts: dict[str, SourceFacts] = {path: read_source_facts(text) for path, text in texts.items()}
    contexts = read_contexts(run, files)
    on_added = [site for site in sites if site.line in added[site.path]]
    declared = [
        site for site in on_added if is_declaration(facts[site.path], (site.line, site.column))
    ]
    chosen: list[tuple[Site, tuple[str, ...] | None]] = []
    for site in on_added:
        if site not in declared:
            recorded = contexts.get(site.path, {})
            line = coverage_line(facts[site.path], site.line, frozenset(recorded))
            chosen.append((site, selection(recorded.get(line))))
    sources = {path: text.splitlines() for path, text in texts.items()}
    return chosen, len(declared), sources


def run_mutation_step(run: MutationRun) -> int:
    """Run the step in one package, after its suite passed.

    Args:
        run: The step's run, its scratch directory made.

    Returns:
        1 when any mutant survived, else 0.

    Raises:
        AppError: Every refusal :class:`MutationErrorCode` names.
    """
    base, added = changed_sources(run)
    files = sorted(added)
    if not files:
        _test_hooks.write_line(
            f"mutation: not applicable: 0 lines added to src files against origin/main (merge "
            f"base {base}), so 0 mutants were made"
        )
        return 0
    dirty = differing_from_head(run, files)
    if dirty:
        raise AppError(
            MutationErrorCode.TREE_DIRTY,
            f"{', '.join(dirty)} differ from HEAD; the step mutates files in place and puts "
            "HEAD's bytes back, so it runs only over committed work",
        )
    chosen, declarations, sources = choose(run, enumerate_sites(run, files), added, files)
    uncaught = judge(run, chosen, sources)
    verify_restored(run, files)
    tally = Tally(
        files=len(files),
        base=base,
        by_tests=sum(1 for _, tests in chosen if tests),
        by_suite=sum(1 for _, tests in chosen if tests == ()),
        by_none=sum(1 for _, tests in chosen if tests is None),
        declarations=declarations,
    )
    for line in render_verdict(tally, uncaught):
        _test_hooks.write_line(line)
    return 1 if uncaught else 0


__all__ = [
    "changed_sources",
    "choose",
    "differing_from_head",
    "enumerate_sites",
    "judge",
    "read_contexts",
    "run_mutation_step",
    "selection",
    "verify_restored",
]
