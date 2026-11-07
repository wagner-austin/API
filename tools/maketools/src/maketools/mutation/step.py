"""The mutation step a Python package's ``make check`` runs after its suite.

Board task fec5851c, the Python half of mcp-shared's TypeScript step
(``mcp-shared/src/mutation/step.ts``, board task a48b5236): the same scope,
the same refusals and the same lines. MCPs runs it as ``python -m
mcp_shared_py.mutation`` (:mod:`.cli`); API lifts this module and the rest
of the core byte for byte and runs it from its test launcher (board task
e52e5148), so it imports only the standard library and the core.

WHAT IT MUTATES. The lines the checked commit adds to ``src`` against the
merge base of ``HEAD`` and ``origin/main`` (:mod:`.changes`), less the places
that are not code (:mod:`.sites`). cosmic-ray (:mod:`.cosmic`) enumerates the
mutants of the changed files and applies each in place.

WHICH TESTS RUN A MUTANT. Those the suite's coverage contexts name for its
line: the suite runs with ``--cov-context=test``, so its coverage data
records which test ran every line, and a mutant runs only those (0.63 s a
mutant measured on sedona, where the whole suite takes 50 s). A line run
only at import (a decorator, a module constant, a default) was run by no
test in particular, so its mutant runs the whole suite, as StrykerJS runs
every test for a static mutant. A line no test runs is a surviving mutant
without a run. Each set of tests is first run unmutated, which must pass,
and its time sets the mutant's timeout (:mod:`.trials`).

IN PLACE, NEVER OVER UNCOMMITTED WORK. The step refuses a changed file whose
bytes differ from ``HEAD``, and after the run verifies each holds ``HEAD``'s
bytes, restoring them and failing when cosmic-ray did not. A package whose
setup derives a file from its sources (mcp-shared-py's contract hash) names
that command with ``--prepare``: it runs before each set of tests, as setup
runs before the suite, and once more over the restored sources at the end.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from .changes import BASE_REF, added_lines, diff_arguments
from .context import (
    JsonCodec,
    JsonT,
    MutationErrorCode,
    MutationHooks,
    MutationStepError,
    StepContext,
    run_or_refuse,
)
from .cosmic import Site, command, decode_sites, session_config
from .report import Tally, Uncaught, UncaughtMutant, render_verdict
from .sites import SourceFacts, coverage_line, is_declaration, read_source_facts
from .trials import run_baseline, run_mutant


def changed_sources(context: StepContext[JsonT]) -> tuple[str, dict[str, frozenset[int]]]:
    """The lines the checked commit adds to source files against ``origin/main``.

    Args:
        context: The run.

    Returns:
        The merge base, and each changed ``.py`` file's added lines.

    Raises:
        MutationStepError: ``MUTATION_BASE_UNAVAILABLE`` when
            :data:`~.changes.BASE_REF` names no commit, and
            ``MUTATION_GIT_FAILED`` when git fails after.
    """
    env = context.environment
    tip = context.hooks.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{BASE_REF}^{{commit}}"], env
    )
    if tip.exit_code:
        raise MutationStepError(
            MutationErrorCode.BASE_UNAVAILABLE,
            f"{BASE_REF} names no commit here (git rev-parse exited {tip.exit_code}), so nothing "
            "says what this commit changes against main. A fleet check has it when the "
            "package's entry in API tools/fleet/registry/projects declares base_branch main; a "
            "checkout has it after git fetch.",
        )
    git = MutationErrorCode.GIT_FAILED
    base = run_or_refuse(context, ["git", "merge-base", "HEAD", BASE_REF], env, git).strip()
    return base, added_lines(run_or_refuse(context, ["git", *diff_arguments(base)], env, git))


def differing_from_head(context: StepContext[JsonT], files: list[str]) -> list[str]:
    """The files among ``files`` whose working-tree bytes differ from ``HEAD``.

    Args:
        context: The run.
        files: Tracked files, relative to the package.

    Returns:
        Each differing file, as git names it.
    """
    argv = ["git", "status", "--porcelain", "--untracked-files=no", "--", *files]
    status = run_or_refuse(context, argv, context.environment, MutationErrorCode.GIT_FAILED)
    return [line[3:] for line in status.splitlines()]


def verify_restored(context: StepContext[JsonT], files: list[str]) -> None:
    """Verify every mutated file holds ``HEAD``'s bytes again, restoring any that does not.

    The restore names ``files``, relative to the package where it runs, and
    not the names ``git status --porcelain`` printed: those are relative to
    the repository's root, which a package in a subdirectory is not (board
    task e52e5148 measured `` M pkg/src/a.py`` printed from inside ``pkg``).

    Args:
        context: The run.
        files: The files the run mutated in place.

    Raises:
        MutationStepError: ``MUTATION_TREE_NOT_RESTORED`` naming each file
            cosmic-ray left differing, once ``HEAD``'s bytes are back in it.
    """
    left = differing_from_head(context, files)
    if left:
        argv = ["git", "restore", "--source=HEAD", "--worktree", "--", *files]
        run_or_refuse(context, argv, context.environment, MutationErrorCode.GIT_FAILED)
        raise MutationStepError(
            MutationErrorCode.TREE_NOT_RESTORED,
            f"cosmic-ray left {', '.join(left)} differing from HEAD after mutating in place; "
            "HEAD's bytes were put back",
        )


def enumerate_sites(context: StepContext[JsonT], files: list[str]) -> list[Site]:
    """Every mutant cosmic-ray can make in the changed files.

    Args:
        context: The run.
        files: The changed files.

    Returns:
        The mutants, in cosmic-ray's order.

    Raises:
        MutationStepError: ``MUTATION_COSMIC_RAY_FAILED`` when ``init`` or
            ``dump`` fails.
    """
    config = context.run_dir / "cosmic-ray.toml"
    session = str(context.run_dir / "session.sqlite")
    context.hooks.write_text(config, session_config(files, context.codec))
    program = command(context.python)
    failed = MutationErrorCode.COSMIC_RAY_FAILED
    run_or_refuse(context, [program, "init", str(config), session], context.environment, failed)
    dump = run_or_refuse(context, [program, "dump", session], context.environment, failed)
    return decode_sites(dump, context.codec)


def read_contexts(
    context: StepContext[JsonT], files: list[str]
) -> dict[str, dict[int, frozenset[str]]]:
    """Which tests the suite ran each line of the changed files in.

    Args:
        context: The run.
        files: The changed files.

    Returns:
        Each file coverage measured, to each line it recorded, to the
        contexts it ran in: ``<test node id>|<phase>``, or empty for a line
        run outside any test.

    Raises:
        MutationStepError: ``MUTATION_COVERAGE_UNAVAILABLE`` when coverage
            has no data to report: the suite did not run with coverage here.
    """
    report = context.run_dir / "coverage.json"
    argv = [context.python, "-m", "coverage", "json", "--show-contexts"]
    argv += [f"--include={','.join(files)}", "-o", str(report)]
    run_or_refuse(context, argv, context.environment, MutationErrorCode.COVERAGE_UNAVAILABLE)
    codec = context.codec
    document = codec.mapping(codec.loads(context.hooks.read_text(report)), "coverage report")
    measured = codec.mapping(document["files"], "files")
    contexts: dict[str, dict[int, frozenset[str]]] = {}
    for path, entry in measured.items():
        lines = codec.mapping(codec.mapping(entry, path)["contexts"], f"{path} contexts")
        contexts[path] = {
            int(line): frozenset(
                codec.text(name, f"{path}:{line} context")
                for name in codec.sequence(names, f"{path}:{line}")
            )
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


def _judge(
    context: StepContext[JsonT],
    chosen: list[tuple[Site, tuple[str, ...] | None]],
    sources: dict[str, list[str]],
) -> list[UncaughtMutant]:
    """Run every chosen mutant and keep those that fail the step.

    Args:
        context: The run.
        chosen: Each mutant with its tests (:func:`selection`).
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
            timeouts[tests] = run_baseline(context, tests)
        mutated = run_mutant(context, site, tests, timeouts[tests])
        if mutated is not None:
            uncaught.append(_uncaught(site, mutated, Uncaught.SURVIVED))
    return uncaught


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


def _choose(
    context: StepContext[JsonT],
    sites: list[Site],
    added: dict[str, frozenset[int]],
    files: list[str],
) -> tuple[list[tuple[Site, tuple[str, ...] | None]], int, dict[str, list[str]]]:
    """The mutants on added lines that are code, each with its tests.

    Args:
        context: The run.
        sites: Every mutant of the changed files.
        added: Each changed file's added lines.
        files: The changed files.

    Returns:
        The chosen mutants with their tests, how many places were
        declarations, and each file's lines.
    """
    texts = {path: context.hooks.read_text(context.package_dir / path) for path in files}
    facts: dict[str, SourceFacts] = {path: read_source_facts(text) for path, text in texts.items()}
    contexts = read_contexts(context, files)
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


def run_mutation_step(context: StepContext[JsonT]) -> int:
    """Run the step in one package.

    Args:
        context: The run, its scratch directory made.

    Returns:
        The exit code: 1 when any mutant survived, else 0.

    Raises:
        MutationStepError: Every refusal :class:`~.context.MutationErrorCode` names.
    """
    started = context.hooks.now()
    base, added = changed_sources(context)
    files = sorted(added)
    if not files:
        context.hooks.write(
            f"mutation: not applicable: 0 lines added to src files against origin/main (merge "
            f"base {base}), so 0 mutants were made"
        )
        return 0
    dirty = differing_from_head(context, files)
    if dirty:
        raise MutationStepError(
            MutationErrorCode.TREE_DIRTY,
            f"{', '.join(dirty)} differ from HEAD; the step mutates files in place and puts "
            "HEAD's bytes back, so it runs only over committed work",
        )
    chosen, declarations, sources = _choose(context, enumerate_sites(context, files), added, files)
    uncaught = _judge(context, chosen, sources)
    _close(context, files, started)
    tally = Tally(
        files=len(files),
        base=base,
        by_tests=sum(1 for _, tests in chosen if tests),
        by_suite=sum(1 for _, tests in chosen if tests == ()),
        by_none=sum(1 for _, tests in chosen if tests is None),
        declarations=declarations,
    )
    for line in render_verdict(tally, uncaught, context.codec.dumps):
        context.hooks.write(line)
    return 1 if uncaught else 0


def _close(context: StepContext[JsonT], files: list[str], started: float) -> None:
    """Put the package back as the run found it, and record the run's time.

    Args:
        context: The run.
        files: The files the run mutated in place.
        started: When the run began, on the hooks' clock.

    Raises:
        MutationStepError: The refusals of :func:`verify_restored`, and
            ``MUTATION_PREPARE_FAILED`` when the prepare command fails over
            the restored sources; the boundary's own refusal when it cannot
            record the span.
    """
    verify_restored(context, files)
    if context.prepare:
        argv = ["sh", "-c", context.prepare]
        run_or_refuse(context, argv, context.environment, MutationErrorCode.PREPARE_FAILED)
    context.record_span(round(context.hooks.now() - started))


def run_in_scratch(
    hooks: MutationHooks,
    codec: JsonCodec[JsonT],
    package_dir: Path,
    python: str,
    prepare: str,
    environment: Mapping[str, str],
    record_span: Callable[[int], None],
) -> int:
    """Run the step in a package, in a scratch directory made for it and removed after.

    The one entry both repositories' boundaries call: MCPs' :mod:`.cli` and
    API's test launcher, each with its own hooks, codec and span sink.

    Args:
        hooks: The world the step reaches, bound to the package.
        codec: How the package reads and writes JSON.
        package_dir: The package, where every command runs.
        python: The interpreter of the package's environment.
        prepare: The command that derives files from the sources, or empty.
        environment: The environment the step runs its commands with.
        record_span: Where the step's time goes once it has made mutants.

    Returns:
        The exit code :func:`run_mutation_step` decides.
    """
    run_dir = hooks.make_run_directory()
    context = StepContext(
        hooks=hooks,
        codec=codec,
        package_dir=package_dir,
        python=python,
        prepare=prepare,
        environment=environment,
        run_dir=run_dir,
        record_span=record_span,
    )
    try:
        return run_mutation_step(context)
    finally:
        hooks.remove_tree(run_dir)


__all__ = [
    "changed_sources",
    "differing_from_head",
    "enumerate_sites",
    "read_contexts",
    "run_in_scratch",
    "run_mutation_step",
    "selection",
    "verify_restored",
]
