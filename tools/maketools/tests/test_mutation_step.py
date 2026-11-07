"""Tests for :mod:`maketools.mutation.step` (MCPs board task e52e5148).

Each command the step runs is answered by the ``world`` fixture's
``run_answering``, scripted per case by what git, cosmic-ray and coverage
print; the sources are real files, since the step reads them for their
annotations and statements. ``test_mutation_real.py`` runs the same step
against real git, coverage, pytest and cosmic-ray.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.error_codes_tooling import MutationErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str

from maketools.commands import CommandResult
from maketools.mutation.changes import BASE_REF
from maketools.mutation.cosmic import Site
from maketools.mutation.step import (
    changed_sources,
    differing_from_head,
    enumerate_sites,
    read_contexts,
    run_mutation_step,
    selection,
    verify_restored,
)
from tests._mutation_run import COSMIC_RAY, PYTHON, mutation_run
from tests.conftest import World, failed, ok

SOURCE = "\n".join(
    [
        "def at_least(value: int | None, floor: int) -> bool:",  # 1
        "    return value >= floor",  # 2
        "OLD = 1 + 1",  # 3
        "LIMIT = 3 * 1",  # 4
        "def never() -> int:",  # 5
        "    return 2 - 1",  # 6
        "",
    ]
)

DIFF = """\
diff --git a/src/p.py b/src/p.py
--- a/src/p.py
+++ b/src/p.py
@@ -0,0 +1,2 @@
+def at_least(value: int | None, floor: int) -> bool:
+    return value >= floor
@@ -1,0 +4,3 @@
+LIMIT = 3 * 1
+def never() -> int:
+    return 2 - 1
"""

ABOVE = "tests/t.py::test_above"
BELOW = "tests/t.py::test_below"

#: Each place cosmic-ray can mutate in :data:`SOURCE`: an annotation, the
#: comparison twice, an unchanged line, a line run at import and a line no
#: test runs.
SITES = [
    Site("src/p.py", 1, 24, "core/ReplaceBinaryOperator_BitOr_Add", 0),
    Site("src/p.py", 2, 17, "core/ReplaceComparisonOperator_GtE_Gt", 0),
    Site("src/p.py", 2, 17, "core/ReplaceComparisonOperator_GtE_Lt", 0),
    Site("src/p.py", 3, 8, "core/ReplaceBinaryOperator_Add_Sub", 0),
    Site("src/p.py", 4, 10, "core/ReplaceBinaryOperator_Mul_Add", 0),
    Site("src/p.py", 6, 13, "core/ReplaceBinaryOperator_Sub_Add", 0),
]

CONTEXTS = {
    "src/p.py": {
        "contexts": {
            "1": [""],
            "2": [f"{BELOW}|run", f"{ABOVE}|run", f"{ABOVE}|setup"],
            "3": [""],
            "4": [""],
            "5": [""],
        }
    }
}


def _dump(sites: list[Site]) -> str:
    """What ``cosmic-ray dump`` prints for these mutants before a run.

    Args:
        sites: The mutants.

    Returns:
        One JSON array per line.
    """
    lines: list[str] = []
    for site in sites:
        mutation = {
            "module_path": site.path,
            "operator_name": site.operator,
            "occurrence": site.occurrence,
            "start_pos": [site.line, site.column],
            "end_pos": [site.line, site.column + 1],
            "operator_args": {},
        }
        lines.append(dump_json_str([{"job_id": "j", "mutations": [mutation]}, None]) + "\n")
    return "".join(lines)


def _outcome(tests: str, diff: str) -> str:
    """What ``cosmic-ray mutate-and-test`` prints for a mutant its worker ran.

    Args:
        tests: ``survived`` or ``killed``.
        diff: The mutation.

    Returns:
        The JSON.
    """
    result = {"worker_outcome": "normal", "output": "", "test_outcome": tests, "diff": diff}
    return dump_json_str(result)


class Commands:
    """The answers of every command the step runs, for one package.

    Attributes:
        world: The bound world, whose clock a set of tests moves.
        status: What ``git status`` prints, in call order; the last repeats.
        baselines: The tests of each unmutated run, in order.
        mutated: Each mutant's operator, in the order it was run.
        diff: What ``git diff`` prints.
        base: What ``git rev-parse`` answers.
        coverage: What ``coverage json`` writes, or None to fail.
    """

    def __init__(self, world: World) -> None:
        """Answer as a package whose commit adds :data:`DIFF`.

        Args:
            world: The bound world.
        """
        self.world = world
        self.status = [""]
        self.baselines: list[tuple[str, ...]] = []
        self.mutated: list[str] = []
        self.diff = DIFF
        self.base = ok("0123abcd\n")
        self.coverage: dict[str, dict[str, dict[str, list[str]]]] | None = CONTEXTS
        world.answer = self.answer

    def answer(self, argv: tuple[str, ...]) -> CommandResult:
        """Answer one command.

        Args:
            argv: The command.

        Returns:
            What it printed.
        """
        if argv[0] == "git":
            return self.git(argv[1:])
        return self.tool(argv)

    def git(self, arguments: tuple[str, ...]) -> CommandResult:
        """Answer one git command.

        Args:
            arguments: Its arguments.

        Returns:
            What it printed.
        """
        match arguments:
            case ("rev-parse", *_):
                return self.base
            case ("merge-base", "HEAD", ref) if ref == BASE_REF:
                return ok("abc123\n")
            case ("diff", *_):
                return ok(self.diff)
            case ("status", *_):
                return ok(self.status.pop(0) if len(self.status) > 1 else self.status[0])
            case ("restore", *_):
                return ok("")
        raise AssertionError(f"unscripted git {arguments}")

    def tool(self, argv: tuple[str, ...]) -> CommandResult:
        """Answer one cosmic-ray, coverage or pytest command.

        Args:
            argv: The command.

        Returns:
            What it printed.
        """
        match argv:
            case (program, "init", *_) if program == COSMIC_RAY:
                return ok("")
            case (program, "dump", _) if program == COSMIC_RAY:
                return ok(_dump(SITES))
            case (program, "-m", "coverage", *rest) if program == PYTHON:
                if self.coverage is None:
                    return failed(1, "No data to report.")
                Path(rest[-1]).write_text(dump_json_str({"files": self.coverage}), "utf-8")
                return ok("")
            case (program, "-W", *rest) if program == PYTHON:
                self.world.now_value += 2.0
                self.baselines.append(tuple(rest[rest.index("0") + 1 :]) if "-n" in rest else ())
                return ok("")
            case (program, "mutate-and-test", _, operator, _, _) if program == COSMIC_RAY:
                self.mutated.append(operator)
                if operator.endswith("GtE_Gt"):
                    diff = "--- a/src/p.py\n+++ b/src/p.py\n@@ -2 +2 @@\n+    return value > floor"
                    return ok(_outcome("survived", diff))
                return ok(_outcome("killed", "+x"))
        raise AssertionError(f"unscripted command {argv}")


@pytest.fixture()
def package(tmp_path: Path) -> Path:
    """A package holding :data:`SOURCE`, with a scratch directory beside it.

    Returns:
        The package directory.
    """
    (tmp_path / "pkg" / "src").mkdir(parents=True)
    (tmp_path / "pkg" / "src" / "p.py").write_text(SOURCE, encoding="utf-8")
    (tmp_path / "run").mkdir()
    return tmp_path / "pkg"


def test_the_base_must_name_a_commit_and_says_where_one_comes_from(world: World) -> None:
    commands = Commands(world)
    commands.base = failed(1, "")
    with pytest.raises(AppError) as refused:
        changed_sources(mutation_run())
    assert refused.value.code is MutationErrorCode.BASE_UNAVAILABLE
    assert refused.value.message.startswith(
        f"{BASE_REF} names no commit here (git rev-parse exited 1)"
    )
    assert "declares base_branch main" in refused.value.message


def test_the_changed_sources_are_the_lines_added_since_the_merge_base(world: World) -> None:
    Commands(world)
    added = {"src/p.py": frozenset({1, 2, 4, 5, 6})}
    assert changed_sources(mutation_run()) == ("abc123", added)
    assert [call["argv"][:2] for call in world.answering_calls] == [
        ("git", "rev-parse"),
        ("git", "merge-base"),
        ("git", "diff"),
    ]
    assert world.answering_calls[2]["argv"][-4:] == ("abc123", "HEAD", "--", "src")


def test_a_git_failure_after_the_base_is_refused(world: World) -> None:
    world.answer = lambda argv: ok("x\n") if argv[1] == "rev-parse" else failed(128, "fatal")
    with pytest.raises(AppError) as refused:
        changed_sources(mutation_run())
    assert refused.value.code is MutationErrorCode.GIT_FAILED
    assert refused.value.message == f"git merge-base HEAD {BASE_REF} exited 128: fatal"


def test_differing_files_are_named_as_git_names_them(world: World) -> None:
    commands = Commands(world)
    commands.status = [" M tools/pkg/src/p.py\nM  tools/pkg/src/q.py\n"]
    files = ["src/p.py", "src/q.py"]
    named = ["tools/pkg/src/p.py", "tools/pkg/src/q.py"]
    assert differing_from_head(mutation_run(), files) == named
    assert world.answering_calls[0]["argv"] == (
        "git",
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        *files,
    )


def test_a_tree_cosmic_ray_restored_is_left_alone(world: World) -> None:
    Commands(world)
    verify_restored(mutation_run(), ["src/p.py"])
    assert [call["argv"][1] for call in world.answering_calls] == ["status"]


def test_a_tree_left_mutated_is_put_back_and_refused(world: World) -> None:
    commands = Commands(world)
    commands.status = [" M tools/pkg/src/p.py\n"]
    with pytest.raises(AppError) as refused:
        verify_restored(mutation_run(), ["src/p.py"])
    assert refused.value.code is MutationErrorCode.TREE_NOT_RESTORED
    assert refused.value.message.startswith("cosmic-ray left tools/pkg/src/p.py differing")
    assert world.answering_calls[-1]["argv"] == (
        "git",
        "restore",
        "--source=HEAD",
        "--worktree",
        "--",
        "src/p.py",
    )


def test_the_sites_are_enumerated_from_a_session_of_the_changed_files(
    world: World, tmp_path: Path
) -> None:
    Commands(world)
    run = mutation_run(run_dir=tmp_path)
    assert enumerate_sites(run, ["src/p.py"]) == SITES
    assert (tmp_path / "cosmic-ray.toml").read_text(encoding="utf-8").splitlines()[1] == (
        'module-path = ["src/p.py"]'
    )
    session = str(tmp_path / "session.sqlite")
    assert [call["argv"] for call in world.answering_calls] == [
        (COSMIC_RAY, "init", str(tmp_path / "cosmic-ray.toml"), session),
        (COSMIC_RAY, "dump", session),
    ]


def test_cosmic_ray_failing_to_enumerate_is_refused(world: World, tmp_path: Path) -> None:
    world.answer = lambda argv: failed(2, "no such module")
    with pytest.raises(AppError) as refused:
        enumerate_sites(mutation_run(run_dir=tmp_path), ["src/p.py"])
    assert refused.value.code is MutationErrorCode.COSMIC_RAY_FAILED


def test_the_contexts_are_read_per_file_and_line(world: World, tmp_path: Path) -> None:
    Commands(world)
    contexts = read_contexts(mutation_run(run_dir=tmp_path), ["src/p.py", "src/q.py"])
    assert contexts == {
        "src/p.py": {
            1: frozenset({""}),
            2: frozenset({f"{BELOW}|run", f"{ABOVE}|run", f"{ABOVE}|setup"}),
            3: frozenset({""}),
            4: frozenset({""}),
            5: frozenset({""}),
        }
    }
    report = str(tmp_path / "coverage.json")
    assert world.answering_calls[0]["argv"] == (
        PYTHON,
        "-m",
        "coverage",
        "json",
        "--show-contexts",
        "--include=src/p.py,src/q.py",
        "-o",
        report,
    )


def test_a_suite_run_without_coverage_is_refused(world: World, tmp_path: Path) -> None:
    Commands(world).coverage = None
    with pytest.raises(AppError) as refused:
        read_contexts(mutation_run(run_dir=tmp_path), ["src/p.py"])
    assert refused.value.code is MutationErrorCode.COVERAGE_UNAVAILABLE
    assert refused.value.message.endswith("exited 1: No data to report.")


def test_a_lines_tests_are_its_contexts_test_ids_once_each_sorted() -> None:
    assert selection(None) is None
    assert selection(frozenset({""})) == ()
    assert selection(frozenset({f"{BELOW}|run", f"{ABOVE}|run", f"{ABOVE}|setup", ""})) == (
        ABOVE,
        BELOW,
    )
    assert selection(frozenset({"tests/t.py::test_x[a|b]|run"})) == ("tests/t.py::test_x[a|b]",)


def test_a_commit_adding_no_source_line_mutates_nothing_and_says_so(world: World) -> None:
    Commands(world).diff = ""
    assert run_mutation_step(mutation_run()) == 0
    assert world.lines == [
        "mutation: not applicable: 0 lines added to src files against origin/main (merge base "
        "abc123), so 0 mutants were made"
    ]


def test_uncommitted_changes_to_a_changed_file_are_refused_before_any_mutant(
    world: World, package: Path
) -> None:
    Commands(world).status = [" M tools/pkg/src/p.py\n"]
    with pytest.raises(AppError) as refused:
        run_mutation_step(mutation_run(package_dir=package))
    assert refused.value.code is MutationErrorCode.TREE_DIRTY
    assert refused.value.message.startswith("tools/pkg/src/p.py differ from HEAD")
    assert all(call["argv"][0] == "git" for call in world.answering_calls)


def test_a_full_run_fails_naming_the_survivor_and_the_line_no_test_runs(
    world: World, package: Path
) -> None:
    commands = Commands(world)
    run = mutation_run(package_dir=package, run_dir=package.parent / "run")
    assert run_mutation_step(run) == 1
    assert commands.baselines == [(ABOVE, BELOW), ()]
    assert commands.mutated == [
        "core/ReplaceComparisonOperator_GtE_Gt",
        "core/ReplaceComparisonOperator_GtE_Lt",
        "core/ReplaceBinaryOperator_Mul_Add",
    ]
    assert world.lines == [
        "mutation: 4 mutant(s): 2 run by the tests covering their line, 1 by the whole suite "
        "(lines run only at import), 1 by no test; 1 more in annotations and parameter markers "
        "are not code",
        'MUTANT SURVIVED src/p.py:2:18 ReplaceComparisonOperator_GtE_Gt -> "return value > floor"',
        'MUTANT SURVIVED, NO TEST RAN IT src/p.py:6:14 ReplaceBinaryOperator_Sub_Add -> "return 2 '
        '- 1"',
        "mutation: FAILED: 2 of 4 mutant(s) made in 1 src file(s) changed against origin/main "
        "(merge base abc123) survived. Kill each with a test, or rewrite the code so the "
        "mutation cannot change what it does.",
    ]
    assert world.answering_calls[-1]["argv"][:2] == ("git", "status")


def test_a_full_run_whose_tests_kill_every_mutant_passes(world: World, package: Path) -> None:
    commands = Commands(world)
    commands.coverage = {
        "src/p.py": {"contexts": {**CONTEXTS["src/p.py"]["contexts"], "6": [f"{ABOVE}|run"]}}
    }
    world.answer = lambda argv: (
        ok(_outcome("killed", "+x")) if argv[1] == "mutate-and-test" else commands.answer(argv)
    )
    run = mutation_run(package_dir=package, run_dir=package.parent / "run")
    assert run_mutation_step(run) == 0
    assert world.lines[-1] == (
        "mutation: passed: 0 of 4 mutant(s) made in 1 src file(s) changed against origin/main "
        "(merge base abc123) survived"
    )
