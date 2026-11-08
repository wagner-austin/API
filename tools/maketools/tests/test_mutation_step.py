"""Tests for :mod:`maketools.mutation.step`, lifted from MCPs (MCPs task e52e5148).

MCPs' own cases for the module (``mcp-shared-py/tests/test_mutation_step.py``),
bound to API's boundary: the JSON is read with API's codec, fixtures are
written with ``platform_core.json_utils``, and the step's span lands in the
world's list where MCPs records it through maketools' ``record-span``.
Through :mod:`tests._mutation_world`. The scenario is one commit adding a
module whose mutants take every path: a module constant run only at import
(the whole suite), a comparison the tests keep alive, one on a continued
line they kill with the same tests, a function no test calls, and an
annotation that is not code; a second changed file holds a mutant on a line
the commit did not add. The JSON is the shape cosmic-ray and coverage print
(``test_mutation_cosmic.py``, and ``test_mutation_real.py`` for the real
programs).
"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest
from platform_core.json_utils import JSONValue, dump_json_str

from maketools.mutation.context import MutationErrorCode, MutationStepError, Ran
from maketools.mutation.step import (
    differing_from_head,
    read_contexts,
    run_mutation_step,
    selection,
)
from tests._mutation_world import (
    COSMIC_RAY,
    ENVIRONMENT,
    PACKAGE,
    PYTHON,
    RUN_DIR,
    RUN_ENVIRONMENT,
    World,
    failed,
    ok,
)

MODULE = "\n".join(
    [
        "LIMIT = 3",  # 1
        "",
        "",
        "def check(value: int) -> bool:",  # 4
        "    return value >= LIMIT and (",  # 5
        "        value < 9)",  # 6
        "",
        "",
        "def unused(x: int) -> int:",  # 9
        "    return x + 1",  # 10
        "",
    ]
)
OLD = "def f(a: int) -> int:\n    b = a\n    return b * 2\n\nX = 1\n"

DIFF = (
    "diff --git a/src/p/m.py b/src/p/m.py\nnew file mode 100644\n--- /dev/null\n"
    "+++ b/src/p/m.py\n@@ -0,0 +1,6 @@\n+LIMIT = 3\n+\n+\n+def check\n+    return\n+    value\n"
    "@@ -0,0 +9,2 @@\n+def unused\n+    return\n"
    "diff --git a/src/p/old.py b/src/p/old.py\n--- a/src/p/old.py\n+++ b/src/p/old.py\n"
    "@@ -4,0 +5 @@\n+X = 1\n"
)

STATIC = ("src/p/m.py", "core/NumberReplacer", 0, 1, 8)
ANNOTATION = ("src/p/m.py", "core/ReplaceBinaryOperator_Add_Sub", 1, 4, 17)
KEPT = ("src/p/m.py", "core/ReplaceComparisonOperator_GtE_Gt", 0, 5, 17)
CONTINUED = ("src/p/m.py", "core/ReplaceComparisonOperator_Lt_LtE", 0, 6, 14)
UNRUN = ("src/p/m.py", "core/ReplaceBinaryOperator_Add_Sub", 0, 10, 13)
NOT_ADDED = ("src/p/old.py", "core/NumberReplacer", 0, 3, 15)


def _dump_line(site: tuple[str, str, int, int, int]) -> str:
    path, operator, occurrence, line, column = site
    mutation: dict[str, JSONValue] = {
        "module_path": path,
        "operator_name": operator,
        "occurrence": occurrence,
        "start_pos": [line, column],
        "end_pos": [line, column + 1],
        "operator_args": {},
        "definition_name": "x",
    }
    work_item: dict[str, JSONValue] = {"job_id": "j", "mutations": [mutation]}
    dumped: list[JSONValue] = [work_item, None]
    return dump_json_str(dumped, compact=False)


DUMP = "\n".join(map(_dump_line, [UNRUN, STATIC, ANNOTATION, KEPT, CONTINUED, NOT_ADDED]))

TESTS = ["tests/test_m.py::test_check|run", "tests/test_m.py::test_edge|run"]
CONTEXTS = {"1": [""], "4": [""], "5": [*TESTS, "tests/test_m.py::test_check|setup"], "9": [""]}


def _coverage(contexts: dict[str, list[str]]) -> str:
    """The report ``coverage json --show-contexts`` writes for the scenario.

    Args:
        contexts: Each recorded line of ``src/p/m.py`` to its contexts.

    Returns:
        The JSON text.
    """
    executed: list[JSONValue] = [*sorted(map(int, contexts))]
    named: dict[str, JSONValue] = {}
    for line, names in contexts.items():
        recorded: list[JSONValue] = [*names]
        named[line] = recorded
    measured: dict[str, JSONValue] = {"executed_lines": executed, "contexts": named}
    files: dict[str, JSONValue] = {"src/p/m.py": measured}
    meta: dict[str, JSONValue] = {"show_contexts": True}
    report: dict[str, JSONValue] = {"meta": meta, "files": files}
    return dump_json_str(report, compact=False)


def _result(outcome: str, mutated: str) -> str:
    diff = f"--- mutation diff ---\n--- a/src/p/m.py\n+++ b/src/p/m.py\n@@ -5 +5 @@\n-x\n+{mutated}"
    result: dict[str, JSONValue] = {
        "worker_outcome": "normal",
        "output": "",
        "test_outcome": outcome,
        "diff": diff,
    }
    return dump_json_str(result, compact=False)


class Commands:
    """The scenario's answers, with the git status each call reads."""

    def __init__(self, world_files: dict[Path, str]) -> None:
        self.files = world_files
        self.contexts = {**CONTEXTS}
        self.statuses = ["", ""]
        self.endings: dict[str, Ran] = {}
        self.survivors = {"core/ReplaceComparisonOperator_GtE_Gt"}

    def __call__(self, argv: list[str]) -> Ran:
        key = " ".join(argv[:3])
        if key in self.endings:
            return self.endings[key]
        if argv[:2] == ["git", "merge-base"]:
            return ok("base123\n")
        if argv[:2] == ["git", "diff"]:
            return ok(DIFF)
        if argv[:2] == ["git", "status"]:
            return ok(self.statuses.pop(0))
        if argv[:2] == [COSMIC_RAY, "dump"]:
            return ok(DUMP)
        if argv[:3] == [PYTHON, "-m", "coverage"]:
            self.files[RUN_DIR / "coverage.json"] = _coverage(self.contexts)
        if argv[:2] == [COSMIC_RAY, "mutate-and-test"]:
            alive = argv[3] in self.survivors
            return ok(_result("survived" if alive else "killed", "    return value > LIMIT and ("))
        return ok()


def _world(times: list[float]) -> tuple[World, Commands]:
    files: dict[Path, str] = {}
    commands = Commands(files)
    world = World(commands, times)
    world.files = files
    files[PACKAGE / "src/p/m.py"] = MODULE
    files[PACKAGE / "src/p/old.py"] = OLD
    return world, commands


TIMES = [1000.0, 1001.0, 1051.0, 1060.0, 1062.5, 3200.4]
PYTEST = (
    f"exec {PYTHON} -W error::SyntaxWarning -m pytest -x -q -p no:cacheprovider "
    "--basetemp=/scratch/run/pytest"
)
NAMED = f"{PYTEST} -n 0 tests/test_m.py::test_check tests/test_m.py::test_edge"


def _mutant(site: tuple[str, str, int, int, int], timeout: int, script: str) -> list[str]:
    path, operator, occurrence, _, _ = site
    tests = shlex.join(["timeout", "--kill-after=10", str(timeout), "sh", "-c", script])
    return [COSMIC_RAY, "mutate-and-test", path, operator, str(occurrence), tests]


def test_a_commit_is_mutated_judged_and_reported_in_one_run() -> None:
    world, _ = _world([*TIMES])
    assert run_mutation_step(world.context("python gen.py")) == 1
    assert world.argvs() == [
        ["git", "rev-parse", "--verify", "--quiet", "refs/remotes/origin/main^{commit}"],
        ["git", "merge-base", "HEAD", "refs/remotes/origin/main"],
        [
            "git",
            "diff",
            "--unified=0",
            "--relative",
            "--diff-filter=ACMR",
            "--no-color",
            "--no-ext-diff",
            "--src-prefix=a/",
            "--dst-prefix=b/",
            "base123",
            "HEAD",
            "--",
            "src",
        ],
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "src/p/m.py",
            "src/p/old.py",
        ],
        [COSMIC_RAY, "init", "/scratch/run/cosmic-ray.toml", "/scratch/run/session.sqlite"],
        [COSMIC_RAY, "dump", "/scratch/run/session.sqlite"],
        [
            PYTHON,
            "-m",
            "coverage",
            "json",
            "--show-contexts",
            "--include=src/p/m.py,src/p/old.py",
            "-o",
            "/scratch/run/coverage.json",
        ],
        ["sh", "-c", f"python gen.py && {PYTEST}"],
        _mutant(STATIC, 180, f"python gen.py && {PYTEST}"),
        ["sh", "-c", f"python gen.py && {NAMED}"],
        _mutant(KEPT, 38, f"python gen.py && {NAMED}"),
        _mutant(CONTINUED, 38, f"python gen.py && {NAMED}"),
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "src/p/m.py",
            "src/p/old.py",
        ],
        ["sh", "-c", "python gen.py"],
    ]
    assert world.spans == [2200]
    assert world.lines == [
        "mutation: 4 mutant(s): 2 run by the tests covering their line, 1 by the whole suite "
        "(lines run only at import), 1 by no test; 1 more in annotations and parameter "
        "markers are not code",
        'MUTANT SURVIVED src/p/m.py:5:18 ReplaceComparisonOperator_GtE_Gt -> "return value > '
        'LIMIT and ("',
        "MUTANT SURVIVED, NO TEST RAN IT src/p/m.py:10:14 ReplaceBinaryOperator_Add_Sub -> "
        '"return x + 1"',
        "mutation: FAILED: 2 of 4 mutant(s) made in 2 src file(s) changed against origin/main "
        "(merge base base123) survived. Kill each with a test, or rewrite the code so the "
        "mutation cannot change what it does.",
    ]
    assert world.files[RUN_DIR / "cosmic-ray.toml"].startswith(
        '[cosmic-ray]\nmodule-path = ["src/p/m.py", "src/p/old.py"]\n'
    )


def test_each_command_gets_the_environment_its_kind_needs() -> None:
    world, _ = _world([*TIMES])
    run_mutation_step(world.context())
    tests_runs = [env for argv, env in world.calls if argv[0] in {"sh", COSMIC_RAY}]
    tool_runs = [env for argv, env in world.calls if argv[0] in {"git", PYTHON}]
    assert all(env == RUN_ENVIRONMENT for env in tests_runs[2:])
    assert all(env == ENVIRONMENT for env in tests_runs[:2] + tool_runs)
    assert len(tests_runs) == 7
    assert len(tool_runs) == 6


def test_a_run_whose_mutants_are_all_killed_passes() -> None:
    world, commands = _world([*TIMES])
    commands.survivors = set()
    commands.contexts["10"] = TESTS
    assert run_mutation_step(world.context()) == 0
    assert world.lines == [
        "mutation: 4 mutant(s): 3 run by the tests covering their line, 1 by the whole suite "
        "(lines run only at import), 0 by no test; 1 more in annotations and parameter "
        "markers are not code",
        "mutation: passed: 0 of 4 mutant(s) made in 2 src file(s) changed against origin/main "
        "(merge base base123) survived",
    ]
    assert world.lines[-1] == (
        "mutation: passed: 0 of 4 mutant(s) made in 2 src file(s) changed against origin/main "
        "(merge base base123) survived"
    )
    assert "[cosmic-ray]" in world.files[RUN_DIR / "cosmic-ray.toml"]


def test_a_commit_adding_no_source_line_makes_no_mutant_and_records_no_span() -> None:
    world = World(lambda argv: ok("base9\n" if argv[1] == "merge-base" else ""), [5.0])
    assert run_mutation_step(world.context()) == 0
    assert world.lines == [
        "mutation: not applicable: 0 lines added to src files against origin/main (merge base "
        "base9), so 0 mutants were made"
    ]
    assert len(world.calls) == 3
    assert world.spans == []


def test_a_tree_without_origin_main_is_refused_rather_than_read_as_no_change() -> None:
    world = World(lambda argv: failed(1, ""), [5.0])
    with pytest.raises(MutationStepError) as raised:
        run_mutation_step(world.context())
    assert str(raised.value).startswith(
        "MUTATION_BASE_UNAVAILABLE: refs/remotes/origin/main names no commit here (git "
        "rev-parse exited 1), so nothing says what this commit changes against main."
    )
    assert len(world.calls) == 1


def test_a_changed_file_differing_from_head_is_refused_before_anything_is_mutated() -> None:
    world, commands = _world([*TIMES])
    commands.statuses = [" M src/p/m.py\n"]
    with pytest.raises(MutationStepError) as raised:
        run_mutation_step(world.context())
    assert str(raised.value) == (
        "MUTATION_TREE_DIRTY: src/p/m.py differ from HEAD; the step mutates files in place and "
        "puts HEAD's bytes back, so it runs only over committed work"
    )
    assert len(world.calls) == 4


def test_a_file_left_mutated_is_restored_from_head_and_refused() -> None:
    world, commands = _world([*TIMES])
    commands.statuses = ["", " M src/p/m.py\n M src/p/old.py\n"]
    with pytest.raises(MutationStepError) as raised:
        run_mutation_step(world.context())
    assert str(raised.value) == (
        "MUTATION_TREE_NOT_RESTORED: cosmic-ray left src/p/m.py, src/p/old.py differing from "
        "HEAD after mutating in place; HEAD's bytes were put back"
    )
    assert world.argvs()[-1] == [
        "git",
        "restore",
        "--source=HEAD",
        "--worktree",
        "--",
        "src/p/m.py",
        "src/p/old.py",
    ]


@pytest.mark.parametrize(
    ("key", "code"),
    [
        (f"{PYTHON} -m coverage", MutationErrorCode.COVERAGE_UNAVAILABLE),
        (f"{COSMIC_RAY} init /scratch/run/cosmic-ray.toml", MutationErrorCode.COSMIC_RAY_FAILED),
        ("sh -c python gen.py", MutationErrorCode.PREPARE_FAILED),
    ],
)
def test_a_failing_tool_is_refused_with_its_code(key: str, code: MutationErrorCode) -> None:
    world, commands = _world([*TIMES])
    commands.endings[key] = failed(1, "broken")
    with pytest.raises(MutationStepError) as raised:
        run_mutation_step(world.context("python gen.py"))
    assert raised.value.code == code
    assert str(raised.value).endswith("exited 1: broken")


def test_git_status_names_each_differing_file_after_its_two_status_letters() -> None:
    world = World(lambda argv: ok(" M src/a.py\nMM src/b c.py\n"), [])
    assert differing_from_head(world.context(), ["src/a.py"]) == ["src/a.py", "src/b c.py"]


def test_coverage_contexts_decode_per_file_per_line() -> None:
    world = World(lambda argv: ok(), [])
    world.files[RUN_DIR / "coverage.json"] = _coverage(CONTEXTS)
    assert read_contexts(world.context(), ["src/p/m.py"]) == {
        "src/p/m.py": {
            1: frozenset({""}),
            4: frozenset({""}),
            5: frozenset({*TESTS, "tests/test_m.py::test_check|setup"}),
            9: frozenset({""}),
        }
    }


@pytest.mark.parametrize(
    ("contexts", "tests"),
    [
        (None, None),
        (frozenset({""}), ()),
        (
            frozenset({"", "tests/a.py::t|run", "tests/a.py::t|setup", "tests/a.py::u[x|y]|run"}),
            ("tests/a.py::t", "tests/a.py::u[x|y]"),
        ),
        # MCPs board task 52017b1e, fleet job 0ab3cfd3: an xdist_group's
        # ``@<group>`` suffix is dropped, an ``@`` inside a parameter is
        # kept, and a group suffix after a parameter goes.
        (
            frozenset(
                {
                    "tests/b.py::t@fleet-lock|run",
                    "tests/b.py::u[a@b]|run",
                    "tests/b.py::v[c@d]@fleet-lock|call",
                }
            ),
            ("tests/b.py::t", "tests/b.py::u[a@b]", "tests/b.py::v[c@d]"),
        ),
    ],
)
def test_a_lines_tests_are_its_context_names_less_phase_and_import(
    contexts: frozenset[str] | None, tests: tuple[str, ...] | None
) -> None:
    assert selection(contexts) == tests
