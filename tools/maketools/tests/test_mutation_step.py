"""Tests for :mod:`maketools.mutation.step`, lifted from MCPs (MCPs tasks e52e5148, 3b386c0d).

MCPs' own cases for the module (``mcp-shared-py/tests/test_mutation_step.py``),
bound to API's boundary: the JSON is read with API's codec, fixtures are
written with ``platform_core.json_utils``, the step's span lands in the
world's list where MCPs records it through maketools' ``record-span``, and
MCPs' cases for its own command line (``lease-bounded``, a failing
``record-span``) have no counterpart here, where the launcher runs the step
in process (``test_mutation_launch.py``). Through :mod:`tests._mutation_world`,
over the commit :mod:`tests._mutation_scenario` describes: a module constant
run only at import (the whole suite), a comparison the tests keep alive, one
on a continued line they kill with the same tests, a function no test calls,
an annotation that is not code, and a mutant on a line the commit did not add.
"""

from __future__ import annotations

import shlex

import pytest

from maketools.mutation.context import MutationErrorCode, MutationStepError, Ran
from maketools.mutation.step import (
    differing_from_head,
    read_contexts,
    run_mutation_step,
    selection,
)
from tests._mutation_scenario import (
    CONTEXTS,
    CONTINUED,
    KEPT,
    STATIC,
    TESTS,
    coverage_report,
    junit_report,
    scenario_world,
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

# Each clock read, in order: the start; before STATIC the bound's read
# (1000.5), then both baselines before any trial (MCPs board task 3b386c0d):
# the whole suite's timing (1001.0 to 1051.0) around its own wall time (the
# same, 50 s, which sets STATIC's timeout to 3 x 50 + 30 = 180 s) and the one
# round's (1051.0 to 1053.5), whose report gives KEPT's tests 2.5 s; STATIC's
# trial; before KEPT the bound's read (1055.0) and its trial, as before
# CONTINUED (1064.5); then the span, 3200.4 - 1000.0.
TIMES = [
    1000.0,
    1000.5,
    1001.0,
    1001.0,
    1051.0,
    1051.0,
    1051.0,
    1053.5,
    1053.5,
    1054.5,
    1055.0,
    1063.0,
    1064.0,
    1064.5,
    1065.0,
    1066.0,
    3200.4,
]
TIMING_LINES = [
    "mutation timing: baseline selected runs=1 seconds=2.500",
    "mutation timing: baseline whole-suite runs=1 seconds=50.000",
    "mutation timing: mutant selected runs=2 seconds=2.000",
    "mutation timing: mutant whole-suite runs=1 seconds=1.000",
    "mutation timing: slow baseline whole-suite seconds=50.000 tests=0",
    "mutation timing: slow baseline selected seconds=2.500 tests=2",
    "mutation timing: slow mutant whole-suite seconds=1.000 "
    "src/p/m.py:1:9 core/NumberReplacer occurrence=0 tests=0",
    "mutation timing: slow mutant selected seconds=1.000 "
    "src/p/m.py:5:18 core/ReplaceComparisonOperator_GtE_Gt occurrence=0 tests=2",
    "mutation timing: slow mutant selected seconds=1.000 "
    "src/p/m.py:6:15 core/ReplaceComparisonOperator_Lt_LtE occurrence=0 tests=2",
]
PYTEST = f"exec {PYTHON} -W error::SyntaxWarning -m pytest -x -q -p no:cacheprovider"
JUNIT = (
    "--junitxml=/scratch/run/baseline-1.junit.xml -o junit_family=xunit1 "
    "-o junit_duration_report=total"
)
NAMED = f"{PYTEST} -n 0 tests/test_m.py::test_check tests/test_m.py::test_edge"


def _mutant(site: tuple[str, str, int, int, int], timeout: int, script: str) -> list[str]:
    path, operator, occurrence, _, _ = site
    tests = shlex.join(["timeout", "--kill-after=10", str(timeout), "sh", "-c", script])
    return [COSMIC_RAY, "mutate-and-test", path, operator, str(occurrence), tests]


def test_a_commit_is_mutated_judged_and_reported_in_one_run() -> None:
    world, _ = scenario_world([*TIMES])
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
        [
            "sh",
            "-c",
            f"python gen.py && {PYTEST} {JUNIT} -n 0 tests/test_m.py::test_check "
            "tests/test_m.py::test_edge",
        ],
        _mutant(STATIC, 180, f"python gen.py && {PYTEST}"),
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
        *TIMING_LINES,
        "mutation: 4 mutant(s): 2 run by the tests covering their line, 1 by the whole suite "
        "(lines run only at import), 1 by no test; 1 more in annotations, parameter markers "
        "and typing-only decorators are not code",
        'MUTANT SURVIVED src/p/m.py:5:18 ReplaceComparisonOperator_GtE_Gt -> "return value > '
        'LIMIT and ("',
        "MUTANT SURVIVED, NO TEST RAN IT src/p/m.py:10:14 ReplaceBinaryOperator_Add_Sub -> "
        '"return x + 1"',
        "mutation: FAILED: 2 of 4 mutant(s) made in 2 source file(s) changed against "
        "origin/main "
        "(merge base base123) survived. Kill each with a test, or rewrite the code so the "
        "mutation cannot change what it does.",
    ]
    assert world.files[RUN_DIR / "cosmic-ray.toml"].startswith(
        '[cosmic-ray]\nmodule-path = ["src/p/m.py", "src/p/old.py"]\n'
    )
    # The bound is read once, after the mutants are made and before the first trial.
    assert world.bound_reads == 1
    assert world.times == []


def test_each_command_gets_the_environment_its_kind_needs() -> None:
    world, _ = scenario_world([*TIMES])
    run_mutation_step(world.context())
    tests_runs = [env for argv, env in world.calls if argv[0] in {"sh", COSMIC_RAY}]
    tool_runs = [env for argv, env in world.calls if argv[0] in {"git", PYTHON}]
    assert all(env == RUN_ENVIRONMENT for env in tests_runs[2:])
    assert all(env == ENVIRONMENT for env in tests_runs[:2] + tool_runs)
    # cosmic-ray's init and dump, the two baselines and three trials; git's
    # five questions and coverage's report.
    assert len(tests_runs) == 7
    assert len(tool_runs) == 6


def test_one_round_proves_every_files_share_and_each_mutant_runs_its_cheapest_file_first() -> None:
    # No line runs only at import here, so no mutant needs the whole suite,
    # and each file has one share, so one round runs them all; KEPT's tests
    # span two files, test_m.py's (2.5 s) before test_a.py's (4.0 s). Reads:
    # the start, the bound's before each of three trials and their timings,
    # the round's timing (2 to 3) and the span; 13 in all.
    world, commands = scenario_world([float(i) for i in range(13)])
    commands.contexts = {
        "1": ["tests/test_z.py::t|run"],
        "4": [""],
        "5": [*TESTS, "tests/test_a.py::u|run"],
        "9": [""],
    }
    commands.junit = junit_report(
        {"tests/test_a.py": "4.0", "tests/test_m.py": "2.5", "tests/test_z.py": "0.5"}
    )
    assert run_mutation_step(world.context()) == 1
    trials = [argv for argv in world.argvs() if argv[0] == "sh" or argv[1] == "mutate-and-test"]
    kept = "tests/test_m.py::test_check tests/test_m.py::test_edge tests/test_a.py::u"
    assert trials == [
        [
            "sh",
            "-c",
            f"{PYTEST} {JUNIT} -n 0 tests/test_a.py::u tests/test_m.py::test_check "
            "tests/test_m.py::test_edge tests/test_z.py::t",
        ],
        # ceil(3 x 0.5) + 30 and ceil(3 x 6.5) + 30.
        _mutant(STATIC, 32, f"{PYTEST} -n 0 tests/test_z.py::t"),
        _mutant(KEPT, 50, f"{PYTEST} -n 0 {kept}"),
        _mutant(CONTINUED, 50, f"{PYTEST} -n 0 {kept}"),
    ]
    assert world.lines[:2] == [
        "mutation timing: baseline selected runs=1 seconds=1.000",
        "mutation timing: baseline whole-suite runs=0 seconds=0.000",
    ]
    assert world.times == []


def test_a_run_whose_mutants_are_all_killed_passes() -> None:
    # One clock read per second: the start; the bound's read before each of
    # the four trials and its timing; before the first (UNRUN, which a test
    # now runs) the whole suite's timing (2 to 5) around its wall time (3 to
    # 4) and the round's (6 to 7); then the span; 20 reads in all.
    world, commands = scenario_world([float(i) for i in range(20)])
    commands.survivors = set()
    commands.contexts["10"] = TESTS
    assert run_mutation_step(world.context()) == 0
    assert world.lines == [
        "mutation timing: baseline selected runs=1 seconds=1.000",
        "mutation timing: baseline whole-suite runs=1 seconds=3.000",
        "mutation timing: mutant selected runs=3 seconds=3.000",
        "mutation timing: mutant whole-suite runs=1 seconds=1.000",
        "mutation timing: slow baseline whole-suite seconds=3.000 tests=0",
        "mutation timing: slow baseline selected seconds=1.000 tests=2",
        "mutation timing: slow mutant selected seconds=1.000 "
        "src/p/m.py:10:14 core/ReplaceBinaryOperator_Add_Sub occurrence=0 tests=2",
        "mutation timing: slow mutant whole-suite seconds=1.000 "
        "src/p/m.py:1:9 core/NumberReplacer occurrence=0 tests=0",
        "mutation timing: slow mutant selected seconds=1.000 "
        "src/p/m.py:5:18 core/ReplaceComparisonOperator_GtE_Gt occurrence=0 tests=2",
        "mutation: 4 mutant(s): 3 run by the tests covering their line, 1 by the whole suite "
        "(lines run only at import), 0 by no test; 1 more in annotations, parameter markers "
        "and typing-only decorators are not code",
        "mutation: passed: 0 of 4 mutant(s) made in 2 source file(s) changed against "
        "origin/main "
        "(merge base base123) survived",
    ]
    # The clock read 0 when the step began and 19 at its span.
    assert world.spans == [19]
    assert "[cosmic-ray]" in world.files[RUN_DIR / "cosmic-ray.toml"]
    assert world.times == []


#: TIMES up to the bound's read before KEPT, then the clock at the span.
TO_KEPT = [*TIMES[:11], 1056.0]


@pytest.mark.parametrize("bound", [50, 55])
def test_a_run_whose_bound_comes_starts_no_more_trials_names_them_and_refuses(bound: int) -> None:
    # STATIC runs from 1000.5; the bound's read before KEPT is 1055.0, at or
    # past the deadline 1000.0 + bound, so neither KEPT nor CONTINUED starts
    # and the clock is not read again before CONTINUED.
    world, _ = scenario_world([*TO_KEPT])
    with pytest.raises(MutationStepError) as raised:
        run_mutation_step(world.context(bound=bound))
    assert str(raised.value) == (
        "MUTATION_OVER_BOUND: 1 of the 3 mutant(s) the tests run in 2 source file(s) had been "
        f"run when the step's {bound} s bound came, so 2 were not run, beside 1 already "
        "uncaught, each named above; HEAD's bytes are back in each file it mutates and its "
        "span is recorded. A change too large to mutate inside the bound is one to split"
    )
    assert world.lines == [
        "mutation timing: baseline selected runs=1 seconds=2.500",
        "mutation timing: baseline whole-suite runs=1 seconds=50.000",
        "mutation timing: mutant selected runs=0 seconds=0.000",
        "mutation timing: mutant whole-suite runs=1 seconds=1.000",
        "mutation timing: slow baseline whole-suite seconds=50.000 tests=0",
        "mutation timing: slow baseline selected seconds=2.500 tests=2",
        "mutation timing: slow mutant whole-suite seconds=1.000 "
        "src/p/m.py:1:9 core/NumberReplacer occurrence=0 tests=0",
        "mutation: 4 mutant(s): 2 run by the tests covering their line, 1 by the whole suite "
        "(lines run only at import), 1 by no test; 1 more in annotations, parameter markers "
        "and typing-only decorators are not code",
        "MUTANT NOT RUN, THE BOUND CAME src/p/m.py:5:18 ReplaceComparisonOperator_GtE_Gt -> "
        '"return value >= LIMIT and ("',
        "MUTANT NOT RUN, THE BOUND CAME src/p/m.py:6:15 ReplaceComparisonOperator_Lt_LtE -> "
        '"value < 9)"',
        "MUTANT SURVIVED, NO TEST RAN IT src/p/m.py:10:14 ReplaceBinaryOperator_Add_Sub -> "
        '"return x + 1"',
    ]
    # HEAD's bytes verified, then the span recorded: 1056.0 - 1000.0.
    assert world.argvs()[-1][:3] == ["git", "status", "--porcelain"]
    assert world.spans == [56]
    assert world.times == []


def test_a_bound_one_second_past_the_read_lets_that_trial_start() -> None:
    # The deadline 1056.0 is past the read before KEPT (1055.0), so KEPT runs
    # and survives; the read before CONTINUED (1064.5) is past it.
    world, _ = scenario_world([*TIMES[:14], 1070.0])
    with pytest.raises(MutationStepError) as raised:
        run_mutation_step(world.context(bound=56))
    assert str(raised.value).startswith(
        "MUTATION_OVER_BOUND: 2 of the 3 mutant(s) the tests run in 2 source file(s) had been "
        "run when the step's 56 s bound came, so 1 were not run, beside 2 already uncaught"
    )
    assert [line for line in world.lines if line.startswith("MUTANT ")] == [
        'MUTANT SURVIVED src/p/m.py:5:18 ReplaceComparisonOperator_GtE_Gt -> "return value > '
        'LIMIT and ("',
        "MUTANT NOT RUN, THE BOUND CAME src/p/m.py:6:15 ReplaceComparisonOperator_Lt_LtE -> "
        '"value < 9)"',
        "MUTANT SURVIVED, NO TEST RAN IT src/p/m.py:10:14 ReplaceBinaryOperator_Add_Sub -> "
        '"return x + 1"',
    ]
    assert world.times == []


def test_a_commit_adding_no_source_line_makes_no_mutant_and_records_no_span() -> None:
    world = World(lambda argv: ok("base9\n" if argv[1] == "merge-base" else ""), [5.0])
    assert run_mutation_step(world.context()) == 0
    # A run with nothing to mutate has no use for its bound, so never reads it.
    assert world.bound_reads == 0
    assert world.lines == [
        "mutation: not applicable: 0 lines added to the sources pyproject.toml "
        "[tool.coverage.run] names (src) against origin/main (merge base base9), so 0 mutants "
        "were made"
    ]
    assert len(world.calls) == 3
    assert world.calls[2][0][-2:] == ["--", "src"]
    assert world.spans == []


def test_the_diff_names_every_source_root_and_drops_the_files_omit_matches() -> None:
    hooks_diff = (
        "diff --git a/board-watch.py b/board-watch.py\n+++ b/board-watch.py\n@@ -1,0 +2 @@\n+x\n"
        "diff --git a/board-watch.test.py b/board-watch.test.py\n+++ b/board-watch.test.py\n"
        "@@ -1,0 +2 @@\n+y\n"
        "diff --git a/tests/test_a.py b/tests/test_a.py\n+++ b/tests/test_a.py\n@@ -1,0 +2 @@\n+z\n"
    )

    def answer(argv: list[str]) -> Ran:
        if argv[1] == "merge-base":
            return ok("base9\n")
        return ok(hooks_diff if argv[1] == "diff" else "")

    world = World(answer, [5.0, 6.0])
    world.files[PACKAGE / "pyproject.toml"] = (
        '[tool.coverage.run]\nsource = [".", "scripts"]\nomit = ["*.test.py", "tests/*"]\n'
    )
    world.files[PACKAGE / "board-watch.py"] = "a = 1\nx = 2\n"
    world.files[RUN_DIR / "coverage.json"] = '{"files": {}}'
    assert run_mutation_step(world.context()) == 0
    assert world.calls[2][0][-3:] == ["--", ".", "scripts"]
    assert world.files[RUN_DIR / "cosmic-ray.toml"].startswith(
        '[cosmic-ray]\nmodule-path = ["board-watch.py"]\n'
    )
    assert world.lines[-1] == (
        "mutation: passed: 0 of 0 mutant(s) made in 1 source file(s) changed against "
        "origin/main (merge base base9) survived"
    )


def test_a_package_declaring_no_coverage_source_is_refused_before_git_runs() -> None:
    world = World(lambda argv: ok(), [5.0])
    world.files[PACKAGE / "pyproject.toml"] = "[tool.coverage.run]\nbranch = true\n"
    with pytest.raises(MutationStepError) as raised:
        run_mutation_step(world.context())
    assert raised.value.code == MutationErrorCode.SOURCES_UNDECLARED
    assert world.calls == []


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
    world, commands = scenario_world([*TIMES])
    commands.statuses = [" M src/p/m.py\n"]
    with pytest.raises(MutationStepError) as raised:
        run_mutation_step(world.context())
    assert str(raised.value) == (
        "MUTATION_TREE_DIRTY: src/p/m.py differ from HEAD; the step mutates files in place and "
        "puts HEAD's bytes back, so it runs only over committed work"
    )
    assert len(world.calls) == 4


def test_a_file_left_mutated_is_restored_from_head_and_refused() -> None:
    world, commands = scenario_world([*TIMES])
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
    world, commands = scenario_world([*TIMES])
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
    world.files[RUN_DIR / "coverage.json"] = coverage_report(CONTEXTS)
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
