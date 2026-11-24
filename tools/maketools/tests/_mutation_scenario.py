"""The lifted step's worked scenario, shared by its tests (MCPs board tasks e52e5148, 3b386c0d).

MCPs' own scenario for the same core (``mcp-shared-py/tests/_mutation_scenario.py``),
bound to API's boundary: its JSON is written with ``platform_core.json_utils``.
One commit adding a module whose mutants take every path: a module constant
run only at import (the whole suite), a comparison the tests keep alive, one
on a continued line they kill with the same tests, a function no test calls,
and an annotation that is not code; a second changed file holds a mutant on
a line the commit did not add. The JSON is the shape cosmic-ray and coverage
print (``test_mutation_cosmic.py``, and ``test_mutation_real.py`` for the
real programs), and the baseline's report the shape pytest's xunit1 JUnit
output has (``test_mutation_baseline.py``).
"""

from __future__ import annotations

import re
from pathlib import Path

from platform_core.json_utils import JSONValue, dump_json_str

from maketools.mutation.context import Ran
from tests._mutation_world import COSMIC_RAY, PACKAGE, PYPROJECT_TEXT, PYTHON, RUN_DIR, World, ok

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


def coverage_report(contexts: dict[str, list[str]]) -> str:
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


def junit_report(costs: dict[str, str]) -> str:
    """The baseline's xunit1 report, one case per file at its cost.

    Args:
        costs: Each test file to its case's ``time``.

    Returns:
        The report's text.
    """
    cases = "".join(
        f'<testcase classname="t" name="test" file="{file}" time="{seconds}" />'
        for file, seconds in costs.items()
    )
    return f'<testsuites><testsuite name="pytest">{cases}</testsuite></testsuites>'


#: The baseline round's report: the scenario's one test file, whose two cases
#: took 1.0 and 1.5 s, so a mutant they run times out at ceil(3 x 2.5) + 30 = 38 s.
JUNIT_REPORT = (
    '<testsuites><testsuite name="pytest">'
    '<testcase classname="tests.test_m" name="test_check" file="tests/test_m.py" time="1.0" />'
    '<testcase classname="tests.test_m" name="test_edge" file="tests/test_m.py" time="1.5" />'
    "</testsuite></testsuites>"
)


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
        self.junit = JUNIT_REPORT
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
            self.files[RUN_DIR / "coverage.json"] = coverage_report(self.contexts)
        report = re.search(r"--junitxml=(\S+)", argv[2]) if argv[0] == "sh" else None
        if report is not None:
            self.files[Path(report.group(1))] = self.junit
        if argv[:2] == [COSMIC_RAY, "mutate-and-test"]:
            alive = argv[3] in self.survivors
            return ok(_result("survived" if alive else "killed", "    return value > LIMIT and ("))
        return ok()


def scenario_world(times: list[float]) -> tuple[World, Commands]:
    """The world the scenario's commit is checked out in.

    Args:
        times: What the clock reads, one value per call, in order.

    Returns:
        The world and the answers it gives, which a test may change.
    """
    files: dict[Path, str] = {}
    commands = Commands(files)
    world = World(commands, times)
    world.files = files
    files[PACKAGE / "pyproject.toml"] = PYPROJECT_TEXT
    files[PACKAGE / "src/p/m.py"] = MODULE
    files[PACKAGE / "src/p/old.py"] = OLD
    return world, commands
