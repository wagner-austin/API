"""Which tests a mutant runs, lifted from MCPs (MCPs board task b9bf5690).

MCPs' own cases for :func:`maketools.mutation.step.line_tests`
(``mcp-shared-py/tests/test_mutation_line_tests.py``) with the import path
made this package's. A decorator's line runs only at import, so coverage
records it under no test; a mutant there runs the tests recorded on any
line of the definition it decorates. Every other line keeps its own tests
or its statement's (:func:`maketools.mutation.sites.coverage_line`).
"""

from __future__ import annotations

import pytest

from maketools.mutation.sites import read_source_facts
from maketools.mutation.step import line_tests

SOURCE = "\n".join(
    [
        "class Box:",  # 1
        "    @property",  # 2
        "    def v(self) -> int:",  # 3
        "        return 1",  # 4
        "",
        "    def w(self) -> int:",  # 6
        "        return (2 +",  # 7
        "            3)",  # 8
        "",
    ]
)

#: The import's empty context on every line made at import, one test that
#: re-imports the module on the decorator's line, the property's reader on
#: its body, and the tests just outside the definition on either side.
RECORDED: dict[int, frozenset[str]] = {
    1: frozenset({"", "t.py::before|run"}),
    2: frozenset({"", "t.py::reimport|run"}),
    3: frozenset({""}),
    4: frozenset({"t.py::reader|run", "t.py::reader|setup"}),
    6: frozenset({""}),
    7: frozenset({"t.py::after|run"}),
}


def test_a_decorator_runs_the_tests_of_every_line_of_its_definition_and_no_other() -> None:
    facts = read_source_facts(SOURCE)
    assert line_tests(facts, 2, RECORDED) == ("t.py::reader", "t.py::reimport")


def test_a_definition_run_only_at_import_runs_the_whole_suite() -> None:
    facts = read_source_facts(SOURCE)
    recorded = {line: frozenset({""}) for line in (1, 2, 3, 6)}
    assert line_tests(facts, 2, recorded) == ()


def test_a_definition_coverage_never_recorded_runs_no_test() -> None:
    facts = read_source_facts(SOURCE)
    assert line_tests(facts, 2, {7: frozenset({"t.py::after|run"})}) is None


@pytest.mark.parametrize(
    ("line", "tests"),
    [(7, ("t.py::after",)), (8, ("t.py::after",)), (4, ("t.py::reader",)), (12, None)],
)
def test_any_other_line_runs_its_own_or_its_statements_tests(
    line: int, tests: tuple[str, ...] | None
) -> None:
    assert line_tests(read_source_facts(SOURCE), line, RECORDED) == tests
