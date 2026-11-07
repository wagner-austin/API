"""Tests for :mod:`maketools.mutation.sites`, lifted from MCPs (MCPs task e52e5148).

The module is MCPs' ``mcp_shared_py.mutation.sites`` byte for byte, and
these are MCPs' own cases for it (``mcp-shared-py/tests/
test_mutation_sites.py``, board task fec5851c) with the import path made
this package's. One module exercises every place: a non-ASCII parameter
name before an annotation (ast counts UTF-8 bytes, cosmic-ray characters),
an annotation spread over two lines, parameter markers beside the same
characters used as operators, and statements continued over several lines.
"""

from __future__ import annotations

from maketools.mutation.sites import (
    coverage_line,
    is_declaration,
    read_source_facts,
)

SOURCE = "\n".join(
    [
        "from __future__ import annotations",  # 1
        "",
        "",
        "def f(naïve: str | None, /, *, b: int = 2 * 3) -> list[int]:",  # 4
        "    total: int = (b +",  # 5
        "        1)",  # 6
        "    return [total]",  # 7
        "",
        "",
        "x = lambda a, /: a",  # 10
        "y = [*range(2)]",  # 11
        "z = 4 / 2",  # 12
        "",
        "",
        "def g(",  # 15
        "    a: dict[",  # 16
        "        str, int],",  # 17
        ") -> None:",  # 18
        "    pass",  # 19
        "",
        "def h(c):",  # 21
        "    return c",  # 22
        "",
    ]
)


def test_every_annotation_is_found_with_character_columns_and_an_exclusive_end() -> None:
    facts = read_source_facts(SOURCE)
    assert sorted(facts.annotations) == [
        ((4, 13), (4, 23)),
        ((4, 34), (4, 37)),
        ((4, 50), (4, 59)),
        ((5, 11), (5, 14)),
        ((16, 7), (17, 17)),
        ((18, 5), (18, 9)),
    ]


def test_only_a_slash_or_star_before_a_comma_parenthesis_or_colon_is_a_marker() -> None:
    assert read_source_facts(SOURCE).markers == frozenset({(4, 25), (4, 28), (10, 14)})


def test_each_statement_line_maps_to_the_first_line_of_the_innermost_statement() -> None:
    assert read_source_facts(SOURCE).statement_lines == {
        1: 1,
        4: 4,
        5: 5,
        6: 5,
        7: 7,
        10: 10,
        11: 11,
        12: 12,
        15: 15,
        16: 15,
        17: 15,
        18: 15,
        19: 19,
        21: 21,
        22: 22,
    }


def test_an_annotation_holds_its_first_character_and_not_the_one_after_it() -> None:
    facts = read_source_facts(SOURCE)
    assert is_declaration(facts, (4, 13))
    assert is_declaration(facts, (4, 22))
    assert not is_declaration(facts, (4, 23))
    assert not is_declaration(facts, (4, 12))
    assert is_declaration(facts, (17, 16))
    assert not is_declaration(facts, (17, 17))


def test_a_marker_is_a_declaration_and_the_same_character_as_an_operator_is_not() -> None:
    facts = read_source_facts(SOURCE)
    assert is_declaration(facts, (4, 25))
    assert is_declaration(facts, (10, 14))
    assert not is_declaration(facts, (4, 42))
    assert not is_declaration(facts, (12, 6))
    assert not is_declaration(facts, (11, 5))


def test_a_recorded_line_stands_for_itself() -> None:
    facts = read_source_facts(SOURCE)
    assert coverage_line(facts, 6, frozenset({5, 6})) == 6


def test_an_unrecorded_continuation_line_is_read_under_its_statement() -> None:
    facts = read_source_facts(SOURCE)
    assert coverage_line(facts, 6, frozenset({5})) == 5
    assert coverage_line(facts, 17, frozenset({15})) == 15


def test_a_line_outside_every_statement_stands_for_itself() -> None:
    facts = read_source_facts(SOURCE)
    assert coverage_line(facts, 8, frozenset()) == 8
