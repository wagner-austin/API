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


#: MCPs' decorated module (board task b9bf5690): typing names bound three
#: ways, decorators that only the type checker reads beside ones a test sees,
#: and a decorator continued over three lines.
DECORATED = "\n".join(
    [
        "import functools",  # 1
        "import typing as t",  # 2
        "from typing import Any, overload as ov",  # 3
        "from typing_extensions import final",  # 4
        "from os import path",  # 5
        "",
        "",
        "@final",  # 8
        "class Box:",  # 9
        "    @property",  # 10
        "    def v(self) -> int:",  # 11
        "        return 1",  # 12
        "",
        "    @ov",  # 14
        "    def get(self, x: int) -> int: ...",  # 15
        "    @t.overload",  # 16
        "    def get(self, x: str) -> str: ...",  # 17
        "    @classmethod",  # 18
        "    @functools.lru_cache(",  # 19
        "        maxsize=2,",  # 20
        "    )",  # 21
        "    def get(cls, x: Any) -> Any:",  # 22
        "        return x",  # 23
        "",
        "",
        "@path.final",  # 26
        "@t.cast",  # 27
        "async def run(",  # 28
        "    x: int,",  # 29
        ") -> int:",  # 30
        "    return x",  # 31
    ]
)


def test_each_decorator_line_maps_to_the_lines_of_the_definition_it_decorates() -> None:
    # A class decorator's definition runs to the class's last line; a method's
    # decorators, a decorator continued over three lines among them, map to
    # the method's first decorator through its last line.
    assert read_source_facts(DECORATED).definitions == {
        8: (8, 23),
        10: (10, 12),
        14: (14, 15),
        16: (16, 17),
        18: (18, 23),
        19: (18, 23),
        20: (18, 23),
        21: (18, 23),
        26: (26, 31),
        27: (26, 31),
    }


def test_only_overload_and_final_imported_from_typing_are_decorators_for_the_type_checker() -> None:
    # final from typing_extensions, overload imported as ov and through the
    # module alias t; not property, classmethod, a cache, os.path's attribute
    # of the same name or another typing function.
    facts = read_source_facts(DECORATED)
    assert is_declaration(facts, (8, 0))
    assert is_declaration(facts, (14, 4))
    assert is_declaration(facts, (16, 4))
    assert not is_declaration(facts, (10, 4))
    assert not is_declaration(facts, (18, 4))
    assert not is_declaration(facts, (26, 0))
    assert not is_declaration(facts, (27, 0))
    assert facts.typing_decorator_lines == frozenset({8, 14, 16})


def test_a_module_importing_nothing_from_typing_has_no_typing_decorator() -> None:
    source = "def final(f):\n    return f\n\n\n@final\ndef g() -> None:\n    pass\n"
    facts = read_source_facts(source)
    assert facts.typing_decorator_lines == frozenset()
    assert facts.definitions == {5: (5, 7)}
