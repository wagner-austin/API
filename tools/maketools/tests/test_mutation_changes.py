"""Tests for :mod:`maketools.mutation.changes`, lifted from MCPs (MCPs tasks e52e5148, 9c456c33).

The module is MCPs' ``mcp_shared_py.mutation.changes`` byte for byte, and
these are MCPs' own cases for it (``mcp-shared-py/tests/
test_mutation_changes.py``, board tasks fec5851c and 9c456c33) with the
import path made this package's: a lifted file is covered here as it is
there. The diff text is the shape ``git diff --unified=0 --src-prefix=a/
--dst-prefix=b/`` prints; ``test_mutation_real.py`` reads the same through
real git. The ``pyproject.toml`` declarations are the two shapes MCPs'
packages have: ``src`` (and ``scripts``, as every API package declares) for
a poetry package, and MCPs claude-hooks' package root with its suites
omitted, copied from MCPs ``packages/claude-hooks/pyproject.toml``.
"""

from __future__ import annotations

import tomllib

import pytest

from maketools.mutation.changes import (
    BASE_REF,
    Sources,
    added_lines,
    declared_sources,
    diff_arguments,
)
from maketools.mutation.context import MutationErrorCode, MutationStepError

SRC = Sources(roots=("src",), omit=())

DIFF = """\
diff --git a/src/a.py b/src/a.py
index 1111111..2222222 100644
--- a/src/a.py
+++ b/src/a.py
@@ -2,0 +3,2 @@ def head():
+x = 1
+++ b/fake.py
@@ -9 +10 @@ def tail():
-y
+y = 2
diff --git a/src/b.txt b/src/b.txt
index 3333333..4444444 100644
--- a/src/b.txt
+++ b/src/b.txt
@@ -1 +1,3 @@
+t
+u
+v
diff --git a/src/c.py b/src/c.py
index 5555555..6666666 100644
--- a/src/c.py
+++ b/src/c.py
@@ -5,2 +4,0 @@
-gone
-gone
diff --git a/src/d.py b/src/d.py
new file mode 100644
index 0000000..7777777
--- /dev/null
+++ b/src/d.py
@@ -0,0 +1,3 @@
+one
+two
+three
"""

#: claude-hooks' declaration, as its pyproject.toml writes it.
HOOKS_PYPROJECT = """\
[tool.coverage.run]
source = ["."]
branch = true
omit = [
    "*.test.py",
    "tests/*",
]
"""

#: A change to claude-hooks: a hook, a hook package's module, a suite beside
#: the hook, a suite under a hook package, and a test under tests/.
HOOKS_DIFF = "".join(
    f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1,0 +2 @@\n+x\n"
    for path in [
        "board-watch.py",
        "board_watch/feed.py",
        "board-watch.test.py",
        "board_watch/feed.test.py",
        "tests/test_hook_suites.py",
    ]
)


def test_each_python_file_maps_to_the_lines_its_hunks_add() -> None:
    assert added_lines(DIFF, SRC) == {
        "src/a.py": frozenset({3, 4, 10}),
        "src/d.py": frozenset({1, 2, 3}),
    }


def test_an_added_line_that_reads_like_a_file_header_stays_an_added_line() -> None:
    lines = added_lines(DIFF, SRC)
    assert "fake.py" not in lines
    assert 10 in lines["src/a.py"]


def test_a_file_that_only_loses_lines_and_a_file_that_is_not_python_add_nothing() -> None:
    lines = added_lines(DIFF, SRC)
    assert "src/c.py" not in lines
    assert "src/b.txt" not in lines
    assert "" not in lines


def test_no_diff_adds_no_lines() -> None:
    assert added_lines("", SRC) == {}


def test_the_files_coverage_omits_are_not_sources_at_any_depth() -> None:
    sources = declared_sources(HOOKS_PYPROJECT)
    assert added_lines(HOOKS_DIFF, sources) == {
        "board-watch.py": frozenset({2}),
        "board_watch/feed.py": frozenset({2}),
    }


def test_without_omit_patterns_every_python_file_of_the_diff_is_a_source() -> None:
    assert sorted(added_lines(HOOKS_DIFF, Sources(roots=(".",), omit=()))) == [
        "board-watch.py",
        "board-watch.test.py",
        "board_watch/feed.py",
        "board_watch/feed.test.py",
        "tests/test_hook_suites.py",
    ]


def test_the_diff_lists_added_copied_modified_and_renamed_files_under_every_root() -> None:
    assert diff_arguments("abc123", Sources(roots=("src", "scripts"), omit=("x",))) == [
        "diff",
        "--unified=0",
        "--relative",
        "--diff-filter=ACMR",
        "--no-color",
        "--no-ext-diff",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        "abc123",
        "HEAD",
        "--",
        "src",
        "scripts",
    ]


def test_the_base_is_main_as_the_fleet_stages_it() -> None:
    assert BASE_REF == "refs/remotes/origin/main"


def test_a_package_declares_its_roots_and_its_omit_patterns() -> None:
    assert declared_sources(HOOKS_PYPROJECT) == Sources(roots=(".",), omit=("*.test.py", "tests/*"))
    two_roots = '[tool.coverage.run]\nsource = ["src", "scripts"]\nomit = []\n'
    assert declared_sources(two_roots) == Sources(roots=("src", "scripts"), omit=())


def test_a_declaration_with_no_omit_omits_nothing() -> None:
    assert declared_sources('[tool.coverage.run]\nsource = ["src"]\n').omit == ()


@pytest.mark.parametrize(
    ("text", "detail"),
    [
        ("", "names no source"),
        ("[tool.coverage.run]\nbranch = true\n", "names no source"),
        ("[tool.coverage.run]\nsource = []\n", "names no source"),
        ('[tool.coverage.run]\nsource = "src"\n', "source is not an array of strings"),
        ("[tool.coverage.run]\nsource = [1]\n", "source is not an array of strings"),
        ('[tool.coverage.run]\nsource = ["src"]\nomit = "x"\n', "omit is not an array of strings"),
        ("tool = 1\n", "is unreadable: [tool] is not a table"),
        ("[tool]\ncoverage = 1\n", "is unreadable: [tool.coverage] is not a table"),
        ("[tool.coverage]\nrun = 1\n", "is unreadable: [tool.coverage.run] is not a table"),
    ],
)
def test_a_package_that_declares_no_readable_source_is_refused(text: str, detail: str) -> None:
    with pytest.raises(MutationStepError) as raised:
        declared_sources(text)
    assert raised.value.code == MutationErrorCode.SOURCES_UNDECLARED
    assert str(raised.value) == (
        f"MUTATION_SOURCES_UNDECLARED: pyproject.toml [tool.coverage.run] {detail}; the step "
        "mutates the files coverage measures, so the package declares them there as its 100 "
        "percent gate reads them"
    )


def test_a_pyproject_that_is_not_toml_is_refused_by_the_toml_reader() -> None:
    with pytest.raises(tomllib.TOMLDecodeError):
        declared_sources("[tool.coverage.run\n")
