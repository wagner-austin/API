"""Tests for :mod:`maketools.mutation.changes`, lifted from MCPs (MCPs task e52e5148).

The module is MCPs' ``mcp_shared_py.mutation.changes`` byte for byte, and
these are MCPs' own cases for it (``mcp-shared-py/tests/
test_mutation_changes.py``, board task fec5851c) with the import path made
this package's: a lifted file is covered here as it is there. The diff text
is the shape ``git diff --unified=0 --src-prefix=a/ --dst-prefix=b/``
prints; ``test_mutation_real.py`` reads the same through real git.
"""

from __future__ import annotations

from maketools.mutation.changes import BASE_REF, added_lines, diff_arguments

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


def test_each_python_file_maps_to_the_lines_its_hunks_add() -> None:
    assert added_lines(DIFF) == {
        "src/a.py": frozenset({3, 4, 10}),
        "src/d.py": frozenset({1, 2, 3}),
    }


def test_an_added_line_that_reads_like_a_file_header_stays_an_added_line() -> None:
    lines = added_lines(DIFF)
    assert "fake.py" not in lines
    assert 10 in lines["src/a.py"]


def test_a_file_that_only_loses_lines_and_a_file_that_is_not_python_add_nothing() -> None:
    lines = added_lines(DIFF)
    assert "src/c.py" not in lines
    assert "src/b.txt" not in lines
    assert "" not in lines


def test_no_diff_adds_no_lines() -> None:
    assert added_lines("") == {}


def test_the_diff_lists_added_copied_modified_and_renamed_sources_from_the_base() -> None:
    assert diff_arguments("abc123") == [
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
    ]


def test_the_base_is_main_as_the_fleet_stages_it() -> None:
    assert BASE_REF == "refs/remotes/origin/main"
