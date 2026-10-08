"""A module that shares a fixture across tests keeps those tests on one worker.

The suite distributes with ``--dist loadgroup`` (pyproject.toml): a test with
no ``xdist_group`` mark is scheduled on its own, so the slow tests of one
module spread over every worker instead of queueing on one, which is what
``--dist loadscope`` did (one worker ran 352 s while seven finished at about
197 s in CI job 113082396721, MCPs board task 2f90d785). A fixture with
``scope="module"`` (or class or package) is the exception: it trains once and
several tests assert on the result, and spread over eight workers it would
train up to eight times. Such a module declares
``pytestmark = pytest.mark.xdist_group("<its path under tests/>")``, and this
file refuses one that does not, read from the source so a new shared fixture
cannot quietly multiply its own cost.

The group is the module's own path so no two modules can share one by
accident: a shared group would queue both on one worker, the very tail this
distribution exists to remove.
"""

from __future__ import annotations

import ast
import pathlib
from typing import TypedDict

_TESTS = pathlib.Path(__file__).resolve().parent

#: Fixture scopes that outlive one test and so must not be split over workers.
_SHARED_SCOPES = frozenset({"class", "module", "package"})


def _is_pytest_attribute(node: ast.expr, *names: str) -> bool:
    """Say whether ``node`` spells ``pytest.<names[0]>.<names[1]>...``.

    Args:
        node: The expression to read.
        *names: The attribute chain after ``pytest``.

    Returns:
        True when the expression is exactly that chain.
    """
    for name in reversed(names):
        if not isinstance(node, ast.Attribute) or node.attr != name:
            return False
        node = node.value
    return isinstance(node, ast.Name) and node.id == "pytest"


def _shares_a_fixture(tree: ast.Module) -> bool:
    """Say whether a module declares a fixture that outlives one test.

    Both spellings the suite uses are calls to ``pytest.fixture`` carrying a
    ``scope`` keyword: the decorator ``@pytest.fixture(scope=...)`` and the
    ``pytest.fixture(scope=...)(impl)`` form some modules use for mypy.

    Args:
        tree: The parsed module.

    Returns:
        True when any ``pytest.fixture`` call names a shared scope.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_pytest_attribute(node.func, "fixture"):
            for keyword in node.keywords:
                if (
                    keyword.arg == "scope"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value in _SHARED_SCOPES
                ):
                    return True
    return False


def _declared_group(tree: ast.Module) -> str | None:
    """Return the module's ``pytestmark`` xdist group, if it declares one.

    Args:
        tree: The parsed module.

    Returns:
        The group name, or None when the module declares no group.
    """
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "pytestmark"
            and isinstance(node.value, ast.Call)
            and _is_pytest_attribute(node.value.func, "mark", "xdist_group")
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.Constant)
            and isinstance(node.value.args[0].value, str)
        ):
            return node.value.args[0].value
    return None


class _Scan(TypedDict):
    """What one pass over ``tests/`` read."""

    read: int
    sharing: dict[str, ast.Module]


def _scan() -> _Scan:
    """Find the modules under ``tests/`` that share a fixture across tests.

    Only a file whose text carries ``scope=`` is parsed: without it no
    ``pytest.fixture`` call can name a scope, and parsing all of them cost
    the guard 9 s.

    Returns:
        How many files were read, and each sharing module's tree under its
        POSIX path relative to ``tests/``.
    """
    paths = sorted(_TESTS.rglob("*.py"))
    sharing: dict[str, ast.Module] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if "scope=" in text:
            tree = ast.parse(text)
            if _shares_a_fixture(tree):
                sharing[path.relative_to(_TESTS).as_posix()] = tree
    return {"read": len(paths), "sharing": sharing}


def test_every_module_sharing_a_fixture_declares_its_own_group() -> None:
    scan = _scan()

    declared = {name: _declared_group(tree) for name, tree in scan["sharing"].items()}
    wrong = {name: group for name, group in declared.items() if group != name}

    assert wrong == {}, (
        f"{len(wrong)} of the {len(declared)} modules (of {scan['read']} read) that share a "
        "fixture across tests do not declare pytestmark = pytest.mark.xdist_group(<own path>): "
        f"{wrong}"
    )


def test_the_scan_finds_the_modules_it_exists_for() -> None:
    # A guard over zero modules would pass forever. These two train once at
    # module scope and assert many times, which is the case the rule is for.
    assert {"test_cartridge_capacity.py", "test_cartridge_composition.py"} <= set(
        _scan()["sharing"]
    )


def test_a_function_scoped_fixture_and_a_foreign_mark_are_not_mistaken() -> None:
    tree = ast.parse(
        "import pytest\n"
        "pytestmark = pytest.mark.slow('x')\n"
        "@pytest.fixture(scope='function')\n"
        "def f() -> None: ...\n"
        "g = other.fixture(scope='module')\n"
    )

    assert not _shares_a_fixture(tree)
    assert _declared_group(tree) is None


def test_both_fixture_spellings_and_the_group_mark_are_read() -> None:
    decorated = ast.parse("import pytest\n@pytest.fixture(scope='module')\ndef f() -> None: ...\n")
    called = ast.parse(
        "import pytest\n"
        "pytestmark = pytest.mark.xdist_group('a.py')\n"
        "f = pytest.fixture(scope='class')(impl)\n"
    )

    assert _shares_a_fixture(decorated)
    assert _shares_a_fixture(called)
    assert _declared_group(called) == "a.py"
