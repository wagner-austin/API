"""Tests for the ``_test_hooks`` restoration guard rule.

The first two tests are the paired negative control: the rule must fire
on a swap that is never put back, and must stay silent on each of the
four ways the suite legitimately restores one. A first draft of this
sweep reported 24 violations of which 23 were the legitimate patterns,
so both directions are asserted here rather than only the catching one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.hook_restore_rules import run_hook_restore_rules

_RESET_CONFTEST = """
import pytest
from tankpit_bot import _test_hooks


@pytest.fixture(autouse=True)
def _restore_hooks():
    _test_hooks.write_text = _test_hooks._real_write_text
    yield
    _test_hooks.write_text = _test_hooks._real_write_text
"""


def _write(root: Path, relative: str, body: str) -> Path:
    """Create a file inside a fake project's tests tree.

    Args:
        root: Fake project root.
        relative: Path under ``tests`` (e.g. ``sub/test_x.py``).
        body: File source text.

    Returns:
        Path to the created file.
    """
    target = root / "tests" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


def test_missing_tests_tree_yields_zero_violations(tmp_path: Path) -> None:
    """A project with no tests directory passes."""
    assert run_hook_restore_rules(tmp_path) == 0


def test_unrestored_swap_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Negative control: the leak class this rule exists to kill."""
    module_path = _write(
        tmp_path,
        "test_leak.py",
        "from tankpit_bot import _test_hooks\n"
        "\n"
        "def test_leaks():\n"
        "    _test_hooks.remove_file = lambda p: None\n",
    )
    assert run_hook_restore_rules(tmp_path) == 1
    out = capsys.readouterr().out
    assert f"hook_restore_violation {module_path}:remove_file" in out


def test_attr_in_reset_fixture_is_safe(tmp_path: Path) -> None:
    """A swap the autouse reset fixture covers is not a violation."""
    _write(tmp_path, "conftest.py", _RESET_CONFTEST)
    _write(
        tmp_path,
        "test_ok.py",
        "from tankpit_bot import _test_hooks\n"
        "\n"
        "def test_swaps():\n"
        "    _test_hooks.write_text = lambda p, c: None\n",
    )
    assert run_hook_restore_rules(tmp_path) == 0


def test_finally_restore_is_safe(tmp_path: Path) -> None:
    """A swap put back in a ``finally`` body is not a violation."""
    _write(
        tmp_path,
        "test_ok.py",
        "from tankpit_bot import _test_hooks\n"
        "\n"
        "def test_swaps():\n"
        "    orig = _test_hooks.http_get\n"
        "    _test_hooks.http_get = lambda u: None\n"
        "    try:\n"
        "        pass\n"
        "    finally:\n"
        "        _test_hooks.http_get = orig\n",
    )
    assert run_hook_restore_rules(tmp_path) == 0


def test_teardown_method_restore_is_safe(tmp_path: Path) -> None:
    """A ``setup_method``/``teardown_method`` pair is not a violation."""
    _write(
        tmp_path,
        "test_ok.py",
        "from tankpit_bot import _test_hooks\n"
        "\n"
        "class TestPaired:\n"
        "    def setup_method(self):\n"
        "        self._o = _test_hooks.glob_paths\n"
        "        _test_hooks.glob_paths = lambda d, p: []\n"
        "\n"
        "    def teardown_method(self):\n"
        "        _test_hooks.glob_paths = self._o\n",
    )
    assert run_hook_restore_rules(tmp_path) == 0


def test_post_yield_restore_is_safe(tmp_path: Path) -> None:
    """A fixture that saves, yields, then puts back is not a violation."""
    _write(
        tmp_path,
        "test_ok.py",
        "import pytest\n"
        "from tankpit_bot import _test_hooks\n"
        "\n"
        "@pytest.fixture(autouse=True)\n"
        "def _iso():\n"
        "    orig = _test_hooks.http_get\n"
        "    _test_hooks.http_get = lambda u: None\n"
        "    yield\n"
        "    _test_hooks.http_get = orig\n",
    )
    assert run_hook_restore_rules(tmp_path) == 0


def test_ancestor_conftest_restore_is_safe(tmp_path: Path) -> None:
    """A fixture in a parent conftest protects tests beneath it."""
    _write(
        tmp_path,
        "sub/conftest.py",
        "import pytest\n"
        "from tankpit_bot import _test_hooks\n"
        "\n"
        "@pytest.fixture(autouse=True)\n"
        "def _iso():\n"
        "    orig = _test_hooks.setup_rich_logging\n"
        "    yield\n"
        "    _test_hooks.setup_rich_logging = orig\n",
    )
    _write(
        tmp_path,
        "sub/test_ok.py",
        "from tankpit_bot import _test_hooks\n"
        "\n"
        "def test_swaps():\n"
        "    _test_hooks.setup_rich_logging = lambda: None\n",
    )
    assert run_hook_restore_rules(tmp_path) == 0


def test_deep_attribute_chain_is_detected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``pkg._test_hooks.attr = ...`` is the same swap, spelled longer."""
    module_path = _write(
        tmp_path,
        "test_deep.py",
        "import tankpit_bot\n"
        "\n"
        "def test_swaps():\n"
        "    tankpit_bot._test_hooks.remove_file = lambda p: None\n",
    )
    assert run_hook_restore_rules(tmp_path) == 1
    out = capsys.readouterr().out
    assert out == f"hook_restore_violation {module_path}:remove_file\n"


def test_tuple_target_is_detected(tmp_path: Path) -> None:
    """Two attributes stored by one statement are both counted."""
    _write(
        tmp_path,
        "test_tuple.py",
        "from tankpit_bot import _test_hooks\n"
        "\n"
        "def test_swaps():\n"
        "    _test_hooks.remove_file, _test_hooks.http_get = None, None\n",
    )
    assert run_hook_restore_rules(tmp_path) == 2


def test_annotated_assignment_is_detected(tmp_path: Path) -> None:
    """An annotated swap is still a swap."""
    _write(
        tmp_path,
        "test_ann.py",
        "from collections.abc import Callable\n"
        "from tankpit_bot import _test_hooks\n"
        "\n"
        "def test_swaps():\n"
        "    _test_hooks.remove_file: Callable[..., None] = lambda p: None\n",
    )
    assert run_hook_restore_rules(tmp_path) == 1


def test_non_hook_attribute_assignment_is_ignored(tmp_path: Path) -> None:
    """Assigning to an unrelated module attribute is not this rule's business."""
    _write(
        tmp_path,
        "test_other.py",
        "import logging\n"
        "\n"
        "def test_assigns():\n"
        "    logging.PLACEHOLDER = 1\n"
        "    local = 2\n"
        "    assert local == 2\n",
    )
    assert run_hook_restore_rules(tmp_path) == 0


def test_conftest_without_reset_fixture_covers_nothing(tmp_path: Path) -> None:
    """A conftest lacking ``_restore_hooks`` grants no exemption."""
    _write(tmp_path, "conftest.py", "import pytest\n")
    _write(
        tmp_path,
        "test_leak.py",
        "from tankpit_bot import _test_hooks\n"
        "\n"
        "def test_leaks():\n"
        "    _test_hooks.write_text = lambda p, c: None\n",
    )
    assert run_hook_restore_rules(tmp_path) == 1


def test_project_tests_tree_passes_its_own_rule() -> None:
    """The real suite is clean, so the rule cannot regress silently."""
    project_root = Path(__file__).resolve().parents[2]
    assert run_hook_restore_rules(project_root) == 0
