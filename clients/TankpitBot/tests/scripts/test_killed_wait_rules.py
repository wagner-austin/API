"""Tests for the killed-wait guard rule.

Each catching case is paired with the nearest legitimate shape the rule
must leave alone: an unbounded wait after a kill, a bounded wait on a
process nothing killed, a bounded wait that comes BEFORE the kill, and a
kill of one receiver beside a bounded wait on another.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.killed_wait_rules import run_killed_wait_rules


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
    assert run_killed_wait_rules(tmp_path) == 0


@pytest.mark.parametrize(
    "wait_call",
    ["process.wait(10.0)", "process.wait(timeout=30)"],
)
def test_a_bounded_wait_after_a_kill_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], wait_call: str
) -> None:
    """Negative control: the reap that failed teardown under load (06fc3195)."""
    module_path = _write(
        tmp_path,
        "test_reap.py",
        f"def test_reaps(process):\n    process.kill()\n    {wait_call}\n",
    )

    assert run_killed_wait_rules(tmp_path) == 1
    assert capsys.readouterr().out == f"killed_wait_violation {module_path}:3\n"


def test_a_nested_violation_is_reported_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A wait inside an inner function is inside the outer one too, and counts once."""
    module_path = _write(
        tmp_path,
        "sub/test_nested.py",
        "def test_outer():\n"
        "    def reap(process):\n"
        "        process.kill()\n"
        "        process.wait(5)\n"
        "    reap(None)\n",
    )

    assert run_killed_wait_rules(tmp_path) == 1
    assert capsys.readouterr().out == f"killed_wait_violation {module_path}:4\n"


def test_legitimate_waits_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Every nearby shape that is not a bet on a killed process's rundown."""
    _write(
        tmp_path,
        "test_fine.py",
        "def test_unbounded(process):\n"
        "    process.kill()\n"
        "    process.wait()\n"
        "\n"
        "def test_self_exiting(process):\n"
        "    process.wait(timeout=30)\n"
        "\n"
        "def test_wait_then_kill(process):\n"
        "    process.wait(1.0)\n"
        "    process.kill()\n"
        "\n"
        "def test_other_receiver(first, second):\n"
        "    first.kill()\n"
        "    second.wait(timeout=30)\n"
        "\n"
        "async def test_event(event):\n"
        "    await event.wait()\n"
        "    kill()\n",
    )

    assert run_killed_wait_rules(tmp_path) == 0
    assert capsys.readouterr().out == ""


def test_project_tests_tree_passes_its_own_rule() -> None:
    """The real suite is clean, so the rule cannot regress silently."""
    project_root = Path(__file__).resolve().parents[2]
    assert run_killed_wait_rules(project_root) == 0
