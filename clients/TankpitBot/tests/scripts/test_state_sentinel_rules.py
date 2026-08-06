"""Tests for the inline (0, 0) position-sentinel guard rule."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.state_sentinel_rules import run_state_sentinel_rules


def _write_module(project_root: Path, relative: str, body: str) -> Path:
    """Create a module inside a fake project's package tree.

    Args:
        project_root: Fake project root.
        relative: Path under ``src/tankpit_bot`` (e.g. ``bot/ai/x.py``).
        body: Module source text.

    Returns:
        Path to the created module.
    """
    module_path = project_root / "src" / "tankpit_bot" / relative
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(body, encoding="utf-8")
    return module_path


def test_missing_package_yields_zero_violations(tmp_path: Path) -> None:
    """A tree with no src/tankpit_bot passes."""
    assert run_state_sentinel_rules(tmp_path) == 0


def test_inline_sentinel_conjunction_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The classic hand-copied sentinel is a violation."""
    module_path = _write_module(
        tmp_path,
        "bot/ai/scan.py",
        'def f(tank):\n    return tank["x"] == 0 and tank["y"] == 0\n',
    )
    assert run_state_sentinel_rules(tmp_path) == 1
    out = capsys.readouterr().out
    assert f"state_sentinel_violation {module_path}:2" in out
    assert "has_known_position" in out


def test_negated_and_reversed_spellings_are_the_same_sentinel(tmp_path: Path) -> None:
    """``!=`` disguises and ``0 == x`` operand order still match."""
    _write_module(
        tmp_path,
        "bot/ai/scan.py",
        'def f(tank):\n    return tank["x"] != 0 or 0 != tank["y"]\n',
    )
    assert run_state_sentinel_rules(tmp_path) == 1


def test_canonical_module_is_exempt(tmp_path: Path) -> None:
    """state/types/tank.py is the single legal home of the comparison."""
    _write_module(
        tmp_path,
        "state/types/tank.py",
        "def has_known_position(tank):\n"
        '    if tank["x"] != 0 or tank["y"] != 0:\n'
        "        return True\n"
        '    return tank["last_position_update_ms"] > 0\n',
    )
    assert run_state_sentinel_rules(tmp_path) == 0


def test_single_axis_zero_compare_stays_legal(tmp_path: Path) -> None:
    """Comparing one axis to zero (bounds math) is not the sentinel."""
    _write_module(
        tmp_path,
        "bot/ai/bounds.py",
        'def f(tank, left):\n    return tank["x"] == 0 and left == 0\n',
    )
    assert run_state_sentinel_rules(tmp_path) == 0


def test_both_axes_nonzero_literals_stay_legal(tmp_path: Path) -> None:
    """Comparing both axes against non-zero values is a coordinate
    check, not the construction-default sentinel."""
    _write_module(
        tmp_path,
        "bot/ai/tiles.py",
        'def f(tank):\n    return tank["x"] == 5 and tank["y"] == 0\n',
    )
    assert run_state_sentinel_rules(tmp_path) == 0


def test_axes_of_different_bases_stay_legal(tmp_path: Path) -> None:
    """Zero-comparing x of one object and y of another is unrelated
    bounds math, not the sentinel."""
    _write_module(
        tmp_path,
        "bot/ai/bounds2.py",
        'def f(a, b):\n    return a["x"] == 0 and b["y"] == 0\n',
    )
    assert run_state_sentinel_rules(tmp_path) == 0


def test_ordering_comparisons_stay_legal(tmp_path: Path) -> None:
    """``<`` / ``>`` against zero is bounds math, not the sentinel."""
    _write_module(
        tmp_path,
        "bot/ai/clamp.py",
        'def f(tank):\n    return tank["x"] < 0 and tank["y"] == 0\n',
    )
    assert run_state_sentinel_rules(tmp_path) == 0


def test_chained_comparison_is_not_matched(tmp_path: Path) -> None:
    """A chained compare (two ops) is range math, not the sentinel."""
    _write_module(
        tmp_path,
        "bot/ai/ranges.py",
        'def f(tank):\n    return 0 == tank["x"] == 0 and tank["y"] == 0\n',
    )
    assert run_state_sentinel_rules(tmp_path) == 0
