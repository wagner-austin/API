"""Tests for the raw mine-layer guard rule over bot/ai."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.state_sentinel_rules import run_mine_layer_rules


def _write_module(project_root: Path, relative: str, body: str) -> Path:
    """Create a module inside a fake project's bot/ai tree.

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


def test_missing_ai_tree_yields_zero_violations(tmp_path: Path) -> None:
    """A tree with no bot/ai passes."""
    assert run_mine_layer_rules(tmp_path) == 0


def test_raw_mine_read_in_decision_module_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The 2026-08-05 wrong-layer class: a decision module reads ["mines"]."""
    module_path = _write_module(
        tmp_path,
        "bot/ai/some_selector.py",
        'def f(world):\n    return world["mines"]\n',
    )
    assert run_mine_layer_rules(tmp_path) == 1
    out = capsys.readouterr().out
    assert f"mine_layer_violation {module_path}:2" in out
    assert "hostile_mines" in out


def test_hostile_filter_owner_is_exempt(tmp_path: Path) -> None:
    """equipment.py owns the team scoping and may read the raw layer."""
    _write_module(
        tmp_path,
        "bot/ai/equipment.py",
        'def hostile_mines(world):\n    return world["mines"]\n',
    )
    assert run_mine_layer_rules(tmp_path) == 0


def test_structural_worldstate_copy_in_context_is_exempt(tmp_path: Path) -> None:
    """context.py reconstructs WorldStateDict structurally."""
    _write_module(
        tmp_path,
        "bot/ai/context.py",
        'def copy(world):\n    return {"mines": world["mines"]}\n',
    )
    assert run_mine_layer_rules(tmp_path) == 0


def test_unrelated_subscripts_pass(tmp_path: Path) -> None:
    """Non-mine subscripts in decision modules are untouched."""
    _write_module(
        tmp_path,
        "bot/ai/some_selector.py",
        'def f(world):\n    return world["containers"]\n',
    )
    assert run_mine_layer_rules(tmp_path) == 0
