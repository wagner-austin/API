"""Tests for the shadow-protocol-constant guard rule."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.protocol_constant_rules import (
    SHADOW_NAME_PATTERN,
    run_protocol_constant_rules,
)


def _write_module(project_root: Path, relative: str, body: str) -> Path:
    """Create a module inside a fake project's package tree.

    Args:
        project_root: Fake project root.
        relative: Path under ``src/tankpit_bot`` (e.g. ``bot/loop.py``).
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
    assert run_protocol_constant_rules(tmp_path) == 0


def test_shadow_int_constant_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A private _COMMAND_ERROR_* int re-declaration is a violation."""
    module_path = _write_module(
        tmp_path,
        "bot/loop.py",
        '_COMMAND_ERROR_TANK_FULL = 5  # "Tank full"\n',
    )
    assert run_protocol_constant_rules(tmp_path) == 1
    out = capsys.readouterr().out
    assert f"protocol_constant_violation {module_path}:1" in out
    assert "'_COMMAND_ERROR_TANK_FULL' re-declares protocol constants" in out


def test_shadow_names_dict_is_reported(tmp_path: Path) -> None:
    """A names dict keyed by integer literals is the same fork."""
    _write_module(
        tmp_path,
        "sniffer/dispatch.py",
        '_SUPERVISOR_ERROR_NAMES: dict[int, str] = {0: "cant_do", 5: "tank_full"}\n',
    )
    assert run_protocol_constant_rules(tmp_path) == 1


def test_canonical_module_is_exempt(tmp_path: Path) -> None:
    """protocol/constants.py itself is the single legal home."""
    _write_module(
        tmp_path,
        "protocol/constants.py",
        "SUPERVISOR_ERROR_TANK_FULL = 5\n"
        'SUPERVISOR_ERROR_NAMES: dict[int, str] = {SUPERVISOR_ERROR_TANK_FULL: "tank_full"}\n',
    )
    assert run_protocol_constant_rules(tmp_path) == 0


def test_named_constant_tables_stay_legal(tmp_path: Path) -> None:
    """Referencing the canonical names without literals is not a fork."""
    _write_module(
        tmp_path,
        "bot/actions.py",
        "from tankpit_bot.protocol.constants import SUPERVISOR_ERROR_TANK_FULL\n"
        "_COMMAND_ERROR_APPLICABILITY = {\n"
        '    "collect": frozenset({SUPERVISOR_ERROR_TANK_FULL}),\n'
        "}\n",
    )
    assert run_protocol_constant_rules(tmp_path) == 0


def test_non_name_assignment_targets_are_ignored(tmp_path: Path) -> None:
    """Attribute and tuple targets are out of scope — only bare names
    can shadow the canonical constants."""
    _write_module(
        tmp_path,
        "bot/config_holder.py",
        "class _Holder:\n"
        "    pass\n"
        "holder = _Holder()\n"
        "holder.SUPERVISOR_ERROR_TANK_FULL = 5\n"
        "first, second = 4, 5\n",
    )
    assert run_protocol_constant_rules(tmp_path) == 0


def test_unrelated_names_with_literals_are_ignored(tmp_path: Path) -> None:
    """Only the error-constant name patterns are in scope."""
    _write_module(
        tmp_path,
        "bot/thresholds.py",
        "_SHOT_REJECTING_SET = frozenset({0, 3, 8})\nRETRY_LIMIT = 4\n",
    )
    assert run_protocol_constant_rules(tmp_path) == 0


def test_pattern_covers_both_fork_spellings() -> None:
    """The historical fork names both match; canonical names too."""
    assert SHADOW_NAME_PATTERN.match("_COMMAND_ERROR_CANT_GO_THERE")
    assert SHADOW_NAME_PATTERN.match("_SUPERVISOR_ERROR_NAMES")
    assert SHADOW_NAME_PATTERN.match("SUPERVISOR_ERROR_TANK_FULL")
    assert not SHADOW_NAME_PATTERN.match("_SHOT_REJECTING_COMMAND_ERRORS")
