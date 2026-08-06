"""Guard rule: no inline (0, 0) tank-position sentinel comparisons.

``state/types/tank.py`` owns :func:`has_known_position` — the ONE
place the registry's ``(0, 0)`` construction default may be compared
against. The default is not an edge case: the login choreography opens
every session with a full-roster 0x21 TankInfo dump (name + team, no
coordinates), so every tank spends its first 9-46 s at ``(0, 0)``
(measured 2026-08-04, three captures, 113 tanks). Until then, reading
``(x, y)`` without the predicate aims at, walks around, or walls off
the map corner. Seven modules used to hand-copy the comparison and an
eighth (``state/occupancy.py``, 2026-08-04) forgot it — this rule
keeps the class dead: any boolean conjunction under
``src/tankpit_bot`` (outside the canonical module) comparing both a
``["x"]`` and a ``["y"]`` subscript against the literal ``0`` is a
violation.

Runs as part of ``make lint`` via ``scripts.guard``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

CANONICAL_MODULE = Path("state") / "types" / "tank.py"

_AXIS_KEYS = frozenset({"x", "y"})


def _zero_compared_axis(node: ast.expr) -> tuple[str, str] | None:
    """Return (axis, base) when a node compares ``<base>["x"|"y"]`` to 0.

    Matches both operand orders (``tank["x"] == 0`` and
    ``0 == tank["x"]``) and both ``==`` / ``!=`` operators — the
    negated spelling is the same sentinel in disguise. The base is
    returned as an AST dump so the conjunction check can require both
    axes to belong to the SAME object (``a["x"] == 0 and b["y"] == 0``
    is unrelated bounds math, not the sentinel).

    Args:
        node: Expression to inspect.

    Returns:
        ``(axis, base_dump)`` when the node is a zero-comparison
        against that subscript key; None otherwise.
    """
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    if not isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
        return None
    operands = [node.left, node.comparators[0]]
    axis: tuple[str, str] | None = None
    saw_zero = False
    for operand in operands:
        if (
            isinstance(operand, ast.Constant)
            and not isinstance(operand.value, bool)
            and operand.value == 0
        ):
            saw_zero = True
        elif (
            isinstance(operand, ast.Subscript)
            and isinstance(operand.slice, ast.Constant)
            and operand.slice.value in _AXIS_KEYS
        ):
            axis = (str(operand.slice.value), ast.dump(operand.value))
    return axis if saw_zero and axis is not None else None


def _is_sentinel_conjunction(node: ast.BoolOp) -> bool:
    """Report whether a boolean op zero-compares both axes of one base.

    Args:
        node: ``and`` / ``or`` expression to inspect.

    Returns:
        True when the operands include a zero-comparison against a
        ``["x"]`` subscript and one against a ``["y"]`` subscript of
        the same base expression.
    """
    axes_by_base: dict[str, set[str]] = {}
    for value in node.values:
        matched = _zero_compared_axis(value)
        if matched is None:
            continue
        axis, base = matched
        axes_by_base.setdefault(base, set()).add(axis)
    return any(axes == _AXIS_KEYS for axes in axes_by_base.values())


def _module_violations(path: Path) -> list[str]:
    """Collect inline-sentinel violations in one module.

    Args:
        path: Python file to scan.

    Returns:
        Violation messages, one per inline sentinel comparison.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp) and _is_sentinel_conjunction(node):
            violations.append(
                f"{path}:{node.lineno} inline (0, 0) position sentinel - "
                "use state.types.has_known_position instead"
            )
    return violations


_MINE_LAYER_ALLOWED = frozenset({"equipment.py", "context.py"})
"""bot/ai modules that may touch the raw mine layer.

``equipment.py`` owns :func:`hostile_mines` — the ONE team-scoping
site; ``context.py`` only reconstructs ``WorldStateDict`` structurally.
Everything else in ``bot/ai`` must consume the composed decision
view's team-scoped answers (``is_passable``,
``is_landing_attainable``) or the hostile filter. Live origin: the
2026-08-05 attainability predicate took a caller-chosen mine layer
and one call site passed ALL teams — the bot refused to land in its
own minefields until the archive sweep (1,227 enemy vs 2 friendly
displacements) exposed it. This rule makes the layer choice
structural instead of reviewable.
"""


def _mine_layer_violations(path: Path) -> list[str]:
    """Collect raw ``["mines"]`` reads in one bot/ai module.

    Args:
        path: Python file to scan.

    Returns:
        Violation messages, one per raw mine-layer subscript.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "mines"
        ):
            violations.append(
                f"{path}:{node.lineno} raw mine-layer read in bot/ai - "
                "consume the composed view or equipment.hostile_mines"
            )
    return violations


def run_mine_layer_rules(project_root: Path) -> int:
    """Run the raw-mine-layer guard rule over ``bot/ai``.

    Args:
        project_root: Project root containing ``src/tankpit_bot``.

    Returns:
        Number of violations found (0 means the rule passes).
    """
    ai_root = project_root / "src" / "tankpit_bot" / "bot" / "ai"
    if not ai_root.is_dir():
        return 0
    violations: list[str] = []
    for module_path in sorted(ai_root.rglob("*.py")):
        if module_path.name in _MINE_LAYER_ALLOWED:
            continue
        violations.extend(_mine_layer_violations(module_path))
    for violation in violations:
        sys.stdout.write(f"mine_layer_violation {violation}\n")
    return len(violations)


def run_state_sentinel_rules(project_root: Path) -> int:
    """Run the inline-sentinel guard rule over a project tree.

    Args:
        project_root: Project root containing ``src/tankpit_bot``.

    Returns:
        Number of violations found (0 means the rule passes).
    """
    package_root = project_root / "src" / "tankpit_bot"
    if not package_root.is_dir():
        return 0
    canonical = package_root / CANONICAL_MODULE
    violations: list[str] = []
    for module_path in sorted(package_root.rglob("*.py")):
        if module_path == canonical:
            continue
        violations.extend(_module_violations(module_path))
    for violation in violations:
        sys.stdout.write(f"state_sentinel_violation {violation}\n")
    return len(violations)


__all__ = [
    "CANONICAL_MODULE",
    "run_mine_layer_rules",
    "run_state_sentinel_rules",
]
