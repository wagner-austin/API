"""Rules banning nullable dependency-injection hooks.

The sanctioned hook pattern binds every hook to its real implementation at
import time, so callers invoke the hook directly and the production and test
code paths are byte-identical in shape::

    is_dir: IsDirProtocol = _real_is_dir      # in _test_hooks.py

    def find_root(start: Path) -> Path:       # in the caller
        if _test_hooks.is_dir(start / "libs"):
            ...

The banned shape declares the hook as nullable and branches on it, which puts
a conditional in the production path and leaves a second implementation that
only production reaches::

    guard_find_monorepo_root: FindMonorepoRootProto | None = None

    def find_root(start: Path) -> Path:
        if _test_hooks.guard_find_monorepo_root is not None:
            return _test_hooks.guard_find_monorepo_root(start)
        return _find_root_impl(start)

Both halves are detected: the nullable declaration in the hooks module, and
the check-then-call dispatch wherever it appears.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

from monorepo_guards import Violation


def _is_hooks_module(path: Path) -> bool:
    """Report whether a path is a dependency-injection hooks module.

    Args:
        path: File to test.

    Returns:
        True when the file name marks it as a hooks module.
    """
    stem = path.stem
    return stem.endswith("_hooks") or stem.endswith("_hooks_guard")


def _local_protocol_names(tree: ast.AST) -> frozenset[str]:
    """Collect names of Protocol classes defined in a module.

    Hook protocols are not always named with a ``Proto`` suffix -- a container
    may declare ``create_peft_model: PeftModelCreator | None`` -- so the
    suffix alone misses them. Reading the module's own Protocol classes
    catches those without widening the rule to every optional attribute.

    Args:
        tree: Parsed module.

    Returns:
        Names of classes declared with a Protocol base.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            if (isinstance(base, ast.Name) and base.id == "Protocol") or (
                isinstance(base, ast.Attribute) and base.attr == "Protocol"
            ):
                names.add(node.name)
    return frozenset(names)


def _annotation_names_a_hook(node: ast.expr, local_protocols: frozenset[str]) -> bool:
    """Report whether an annotation names a callable hook type.

    Protocol types and ``Callable`` aliases are hooks; ordinary data types
    such as ``Path`` are module state and may legitimately be optional.

    Args:
        node: Annotation expression, already stripped of the ``| None``.
        local_protocols: Protocol classes declared in the same module.

    Returns:
        True when the annotation names a hook type.
    """
    if isinstance(node, ast.Name):
        return (
            node.id.endswith("Proto") or node.id.endswith("Protocol") or node.id in local_protocols
        )
    if isinstance(node, ast.Attribute):
        return (
            node.attr.endswith("Proto")
            or node.attr.endswith("Protocol")
            or node.attr in local_protocols
        )
    if isinstance(node, ast.Subscript):
        base = node.value
        if isinstance(base, ast.Name):
            return base.id == "Callable"
        if isinstance(base, ast.Attribute):
            return base.attr == "Callable"
    return False


def _optional_hook_annotation(node: ast.expr, local_protocols: frozenset[str]) -> bool:
    """Report whether an annotation is ``<HookType> | None``.

    Args:
        node: Annotation expression from an annotated assignment.
        local_protocols: Protocol classes declared in the same module.

    Returns:
        True when the annotation unions a hook type with None.
    """
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.BitOr):
        return False
    left, right = node.left, node.right
    left_is_none = isinstance(left, ast.Constant) and left.value is None
    right_is_none = isinstance(right, ast.Constant) and right.value is None
    if right_is_none:
        return _annotation_names_a_hook(left, local_protocols)
    if left_is_none:
        return _annotation_names_a_hook(right, local_protocols)
    return False


def _attribute_path(node: ast.expr) -> str | None:
    """Render a dotted attribute chain as text.

    Args:
        node: Expression that may be an attribute chain.

    Returns:
        The dotted path, or None when the expression is not one.
    """
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _called_function(stmt: ast.stmt) -> ast.expr | None:
    """Return the function a statement calls, if it is a call at all.

    Args:
        stmt: First statement of a conditional's body.

    Returns:
        The called expression for a bare call or a returned call, else None.
    """
    if isinstance(stmt, (ast.Return, ast.Expr)):
        value = stmt.value
        if isinstance(value, ast.Call):
            return value.func
    return None


class NullableHookRule:
    """Ban hooks declared as ``<HookType> | None = None`` in hooks modules."""

    name = "nullable-hook"

    def _scan(self, path: Path, tree: ast.AST) -> list[Violation]:
        """Collect nullable hook declarations in one module.

        Args:
            path: File being scanned.
            tree: Parsed module.

        Returns:
            One violation per nullable hook declaration.
        """
        out: list[Violation] = []
        local_protocols = _local_protocol_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.AnnAssign):
                continue
            value = node.value
            if not isinstance(value, ast.Constant) or value.value is not None:
                continue
            if not _optional_hook_annotation(node.annotation, local_protocols):
                continue
            target = node.target
            hook_name = target.id if isinstance(target, ast.Name) else "<hook>"
            out.append(
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind="nullable-hook-declaration",
                    line=(
                        f"{hook_name} must be bound to its real implementation, "
                        "not declared as '| None = None'"
                    ),
                )
            )
        return out

    def run(self, files: list[Path]) -> list[Violation]:
        """Run the rule over the given files.

        Args:
            files: Python files to scan.

        Returns:
            Every nullable hook declaration found.
        """
        out: list[Violation] = []
        for path in files:
            if not _is_hooks_module(path):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8", errors="strict"), filename=str(path))
            out.extend(self._scan(path, tree))
        return out


class HookDispatchRule:
    """Ban ``if hook is not None: return hook(...)`` conditional dispatch."""

    name = "hook-dispatch"

    _HOOK_MODULE_MARKERS: ClassVar[tuple[str, ...]] = ("hook", "hooks", "testing")

    def _references_hooks(self, dotted: str) -> bool:
        """Report whether a dotted path reaches through a hooks module.

        Args:
            dotted: Dotted attribute path, such as ``_test_hooks.is_dir``.

        Returns:
            True when any segment but the last names a hooks module.
        """
        segments = dotted.split(".")[:-1]
        return any(
            any(marker in segment.lower() for marker in self._HOOK_MODULE_MARKERS)
            for segment in segments
        )

    def _dispatch_target(self, node: ast.If) -> str | None:
        """Return the hook a check-then-call branch dispatches on.

        Args:
            node: Conditional to inspect.

        Returns:
            The hook's dotted path, or None when this is not the pattern.
        """
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            return None
        if not isinstance(test.ops[0], ast.IsNot):
            return None
        comparator = test.comparators[0]
        if not isinstance(comparator, ast.Constant) or comparator.value is not None:
            return None
        checked = _attribute_path(test.left)
        if checked is None or not self._references_hooks(checked):
            return None

        called = _called_function(node.body[0])
        if called is None:
            return None
        if _attribute_path(called) != checked:
            return None
        return checked

    def run(self, files: list[Path]) -> list[Violation]:
        """Run the rule over the given files.

        Args:
            files: Python files to scan.

        Returns:
            Every check-then-call hook dispatch found.
        """
        out: list[Violation] = []
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="strict"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.If):
                    continue
                hook = self._dispatch_target(node)
                if hook is None:
                    continue
                out.append(
                    Violation(
                        file=path,
                        line_no=node.lineno,
                        kind="hook-conditional-dispatch",
                        line=(
                            f"call {hook} directly; production binds it to the real "
                            "implementation, so no 'is not None' branch is needed"
                        ),
                    )
                )
        return out


__all__ = ["HookDispatchRule", "NullableHookRule"]
