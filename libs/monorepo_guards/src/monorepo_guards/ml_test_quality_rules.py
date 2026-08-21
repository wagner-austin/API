"""Guard rule for ML project test quality.

Enforces that ML tests verify actual learning behavior, not just
execution: training must compare losses, forward passes must check
values (not only shapes), and optimizer steps must verify weights
moved. The general weak-assertion rules live in
:mod:`monorepo_guards.test_quality_rules`.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

from monorepo_guards import Violation
from monorepo_guards.util import read_lines


class _MLPatternVisitor(ast.NodeVisitor):
    """Visitor to detect ML patterns in test functions using AST."""

    # Object names that indicate HTTP client calls, not ML training
    _HTTP_CLIENT_NAMES: ClassVar[frozenset[str]] = frozenset(
        {"http", "client", "api", "api_client", "http_client", "trainer_client"}
    )

    def __init__(self) -> None:
        self.has_backward: bool = False
        self.has_step: bool = False
        self.has_train_call: bool = False
        self.has_forward_call: bool = False
        self.has_loss_compare: bool = False
        self.has_weight_check: bool = False
        self.has_value_check: bool = False
        self.has_clone: bool = False
        self.has_state_dict: bool = False
        self.has_allclose: bool = False

    _ATTR_FLAGS: ClassVar[dict[str, str]] = {
        "backward": "has_backward",
        "step": "has_step",
        "train": "has_train_call",
        "forward": "has_forward_call",
        "clone": "has_clone",
        "state_dict": "has_state_dict",
        "allclose": "has_allclose",
        "item": "has_value_check",
        "mean": "has_value_check",
        "sum": "has_value_check",
    }

    def _is_http_client_call(self, node: ast.Attribute) -> bool:
        """Check if the attribute call is on an HTTP client object."""
        if isinstance(node.value, ast.Name):
            return node.value.id.lower() in self._HTTP_CLIENT_NAMES
        return False

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            # Skip HTTP client calls for train detection
            if node.func.attr == "train" and self._is_http_client_call(node.func):
                self.generic_visit(node)
                return
            flag = self._ATTR_FLAGS.get(node.func.attr)
            if flag is not None:
                setattr(self, flag, True)
        elif isinstance(node.func, ast.Name) and node.func.id == "model":
            self.has_forward_call = True
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        """Detect loss comparisons like loss_after < loss_before."""
        for op in node.ops:
            if isinstance(op, ast.Lt | ast.LtE):
                left_has_loss = self._name_contains(node.left, ("loss", "after", "final"))
                right_has_loss = any(
                    self._name_contains(c, ("loss", "before", "initial")) for c in node.comparators
                )
                if left_has_loss and right_has_loss:
                    self.has_loss_compare = True
                left_has_weight = self._name_contains(node.left, ("weight", "param"))
                right_has_weight = any(
                    self._name_contains(c, ("weight", "param", "before")) for c in node.comparators
                )
                if left_has_weight or right_has_weight:
                    self.has_weight_check = True
        self.generic_visit(node)

    def _name_contains(self, node: ast.expr, keywords: tuple[str, ...]) -> bool:
        if isinstance(node, ast.Name):
            name_lower = node.id.lower()
            return any(kw in name_lower for kw in keywords)
        return False


class MLTestQualityRule:
    """Guard rule specifically for ML project test quality.

    Enforces that ML tests verify actual learning behavior, not just execution.
    """

    name = "ml-test-quality"

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []

        for path in files:
            if "/tests/" not in path.as_posix() and "\\tests\\" not in str(path):
                continue
            if not path.name.startswith("test_"):
                continue

            lines = read_lines(path)
            source = "\n".join(lines)

            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError as exc:
                raise RuntimeError(f"failed to parse {path}: {exc}") from exc

            out.extend(self._check_ml_patterns(path, tree))

        return out

    def _check_ml_patterns(self, path: Path, tree: ast.AST) -> list[Violation]:
        violations: list[Violation] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue

            visitor = _MLPatternVisitor()
            visitor.visit(node)

            violations.extend(self._check_training(path, node, visitor))
            violations.extend(self._check_forward_pass(path, node, visitor))
            violations.extend(self._check_optimizer(path, node, visitor))

        return violations

    def _check_training(
        self,
        path: Path,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        visitor: _MLPatternVisitor,
    ) -> list[Violation]:
        """Check for training tests without loss comparison."""
        is_training = visitor.has_backward or visitor.has_train_call
        if not is_training:
            return []
        if visitor.has_loss_compare:
            return []

        return [
            Violation(
                file=path,
                line_no=node.lineno,
                kind="ml-train-no-loss-check",
                line=f"{node.name}: trains but doesn't verify loss decreases",
            )
        ]

    def _check_forward_pass(
        self,
        path: Path,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        visitor: _MLPatternVisitor,
    ) -> list[Violation]:
        """Check for forward pass tests that only check shapes."""
        if not visitor.has_forward_call:
            return []
        has_value = visitor.has_value_check or visitor.has_allclose or visitor.has_loss_compare
        if has_value:
            return []

        return [
            Violation(
                file=path,
                line_no=node.lineno,
                kind="ml-forward-shape-only",
                line=f"{node.name}: forward pass only checks shapes",
            )
        ]

    def _check_optimizer(
        self,
        path: Path,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        visitor: _MLPatternVisitor,
    ) -> list[Violation]:
        """Check for optimizer tests that don't verify weight changes."""
        if not visitor.has_step:
            return []
        has_weight = (
            visitor.has_weight_check
            or visitor.has_clone
            or visitor.has_state_dict
            or visitor.has_allclose
        )
        if has_weight:
            return []

        return [
            Violation(
                file=path,
                line_no=node.lineno,
                kind="ml-optimizer-no-weight-check",
                line=f"{node.name}: uses optimizer but doesn't verify weights",
            )
        ]


__all__ = ["MLTestQualityRule"]
