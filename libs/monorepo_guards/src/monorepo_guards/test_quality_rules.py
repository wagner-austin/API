"""Guard rules for detecting weak or fake tests.

These rules identify test anti-patterns that achieve code coverage
without actually verifying behavior. Coverage shows lines executed,
not correctness proven.

Violations:
- weak-assertion-is-not-none: `assert x is not None` proves existence only
- weak-assertion-isinstance: Type check doesn't verify behavior
- weak-assertion-hasattr: Attribute exists, but what's its value?
- weak-assertion-len-zero: `assert len(x) > 0` checks existence not content
- weak-assertion-in-output: String matching in captured output is fragile
- mock-without-assert-called-with: Mock verified called but not with what args
- test-no-comparison: Test has no before/after or expected/actual comparison
- ml-train-no-loss-comparison: ML training test without loss decrease check
- ml-forward-no-value-check: Forward pass test only checks shapes
- excessive-mocking: Test mocks more than 3 things, probably not integration
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

from monorepo_guards import Violation
from monorepo_guards.util import read_lines


def _is_patch_call(func: ast.expr) -> bool:
    """Check if func is a patch() call."""
    if isinstance(func, ast.Attribute) and func.attr == "patch":
        return True
    return isinstance(func, ast.Name) and func.id == "patch"


class _AssertVisitor(ast.NodeVisitor):
    """Visitor to analyze assert statements in test functions."""

    def __init__(self, path: Path, is_ml_project: bool) -> None:
        self.path = path
        self.is_ml_project = is_ml_project
        self.violations: list[Violation] = []
        self.current_function: str = ""
        self.function_has_comparison: bool = False
        self.function_mock_count: int = 0
        self.function_start_line: int = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name.startswith("test_"):
            self._analyze_test_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name.startswith("test_"):
            self._analyze_test_function(node)
        self.generic_visit(node)

    def _analyze_test_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.current_function = node.name
        self.function_has_comparison = False
        self.function_mock_count = 0
        self.function_start_line = node.lineno

        for child in ast.walk(node):
            self._check_assert(child)
            self._check_mock_usage(child)
            self._check_comparison(child)

        self._check_function_level_issues(node)

    def _check_assert(self, node: ast.AST) -> None:
        """Check for weak assertion patterns."""
        if not isinstance(node, ast.Assert):
            return

        test = node.test

        if self._is_identity_check_negated(test, "None"):
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=node.lineno,
                    kind="weak-assertion-is-not-none",
                    line=f"in {self.current_function}: assert ... is not None",
                )
            )

        if self._is_isinstance_check(test):
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=node.lineno,
                    kind="weak-assertion-isinstance",
                    line=f"in {self.current_function}: isinstance checks type",
                )
            )

        if self._is_hasattr_check(test):
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=node.lineno,
                    kind="weak-assertion-hasattr",
                    line=f"in {self.current_function}: hasattr checks existence",
                )
            )

        if self._is_len_existence_check(test):
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=node.lineno,
                    kind="weak-assertion-len-zero",
                    line=f"in {self.current_function}: len > 0 checks existence",
                )
            )

        if self._is_string_in_output(test):
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=node.lineno,
                    kind="weak-assertion-in-output",
                    line=f"in {self.current_function}: string in output is fragile",
                )
            )

    def _check_mock_usage(self, node: ast.AST) -> None:
        """Check for mock-related issues."""
        if isinstance(node, ast.Call) and _is_patch_call(node.func):
            self.function_mock_count += 1

        if isinstance(node, ast.Assert) and self._is_mock_called_check(node.test):
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=node.lineno,
                    kind="mock-without-assert-called-with",
                    line=f"in {self.current_function}: verify mock args",
                )
            )

    def _check_comparison(self, node: ast.AST) -> None:
        """Track if test has meaningful comparisons."""
        if not isinstance(node, ast.Compare):
            return

        comparison_ops = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
        for op in node.ops:
            if isinstance(op, comparison_ops) and self._is_variable_comparison(node):
                self.function_has_comparison = True

    def _check_function_level_issues(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Check issues that require analyzing the whole function."""
        if self.function_mock_count > 3:
            msg = f"{self.current_function}: {self.function_mock_count} mocks"
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=self.function_start_line,
                    kind="excessive-mocking",
                    line=msg,
                )
            )

        is_training = self.is_ml_project and self._is_training_test(node)
        if is_training and not self._has_loss_comparison(node):
            self.violations.append(
                Violation(
                    file=self.path,
                    line_no=self.function_start_line,
                    kind="ml-train-no-loss-comparison",
                    line=f"{self.current_function}: no loss decrease check",
                )
            )

    def _is_identity_check_negated(self, node: ast.expr, const_name: str) -> bool:
        """Check if node is `x is not <const>`."""
        if not isinstance(node, ast.Compare):
            return False
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.IsNot):
            return False

        comparator = node.comparators[0]
        if not isinstance(comparator, ast.Constant):
            return False

        return const_name == "None" and comparator.value is None

    def _is_isinstance_check(self, node: ast.expr) -> bool:
        """Check if node is isinstance(x, Y)."""
        if not isinstance(node, ast.Call):
            return False
        return isinstance(node.func, ast.Name) and node.func.id == "isinstance"

    def _is_hasattr_check(self, node: ast.expr) -> bool:
        """Check if node is hasattr(x, "y")."""
        if not isinstance(node, ast.Call):
            return False
        return isinstance(node.func, ast.Name) and node.func.id == "hasattr"

    def _is_len_existence_check(self, node: ast.expr) -> bool:
        """Check if node is len(x) > 0 or len(x) >= 1."""
        if not isinstance(node, ast.Compare):
            return False
        if not isinstance(node.left, ast.Call):
            return False

        func = node.left.func
        if not (isinstance(func, ast.Name) and func.id == "len"):
            return False
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return False

        op = node.ops[0]
        comp = node.comparators[0]
        if not isinstance(comp, ast.Constant):
            return False

        if isinstance(op, ast.Gt) and comp.value == 0:
            return True
        return isinstance(op, ast.GtE) and comp.value == 1

    def _is_string_in_output(self, node: ast.expr) -> bool:
        """Check if node is 'string' in x.out or x.err."""
        if not isinstance(node, ast.Compare):
            return False
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.In):
            return False

        comparator = node.comparators[0]
        if not isinstance(comparator, ast.Attribute):
            return False

        return comparator.attr in ("out", "err", "stdout", "stderr")

    def _is_mock_called_check(self, node: ast.expr) -> bool:
        """Check if node is mock.called without args check."""
        return isinstance(node, ast.Attribute) and node.attr == "called"

    def _is_variable_comparison(self, node: ast.Compare) -> bool:
        """Check if comparison involves variables (not just constants)."""
        var_types = (ast.Name, ast.Attribute, ast.Subscript)
        left_is_var = isinstance(node.left, var_types)
        right_is_var = any(isinstance(c, var_types) for c in node.comparators)
        return left_is_var and right_is_var

    def _is_training_test(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if this is a training-related test."""
        name_lower = node.name.lower()
        training_keywords = ("train", "fit", "epoch", "learn", "optimize")
        return any(kw in name_lower for kw in training_keywords)

    def _has_loss_comparison(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if test compares loss before/after using AST."""
        for child in ast.walk(node):
            if not isinstance(child, ast.Compare):
                continue
            for op in child.ops:
                if not isinstance(op, ast.Lt | ast.LtE | ast.Gt | ast.GtE):
                    continue
                left_kw = self._get_comparison_keywords(child.left)
                right_kws = [self._get_comparison_keywords(c) for c in child.comparators]
                if self._is_loss_comparison_pair(left_kw, right_kws):
                    return True
        return False

    def _get_comparison_keywords(self, node: ast.expr) -> set[str]:
        """Extract keywords from a name for comparison detection."""
        if isinstance(node, ast.Name):
            name_lower = node.id.lower()
            keywords = {"loss", "after", "final", "before", "initial"}
            return {kw for kw in keywords if kw in name_lower}
        return set()

    def _is_loss_comparison_pair(self, left: set[str], rights: list[set[str]]) -> bool:
        """Check if left/right keywords form a valid loss comparison."""
        left_is_after = bool(left & {"loss", "after", "final"})
        right_is_before = any(bool(r & {"loss", "before", "initial"}) for r in rights)
        return left_is_after and right_is_before


class WeakAssertionRule:
    """Guard rule for detecting weak or fake tests."""

    name = "test-quality"

    def __init__(self, is_ml_project: bool = False) -> None:
        self.is_ml_project = is_ml_project

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

            visitor = _AssertVisitor(path, self.is_ml_project)
            visitor.visit(tree)
            out.extend(visitor.violations)

        return out


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


__all__ = ["MLTestQualityRule", "WeakAssertionRule"]
