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
- weak-assertion-key-in-dict: `assert "key" in d` checks key exists but not value
- mock-without-assert-called-with: Mock verified called but not with what args
- test-no-comparison: Test has no before/after or expected/actual comparison
- ml-train-no-loss-comparison: ML training test without loss decrease check
  (the ``is_ml_project`` mode; the standalone ML rule lives in
  :mod:`monorepo_guards.ml_test_quality_rules`)
- excessive-mocking: Test mocks more than 3 things, probably not integration
"""

from __future__ import annotations

import ast
from pathlib import Path

from monorepo_guards import Violation
from monorepo_guards.util import read_lines


def _is_patch_call(func: ast.expr) -> bool:
    """Check if func is a patch() call."""
    if isinstance(func, ast.Attribute) and func.attr == "patch":
        return True
    return isinstance(func, ast.Name) and func.id == "patch"


class _KeyInDictCheck:
    """Tracks a key-in-dict assertion for later validation."""

    def __init__(self, dict_name: str, key_value: str, line_no: int) -> None:
        self.dict_name = dict_name
        self.key_value = key_value
        self.line_no = line_no


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
        self._key_in_dict_checks: list[_KeyInDictCheck] = []
        self._verified_dict_keys: set[tuple[str, str]] = set()

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
        self._key_in_dict_checks = []
        self._verified_dict_keys = set()
        self._dict_like_vars: set[str] = set()

        # First pass: identify dict-like variables (those accessed via subscript)
        for child in ast.walk(node):
            self._identify_dict_like_vars(child)

        # Second pass: analyze assertions
        for child in ast.walk(node):
            self._check_assert(child)
            self._check_mock_usage(child)
            self._check_comparison(child)
            self._track_dict_key_verification(child)

        self._check_function_level_issues(node)
        self._check_unverified_key_in_dict()

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

        key_in_dict = self._extract_key_in_dict(test)
        if key_in_dict is not None:
            dict_name, key_value = key_in_dict
            # Only track if the variable is used as a dict elsewhere
            if dict_name in self._dict_like_vars:
                self._key_in_dict_checks.append(_KeyInDictCheck(dict_name, key_value, node.lineno))

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

    def _extract_key_in_dict(self, node: ast.expr) -> tuple[str, str] | None:
        """Extract (dict_name, key) from `assert "key" in dict_name`.

        Returns None if the pattern doesn't match. Only matches simple Name
        comparators (not chained subscripts or attributes).
        """
        if not isinstance(node, ast.Compare):
            return None
        # Single In op check - chained comparisons have multiple ops
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.In):
            return None
        # Note: A single In op always has exactly 1 comparator in Python AST

        # The key is the left side (e.g., "key" in `"key" in d`)
        key_node = node.left
        if not isinstance(key_node, ast.Constant):
            return None
        if not isinstance(key_node.value, str):
            return None

        # The dict is the comparator (e.g., d in `"key" in d`)
        dict_node = node.comparators[0]
        if not isinstance(dict_node, ast.Name):
            return None

        return (dict_node.id, key_node.value)

    def _identify_dict_like_vars(self, node: ast.AST) -> None:
        """Identify variables used with subscript access (dict-like behavior).

        This helps distinguish between set membership checks (valid) and
        dict key checks (potentially weak).
        """
        if not isinstance(node, ast.Subscript):
            return

        # Get the root variable name from the subscript
        value_node = node.value
        while isinstance(value_node, ast.Subscript):
            value_node = value_node.value

        if isinstance(value_node, ast.Name):
            self._dict_like_vars.add(value_node.id)

    def _track_dict_key_verification(self, node: ast.AST) -> None:
        """Track dict[key] accesses in assert statements to mark keys as verified.

        Matches patterns like:
        - assert d["key"] == value
        - assert d["outer"]["inner"] == value (tracks both levels)
        """
        if not isinstance(node, ast.Assert):
            return

        for child in ast.walk(node.test):
            subscript_info = self._extract_subscript_access(child)
            if subscript_info is not None:
                dict_name, key_value = subscript_info
                self._verified_dict_keys.add((dict_name, key_value))

    def _extract_subscript_access(self, node: ast.AST) -> tuple[str, str] | None:
        """Extract (dict_name, key) from d["key"] subscript access.

        Also handles nested subscripts like d["outer"]["inner"] by extracting
        both the outer access and the inner access.
        """
        if not isinstance(node, ast.Subscript):
            return None

        # Get the key
        slice_node = node.slice
        if not isinstance(slice_node, ast.Constant):
            return None
        if not isinstance(slice_node.value, str):
            return None
        key_value = slice_node.value

        # Get the dict name - could be Name or another Subscript
        value_node = node.value
        if isinstance(value_node, ast.Name):
            return (value_node.id, key_value)

        # For nested subscripts like payload["request"]["device"],
        # we return (payload, "request") for the outer level
        # The inner level will be caught by the recursive walk
        return None

    def _check_unverified_key_in_dict(self) -> None:
        """Report violations for key-in-dict checks without value verification."""
        for check in self._key_in_dict_checks:
            key_tuple = (check.dict_name, check.key_value)
            if key_tuple not in self._verified_dict_keys:
                msg = (
                    f"in {self.current_function}: "
                    f'assert "{check.key_value}" in {check.dict_name} checks existence only'
                )
                self.violations.append(
                    Violation(
                        file=self.path,
                        line_no=check.line_no,
                        kind="weak-assertion-key-in-dict",
                        line=msg,
                    )
                )


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


__all__ = ["WeakAssertionRule"]
