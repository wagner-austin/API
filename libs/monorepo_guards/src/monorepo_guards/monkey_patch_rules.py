"""Guard rule to ban monkey-patching module attributes in tests.

Tests must use dependency injection via _test_hooks modules or the
save-and-restore hook pattern, not ad-hoc module attribute mutation.

Allowed patterns:
1. Assignments to _test_hooks module aliases (action_hooks, core_hooks, etc.)
2. Assignments to sys/os (standard test patterns)
3. Assigning None (state reset for test isolation)
4. Save-and-restore fixtures: if a fixture/function saves
   ``original = module.attr`` and later restores ``module.attr = original``,
   intermediate ``module.attr = fake`` assignments within that function
   are treated as proper hook-based DI.

Banned patterns:
    module.function = fake_function  (without save/restore in same scope)
"""

from __future__ import annotations

import ast
from pathlib import Path

from monorepo_guards import Violation


_HOOKS_SUFFIXES = ("_test_hooks", "_hooks", "_hooks_guard")

_ALLOWED_TARGETS = frozenset({
    "sys",
    "os",
})


class MonkeyPatchBanRule:
    """Rule to ban module attribute reassignment outside _test_hooks in tests."""

    name = "monkey-patch-ban"

    def _is_hooks_target(self, name: str) -> bool:
        """Check if the assignment target is a _test_hooks module alias.

        Args:
            name: Variable name used as the attribute target.

        Returns:
            True if the name ends with a recognized hooks suffix.
        """
        for suffix in _HOOKS_SUFFIXES:
            if name == suffix or name.endswith("_" + suffix):
                return True
        return name in ("action_hooks", "core_hooks", "script_hooks", "bot_hooks")

    def _collect_module_aliases(self, tree: ast.AST) -> set[str]:
        """Collect names bound by 'import ... as ...' statements.

        Args:
            tree: Parsed AST.

        Returns:
            Set of alias names that refer to imported modules.
        """
        aliases: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bound = alias.asname if alias.asname else alias.name
                    aliases.add(bound)
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    for alias in node.names:
                        bound = alias.asname if alias.asname else alias.name
                        aliases.add(bound)
        return aliases

    def _find_restored_attrs_in_scope(
        self,
        body: list[ast.stmt],
    ) -> set[tuple[str, str]]:
        """Find (module, attr) pairs that are saved and restored in a scope.

        Detects the pattern:
            original = module.attr   (save)
            ...
            module.attr = original   (restore)

        Both the save and restore must appear in the same function body
        (including fixture teardown after yield).

        Args:
            body: List of AST statements in a function body.

        Returns:
            Set of (module_name, attr_name) pairs that are saved/restored.
        """
        saved: dict[str, tuple[str, str]] = {}
        restored: set[tuple[str, str]] = set()

        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Attribute):
                    if isinstance(node.value.value, ast.Name):
                        saved[target.id] = (node.value.value.id, node.value.attr)

                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    if isinstance(node.value, ast.Name) and node.value.id in saved:
                        mod_name, attr_name = saved[node.value.id]
                        if target.value.id == mod_name and target.attr == attr_name:
                            restored.add((mod_name, attr_name))

        return restored

    def _collect_restored_attrs(self, tree: ast.AST) -> set[tuple[str, str]]:
        """Collect all (module, attr) pairs with save-restore in any function.

        Args:
            tree: Parsed AST of the file.

        Returns:
            Set of (module_name, attr_name) pairs that are properly restored.
        """
        restored: set[tuple[str, str]] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                restored.update(self._find_restored_attrs_in_scope(node.body))
        return restored

    def _check_assignment(
        self,
        path: Path,
        node: ast.Assign,
        module_aliases: set[str],
        restored_attrs: set[tuple[str, str]],
        lines: list[str],
    ) -> list[Violation]:
        """Check a single assignment for module attribute monkey-patching.

        Args:
            path: Source file path.
            node: Assignment AST node.
            module_aliases: Names known to be module imports.
            restored_attrs: (module, attr) pairs with save-restore fixtures.
            lines: Source lines for violation context.

        Returns:
            List of violations found.
        """
        violations: list[Violation] = []
        for target in node.targets:
            if not isinstance(target, ast.Attribute):
                continue
            if not isinstance(target.value, ast.Name):
                continue
            obj_name = target.value.id
            if obj_name == "self":
                continue
            if self._is_hooks_target(obj_name):
                continue
            if obj_name in _ALLOWED_TARGETS:
                continue
            if obj_name not in module_aliases:
                continue
            if isinstance(node.value, ast.Constant) and node.value.value is None:
                continue
            if (obj_name, target.attr) in restored_attrs:
                continue
            line_text = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
            violations.append(
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind="monkey-patch",
                    line=f"{obj_name}.{target.attr} = ... ({line_text})",
                )
            )
        return violations

    def _is_test_file(self, path: Path) -> bool:
        """Check if this is a test file.

        Args:
            path: File path to check.

        Returns:
            True if the file is under a tests directory.
        """
        posix = path.as_posix()
        return "/tests/" in posix or "\\tests\\" in str(path)

    def run(self, files: list[Path]) -> list[Violation]:
        """Check all test files for monkey-patch violations.

        Args:
            files: List of source files to check.

        Returns:
            List of violations found.
        """
        out: list[Violation] = []
        for path in files:
            if not self._is_test_file(path):
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError:
                continue
            lines = source.splitlines()
            module_aliases = self._collect_module_aliases(tree)
            restored_attrs = self._collect_restored_attrs(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    out.extend(
                        self._check_assignment(
                            path, node, module_aliases, restored_attrs, lines,
                        )
                    )
        return out


__all__ = ["MonkeyPatchBanRule"]
