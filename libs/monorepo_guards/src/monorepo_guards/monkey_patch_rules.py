"""Guard rule to ban monkey-patching module attributes in tests.

Tests must use dependency injection via _test_hooks modules or the
save-and-restore hook pattern, not ad-hoc module attribute mutation.

Allowed patterns:
1. Assignments to _test_hooks module aliases (action_hooks, core_hooks, etc.)
2. Assignments to sys/os (standard test patterns)
3. Assigning None (state reset for test isolation)
4. Save-and-restore fixtures: if a fixture/function saves
   ``original = module.attr`` (or ``original = getattr(module, "attr")``)
   and later restores ``module.attr = original`` (or
   ``setattr(module, "attr", original)``), intermediate
   ``module.attr = fake`` and ``setattr(module, "attr", fake)``
   assignments within that function are treated as proper hook-based DI.

Banned patterns:
    module.function = fake_function          (without save/restore in same scope)
    setattr(module, "function", fake)        (without save/restore in same scope)
    setattr(module, attr_var, fake)          (without save/restore in same scope)
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


def _attr_key(node: ast.expr) -> str | None:
    """Normalize the attribute identifier from a setattr/getattr argument.

    Args:
        node: AST expression in the attribute-name slot.

    Returns:
        ``"name:<literal>"`` for string-constant attribute names,
        ``"var:<varname>"`` for ``Name`` references, or ``None`` when
        the form is unsupported.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return f"name:{node.value}"
    if isinstance(node, ast.Name):
        return f"var:{node.id}"
    return None


class MonkeyPatchBanRule:
    """Rule to ban module attribute reassignment outside _test_hooks in tests."""

    name = "monkey-patch-ban"

    def _is_hooks_target(self, name: str) -> bool:
        """Check if the assignment target is a _test_hooks module alias.

        Args:
            name: Variable name used as the attribute target.

        Returns:
            True if the name matches or ends with a recognized hooks suffix.
        """
        for suffix in _HOOKS_SUFFIXES:
            if name.endswith(suffix):
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

    def _record_save(
        self,
        node: ast.Assign,
        saved: dict[str, tuple[str, str]],
    ) -> None:
        """Record a save assignment if it matches a supported pattern.

        Supported save forms:
            original = module.attr
            original = getattr(module, "attr")
            original = getattr(module, attr_var)

        Args:
            node: Assignment AST node with exactly one target.
            saved: Mutable mapping from local-name to (module_name, attr_key).
        """
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return
        value = node.value
        if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
            saved[target.id] = (value.value.id, f"name:{value.attr}")
            return
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "getattr"
            and len(value.args) >= 2
            and isinstance(value.args[0], ast.Name)
        ):
            key = _attr_key(value.args[1])
            if key is not None:
                saved[target.id] = (value.args[0].id, key)

    def _record_direct_restore(
        self,
        node: ast.Assign,
        saved: dict[str, tuple[str, str]],
        restored: set[tuple[str, str]],
    ) -> None:
        """Record a direct-attribute restore: ``module.attr = original``.

        Args:
            node: Assignment AST node.
            saved: Map produced by :meth:`_record_save`.
            restored: Mutable set of restored (module, attr_key) pairs.
        """
        target = node.targets[0]
        if not isinstance(target, ast.Attribute):
            return
        if not isinstance(target.value, ast.Name):
            return
        if not isinstance(node.value, ast.Name):
            return
        if node.value.id not in saved:
            return
        mod_name, attr_key = saved[node.value.id]
        if target.value.id == mod_name and attr_key == f"name:{target.attr}":
            restored.add((mod_name, attr_key))

    def _record_setattr_restore(
        self,
        call: ast.Call,
        saved: dict[str, tuple[str, str]],
        restored: set[tuple[str, str]],
    ) -> None:
        """Record a setattr restore: ``setattr(module, "attr", original)``.

        Args:
            call: setattr call AST node.
            saved: Map produced by :meth:`_record_save`.
            restored: Mutable set of restored (module, attr_key) pairs.
        """
        if not (isinstance(call.func, ast.Name) and call.func.id == "setattr"):
            return
        if len(call.args) < 3:
            return
        target = call.args[0]
        if not isinstance(target, ast.Name):
            return
        value = call.args[2]
        if not isinstance(value, ast.Name) or value.id not in saved:
            return
        key = _attr_key(call.args[1])
        if key is None:
            return
        mod_name, saved_key = saved[value.id]
        if target.id == mod_name and key == saved_key:
            restored.add((mod_name, key))

    def _find_restored_attrs_in_scope(
        self,
        body: list[ast.stmt],
    ) -> set[tuple[str, str]]:
        """Find (module, attr_key) pairs that are saved and restored in a scope.

        Detects both direct-attribute and setattr forms in either save or
        restore. Both save and restore must appear in the same function body
        (including fixture teardown after yield).

        Args:
            body: List of AST statements in a function body.

        Returns:
            Set of (module_name, attr_key) pairs that are saved/restored.
        """
        saved: dict[str, tuple[str, str]] = {}
        restored: set[tuple[str, str]] = set()

        scope = ast.Module(body=body, type_ignores=[])

        for node in ast.walk(scope):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                self._record_save(node, saved)
                self._record_direct_restore(node, saved, restored)
            elif isinstance(node, ast.Call):
                self._record_setattr_restore(node, saved, restored)

        return restored

    def _collect_restored_attrs(self, tree: ast.AST) -> set[tuple[str, str]]:
        """Collect all (module, attr_key) pairs with save-restore in any function.

        Args:
            tree: Parsed AST of the file.

        Returns:
            Set of (module_name, attr_key) pairs that are properly restored.
        """
        restored: set[tuple[str, str]] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                restored.update(self._find_restored_attrs_in_scope(node.body))
        return restored

    def _is_module_assignment_target(
        self,
        obj_name: str,
        module_aliases: set[str],
    ) -> bool:
        """Return True if the target name refers to an imported module."""
        if obj_name == "self":
            return False
        if self._is_hooks_target(obj_name):
            return False
        if obj_name in _ALLOWED_TARGETS:
            return False
        return obj_name in module_aliases

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
            restored_attrs: (module, attr_key) pairs with save-restore fixtures.
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
            if not self._is_module_assignment_target(obj_name, module_aliases):
                continue
            if isinstance(node.value, ast.Constant) and node.value.value is None:
                continue
            if (obj_name, f"name:{target.attr}") in restored_attrs:
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

    def _check_setattr_call(
        self,
        path: Path,
        node: ast.Call,
        module_aliases: set[str],
        restored_attrs: set[tuple[str, str]],
        lines: list[str],
    ) -> list[Violation]:
        """Check a setattr call for module attribute monkey-patching.

        Args:
            path: Source file path.
            node: Call AST node.
            module_aliases: Names known to be module imports.
            restored_attrs: (module, attr_key) pairs with save-restore fixtures.
            lines: Source lines for violation context.

        Returns:
            List of violations found.
        """
        if not (isinstance(node.func, ast.Name) and node.func.id == "setattr"):
            return []
        if len(node.args) < 3:
            return []
        target = node.args[0]
        if not isinstance(target, ast.Name):
            return []
        obj_name = target.id
        if not self._is_module_assignment_target(obj_name, module_aliases):
            return []
        value = node.args[2]
        if isinstance(value, ast.Constant) and value.value is None:
            return []
        key = _attr_key(node.args[1])
        if key is None:
            return []
        if (obj_name, key) in restored_attrs:
            return []
        line_text = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
        attr_display = key.split(":", 1)[1]
        return [
            Violation(
                file=path,
                line_no=node.lineno,
                kind="monkey-patch",
                line=f"setattr({obj_name}, {attr_display!r}, ...) ({line_text})",
            )
        ]

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
                elif isinstance(node, ast.Call):
                    out.extend(
                        self._check_setattr_call(
                            path, node, module_aliases, restored_attrs, lines,
                        )
                    )
        return out


__all__ = ["MonkeyPatchBanRule"]
