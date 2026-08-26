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
5. Reset-based hook containers: an autouse fixture calling ``X.reset()``
   leaves every attribute of ``X`` clean at the start of each test in its
   scope, isolating tests exactly as save-and-restore does. Libraries expose
   this shape from ``testing.py`` as a container class rather than a
   ``_test_hooks`` module, so the suffix allowlist alone does not see it.

Banned patterns:
    module.function = fake_function          (without save/restore in same scope)
    setattr(module, "function", fake)        (without save/restore in same scope)
    setattr(module, attr_var, fake)          (without save/restore in same scope)
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TypedDict

from monorepo_guards import Violation
from monorepo_guards.util import parse_source, read_source


class IsolationContext(TypedDict):
    """Isolation guarantees in force for one test file.

    Attributes:
        restored_attrs: (module_name, attr_key) pairs covered by a
            save-and-restore pair, whether written inline or in a conftest.
        reset_containers: Names of hook containers that an autouse fixture
            resets, which covers every attribute of that container.
    """

    restored_attrs: frozenset[tuple[str, str]]
    reset_containers: frozenset[str]


_HOOKS_SUFFIXES = ("_test_hooks", "_hooks", "_hooks_guard")

_ALLOWED_TARGETS = frozenset(
    {
        "sys",
        "os",
    }
)


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
            if isinstance(node, ast.Import) or (
                isinstance(node, ast.ImportFrom) and node.module is not None
            ):
                self._add_bound_names(node.names, aliases)
        return aliases

    def _add_bound_names(self, names: list[ast.alias], aliases: set[str]) -> None:
        """Add the names an import statement binds into the alias set.

        Args:
            names: Alias clauses from an Import or ImportFrom node.
            aliases: Mutable set of bound names to extend.
        """
        for alias in names:
            aliases.add(alias.asname if alias.asname else alias.name)

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
        isolation: IsolationContext,
        lines: list[str],
    ) -> list[Violation]:
        """Check a single assignment for module attribute monkey-patching.

        Args:
            path: Source file path.
            node: Assignment AST node.
            module_aliases: Names known to be module imports.
            isolation: Restore guarantees in force for this file.
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
            if obj_name in isolation["reset_containers"]:
                continue
            if (obj_name, f"name:{target.attr}") in isolation["restored_attrs"]:
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
        isolation: IsolationContext,
        lines: list[str],
    ) -> list[Violation]:
        """Check a setattr call for module attribute monkey-patching.

        Args:
            path: Source file path.
            node: Call AST node.
            module_aliases: Names known to be module imports.
            isolation: Restore guarantees in force for this file.
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
        if obj_name in isolation["reset_containers"]:
            return []
        key = _attr_key(node.args[1])
        if key is None:
            return []
        if (obj_name, key) in isolation["restored_attrs"]:
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

    def _is_autouse_fixture(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check whether a function is a pytest fixture with autouse=True.

        Only autouse fixtures qualify: a fixture a test must request by name
        proves nothing about the tests that do not request it.

        Args:
            node: Function definition to inspect.

        Returns:
            True if any decorator passes ``autouse=True``.
        """
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            for keyword in decorator.keywords:
                if (
                    keyword.arg == "autouse"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                ):
                    return True
        return False

    def _collect_reset_containers(self, tree: ast.AST) -> set[str]:
        """Collect names reset by an autouse fixture in this file.

        Args:
            tree: Parsed AST of a test module or conftest.

        Returns:
            Names on which ``<name>.reset()`` is called inside an autouse
            fixture, so every attribute of that name is restored per test.
        """
        containers: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not self._is_autouse_fixture(node):
                continue
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "reset"
                    and isinstance(inner.func.value, ast.Name)
                ):
                    containers.add(inner.func.value.id)
        return containers

    def _isolation_for(self, path: Path, tree: ast.AST) -> IsolationContext:
        """Build the isolation guarantees covering one test file.

        pytest applies a conftest fixture to every test module at or below its
        directory, so isolation living in conftest.py protects sibling modules
        just as effectively as isolation written inline. Analysing each file
        alone cannot see that, which reported correctly isolated tests as
        monkey-patching.

        Args:
            path: Test file whose governing conftests should be read.
            tree: Parsed AST of that test file.

        Returns:
            The restored attributes and reset containers in force for it.
        """
        restored = self._collect_restored_attrs(tree)
        containers = self._collect_reset_containers(tree)
        # Every ancestor is walked, matching pytest, which collects conftest.py
        # from the rootdir downward rather than stopping at a "tests" segment.
        for parent in path.parents:
            conftest = parent / "conftest.py"
            if conftest.is_file():
                # Read and parse without a guard: a conftest that cannot be
                # read or parsed is a real problem in the tree being checked,
                # and should surface rather than be silently skipped.
                #
                # `parse_source` also stops this being quadratic. Every test
                # file walks its ancestors, so before it was cached the one
                # conftest governing covenant-radar-api's 200 test files was
                # parsed 212 times in a single run -- which is why this was
                # the slowest rule of the thirty-one.
                conftest_tree = parse_source(conftest)
                restored.update(self._collect_restored_attrs(conftest_tree))
                containers.update(self._collect_reset_containers(conftest_tree))
        return {
            "restored_attrs": frozenset(restored),
            "reset_containers": frozenset(containers),
        }

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
            # Read and parse without a guard: a test file that cannot be read
            # or parsed is a real problem in the tree being checked. Silently
            # skipping it meant a file with a syntax error was also silently
            # exempt from this rule.
            source = read_source(path)
            tree = parse_source(path)
            lines = source.splitlines()
            module_aliases = self._collect_module_aliases(tree)
            isolation = self._isolation_for(path, tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    out.extend(
                        self._check_assignment(
                            path,
                            node,
                            module_aliases,
                            isolation,
                            lines,
                        )
                    )
                elif isinstance(node, ast.Call):
                    out.extend(
                        self._check_setattr_call(
                            path,
                            node,
                            module_aliases,
                            isolation,
                            lines,
                        )
                    )
        return out


__all__ = ["MonkeyPatchBanRule"]
