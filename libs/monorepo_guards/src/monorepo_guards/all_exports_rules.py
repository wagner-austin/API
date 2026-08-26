"""Every name promised by ``__all__`` must actually resolve.

``__all__`` is a promise: these names exist and are the public surface. Nothing
in the toolchain checks the promise for the one file whose entire job is to
make it. ``mypy --strict`` does not inspect ``__all__`` contents at all, and
ruff's F822 is suppressed inside ``__init__.py`` by default -- so a stale entry
there passes lint, passes types, and fails only at runtime on ``import *``.

That is not hypothetical. A stale name was reintroduced into a package's
``__all__`` and a full ``make check`` went green on it -- 100% statements and
branches, lint clean, mypy clean -- while ``from platform_ml import *`` would
have raised. A fleet sweep then found ``procart_api`` shipping
``__all__ = ["create_app"]`` with no such name anywhere in the module.

WHY THIS IS A GUARD AND NOT A RUFF SETTING. Ruff can be made to emit F822 in
``__init__.py`` by turning preview on. Measured across all forty-one packages,
that reports twenty-two names, of which twenty-one are false: a SUBMODULE named
in ``__all__`` resolves fine, because ``_handle_fromlist`` imports it on demand,
so ``from platform_discord.qr import *`` binds ``embeds`` and ``types`` even
though ``__init__.py`` imports neither. Shipping that setting would have meant
either twenty-one spurious edits or five exempted packages. The predicate below
is the one that is actually true -- a name resolves if it is bound at module
level, OR the file is a package ``__init__`` with a submodule of that name --
and it reports exactly the one real defect with no exemption list.
"""

from __future__ import annotations

import ast
from pathlib import Path

from monorepo_guards import Violation
from monorepo_guards.util import parse_source

_PACKAGE_INIT = "__init__.py"


def _binds_from_statement(node: ast.stmt) -> list[str]:
    """Collect the module-level names a single statement binds.

    Args:
        node: The statement to inspect.

    Returns:
        Every name the statement binds in the enclosing namespace. A
        ``def``/``class`` contributes only its own name -- bindings inside its
        body belong to a different namespace and must not count toward
        ``__all__``.
    """
    if isinstance(node, ast.Import):
        # ``import a.b`` binds ``a``; ``import a.b as c`` binds ``c``.
        return [alias.asname if alias.asname else alias.name.split(".")[0] for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [alias.asname if alias.asname else alias.name for alias in node.names]
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def _module_level_bindings(body: list[ast.stmt]) -> set[str]:
    """Collect every name bound in a module's own namespace.

    Descends into block statements, whose bodies share the module namespace,
    but never into a function or class body, which does not.

    Args:
        body: The statement list to scan.

    Returns:
        The set of names bound at module level.
    """
    bound: set[str] = set()
    for node in body:
        bound.update(_binds_from_statement(node))
        if isinstance(node, ast.If | ast.For | ast.While | ast.With):
            bound.update(_module_level_bindings(node.body))
        if isinstance(node, ast.If | ast.For | ast.While):
            # ``With`` has no ``orelse``; the other three do.
            bound.update(_module_level_bindings(node.orelse))
    return bound


class AllExportsRule:
    """Reject a name in ``__all__`` that nothing makes importable."""

    name = "all-exports"

    def _resolves(self, path: Path, exported: str, bound: set[str]) -> bool:
        """Decide whether one exported name can actually be imported.

        Args:
            path: The file declaring the name.
            exported: The name as written in ``__all__``.
            bound: Names bound at module level in that file.

        Returns:
            True when the name is bound at module level, or when the file is a
            package ``__init__`` and a submodule of that name sits beside it --
            which ``import *`` imports on demand, so the promise holds.
        """
        if exported in bound:
            return True
        if path.name != _PACKAGE_INIT:
            return False
        package_dir = path.parent
        return (package_dir / f"{exported}.py").is_file() or (
            package_dir / exported / _PACKAGE_INIT
        ).is_file()

    def _check_file(self, path: Path, tree: ast.Module) -> list[Violation]:
        """Check one parsed module's ``__all__``.

        Args:
            path: The file being checked, used for submodule resolution.
            tree: The parsed module.

        Returns:
            One violation per unresolvable name, plus one when ``__all__`` is
            built by an expression rather than written as a literal -- such a
            value cannot be read statically, so the promise it makes cannot be
            checked by anything and must not be made that way.
        """
        violations: list[Violation] = []
        bound = _module_level_bindings(tree.body)

        for node in tree.body:
            if isinstance(node, ast.AugAssign) and _is_all_target(node.target):
                violations.append(_not_literal(path, node.lineno))
                continue
            if not isinstance(node, ast.Assign) or not any(_is_all_target(t) for t in node.targets):
                continue
            names = _literal_names(node.value)
            if names is None:
                violations.append(_not_literal(path, node.lineno))
                continue
            violations.extend(
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind="all-exports-undefined",
                    line=exported,
                )
                for exported in names
                if not self._resolves(path, exported, bound)
            )
        return violations

    def run(self, files: list[Path]) -> list[Violation]:
        """Check every given file's ``__all__``.

        Args:
            files: Python files to check.

        Returns:
            Every violation found, across all files.

        Raises:
            RuntimeError: When a file cannot be parsed.
        """
        out: list[Violation] = []
        for path in files:
            try:
                tree = parse_source(path)
            except SyntaxError as exc:
                raise RuntimeError(f"failed to parse {path}: {exc}") from exc
            out.extend(self._check_file(path, tree))
        return out


def _not_literal(path: Path, line_no: int) -> Violation:
    """Build the violation for an ``__all__`` that cannot be read statically.

    Args:
        path: The file declaring it.
        line_no: The line the assignment starts on.

    Returns:
        The violation to report.
    """
    return Violation(file=path, line_no=line_no, kind="all-exports-not-literal", line="")


def _is_all_target(node: ast.expr) -> bool:
    """Report whether an assignment target is the name ``__all__``.

    Args:
        node: The target expression.

    Returns:
        True when the target is exactly the name ``__all__``.
    """
    return isinstance(node, ast.Name) and node.id == "__all__"


def _literal_names(node: ast.expr) -> list[str] | None:
    """Read an ``__all__`` value as a list of string literals.

    Args:
        node: The assigned expression.

    Returns:
        The exported names, or None when the value is not a list or tuple of
        plain string literals and therefore cannot be read statically.
    """
    if not isinstance(node, ast.List | ast.Tuple):
        return None
    names: list[str] = []
    for element in node.elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
            return None
        names.append(element.value)
    return names


__all__ = ["AllExportsRule"]
