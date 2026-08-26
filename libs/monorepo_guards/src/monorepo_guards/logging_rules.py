from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

from monorepo_guards import Violation
from monorepo_guards.util import parse_source, read_lines


def _parse_module(path: Path) -> ast.Module:
    """Parse a module once for every check in this rule to read.

    Args:
        path: The file, used in the parse-failure message.

    Returns:
        Its parse tree, shared with every other rule in the run.

    Raises:
        RuntimeError: If the module cannot be parsed. A guard that silently
            skipped an unparsable file would report it as clean.
    """
    try:
        return parse_source(path)
    except SyntaxError as exc:
        raise RuntimeError(f"failed to parse {path}: {exc}") from exc


def _by_line(violation: Violation) -> int:
    """Order violations by the line they were found on.

    A named function rather than a lambda: the lambda's parameter carries no
    annotation, and this package holds every expression to a known type.

    Args:
        violation: The violation to order.

    Returns:
        Its line number.
    """
    return violation.line_no


def _bare_print_lines(tree: ast.Module) -> set[int]:
    """Find the lines carrying a real call to the builtin ``print``.

    Read from the parse tree rather than matched against the text, because a
    regex cannot tell a call from a mention. The text form flagged ``print(``
    wherever it appeared -- including inside a string literal, which is how a
    command built here to be executed by a *remote* interpreter registered as
    printing from this process.

    Args:
        tree: The module's parse tree.

    Returns:
        Line numbers holding a call whose callee is the bare name ``print``.
        Attribute calls such as ``console.print(...)`` are not included, which
        is the same distinction the text form was reaching for.
    """
    return {
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    }


class LoggingRule:
    name = "logging"

    # Paths that may use low-level stdlib logging for multiprocessing queue handlers.
    # Queue handler/listener types are now in platform_core.logging, so this list
    # should remain empty. Services import from platform_core.logging instead.
    _ALLOWED_PATHS: ClassVar[frozenset[str]] = frozenset()

    def _should_skip_file(self, path: Path) -> bool:
        """Check if file should be skipped from logging checks.

        The two platform_core modules that IMPLEMENT the logging layer
        (core ``logging.py`` and the rich console half split out of it,
        ``rich_logging.py``) are the boundary everything else is pushed
        behind — they alone touch stdlib logging directly.
        """
        if "platform_core" in path.parts and path.name in ("logging.py", "rich_logging.py"):
            return True
        if "tests" in path.parts:
            return True
        # Allow specific files that need low-level logging for IPC
        path_str = str(path).replace("\\", "/")
        return any(path_str.endswith(allowed) for allowed in self._ALLOWED_PATHS)

    def _extract_logging_aliases(
        self: LoggingRule, path: Path, tree: ast.Module
    ) -> tuple[set[str], set[str], list[Violation]]:
        """Find stdlib logging imports (including aliases) and collect violations.

        Read from the parse tree for the same reason the print check is: a
        regex over lines cannot tell an import from a line of text that looks
        like one, so a docstring or a command string showing
        ``from logging import getLogger`` registered as importing it.

        Args:
            path: File being checked.
            tree: Its parse tree.

        Returns:
            Module aliases (``logging`` and any ``import logging as x``),
            function aliases (each name bound by ``from logging import ...``),
            and one violation per import found.
        """
        module_aliases: set[str] = set()
        func_aliases: set[str] = set()
        violations: list[Violation] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name != "logging":
                        continue
                    module_aliases.add("logging")
                    if alias.asname is not None:
                        module_aliases.add(alias.asname)
                    violations.append(
                        Violation(
                            file=path,
                            line_no=node.lineno,
                            kind="direct-logging-import",
                            line="Use 'from platform_core.logging import get_logger'",
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                # level 0 excludes `from .logging import ...`, which is a
                # local module and not the stdlib one.
                if node.module != "logging" or node.level != 0:
                    continue
                for alias in node.names:
                    func_aliases.add(alias.asname or alias.name)
                violations.append(
                    Violation(
                        file=path,
                        line_no=node.lineno,
                        kind="from-logging-import",
                        line="Use 'from platform_core.logging import get_logger'",
                    )
                )

        return module_aliases, func_aliases, violations

    def _logging_call_violations(
        self,
        path: Path,
        tree: ast.Module,
        module_aliases: set[str],
        func_aliases: set[str],
    ) -> list[Violation]:
        """Find calls into stdlib logging, through a module or a bare name.

        Args:
            path: File being checked.
            tree: Its parse tree.
            module_aliases: Names bound to the ``logging`` module.
            func_aliases: Names bound to something imported out of it.

        Returns:
            One violation per call, ordered by line.
        """
        candidates = set(module_aliases)
        candidates.add("logging")
        violations: list[Violation] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id in candidates and func.attr == "basicConfig":
                    violations.append(
                        Violation(
                            file=path,
                            line_no=node.lineno,
                            kind="logging-basicConfig",
                            line="Configure logging through platform_core.logging",
                        )
                    )
                elif func.value.id in candidates and func.attr == "getLogger":
                    violations.append(
                        Violation(
                            file=path,
                            line_no=node.lineno,
                            kind="logging-getLogger",
                            line="Use 'from platform_core.logging import get_logger'",
                        )
                    )
            elif isinstance(func, ast.Name) and func.id in func_aliases:
                violations.append(
                    Violation(
                        file=path,
                        line_no=node.lineno,
                        kind="logging-getLogger",
                        line="Use 'from platform_core.logging import get_logger'",
                    )
                )

        return sorted(violations, key=_by_line)

    def _print_violations(
        self, path: Path, lines: list[str], print_lines: set[int]
    ) -> list[Violation]:
        """Build a violation per line holding a real ``print`` call.

        Args:
            path: File being checked.
            lines: Its text, used only to quote the offending line back.
            print_lines: Line numbers from :func:`_bare_print_lines`.

        Returns:
            One violation per print, ordered by line.
        """
        return [
            Violation(file=path, line_no=idx, kind="print", line=lines[idx - 1].rstrip("\n"))
            for idx in sorted(print_lines)
        ]

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for path in files:
            if self._should_skip_file(path):
                continue

            if path.name == "logging.py":
                out.append(
                    Violation(
                        file=path,
                        line_no=1,
                        kind="local-logging-module",
                        line="Delete local logging.py; use platform_core.logging",
                    )
                )
                continue

            # Parsed once; every check below reads the tree rather than the
            # text, so none of them can mistake a mention for the thing.
            lines = read_lines(path)
            tree = _parse_module(path)

            module_aliases, func_aliases, import_violations = self._extract_logging_aliases(
                path, tree
            )
            out.extend(import_violations)
            out.extend(self._logging_call_violations(path, tree, module_aliases, func_aliases))
            out.extend(self._print_violations(path, lines, _bare_print_lines(tree)))

        return out


__all__ = ["LoggingRule"]
