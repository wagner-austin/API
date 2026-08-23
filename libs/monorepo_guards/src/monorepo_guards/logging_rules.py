from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import ClassVar

from monorepo_guards import Violation
from monorepo_guards.util import read_lines


def _bare_print_lines(source: str, path: Path) -> set[int]:
    """Find the lines carrying a real call to the builtin ``print``.

    Read from the parse tree rather than matched against the text, because a
    regex cannot tell a call from a mention. The text form flagged ``print(``
    wherever it appeared -- including inside a string literal, which is how a
    command built here to be executed by a *remote* interpreter registered as
    printing from this process.

    Args:
        source: The module's full text.
        path: The file, used in the parse-failure message.

    Returns:
        Line numbers holding a call whose callee is the bare name ``print``.
        Attribute calls such as ``console.print(...)`` are not included, which
        is the same distinction the text form was reaching for.

    Raises:
        RuntimeError: If the module cannot be parsed. A guard that silently
            skipped an unparsable file would report it as clean.
    """
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise RuntimeError(f"failed to parse {path}: {exc}") from exc

    return {
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    }


class LoggingRule:
    name = "logging"
    _pat_import_logging = re.compile(r"^\s*import\s+logging(\s+as\s+(?P<alias>\w+))?\b")
    _pat_from_logging = re.compile(r"^\s*from\s+logging\s+import\s+(?P<imports>.+)$")

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
        self: LoggingRule, path: Path, lines: list[str]
    ) -> tuple[set[str], set[str], list[Violation]]:
        """Find stdlib logging imports (including aliases) and collect violations."""
        module_aliases: set[str] = set()
        func_aliases: set[str] = set()
        violations: list[Violation] = []

        for idx, line in enumerate(lines, start=1):
            match_import = self._pat_import_logging.match(line)
            if match_import is not None:
                alias = match_import.group("alias")
                module_aliases.add("logging")
                if alias is not None:
                    module_aliases.add(alias)
                violations.append(
                    Violation(
                        file=path,
                        line_no=idx,
                        kind="direct-logging-import",
                        line="Use 'from platform_core.logging import get_logger'",
                    )
                )
                continue

            match_from = self._pat_from_logging.match(line)
            if match_from is not None:
                imports_raw_maybe = match_from.group("imports")
                assert isinstance(imports_raw_maybe, str)
                imports_raw: str = imports_raw_maybe
                parts: list[str] = [
                    segment.strip() for segment in imports_raw.split(",") if segment.strip()
                ]
                for part in parts:
                    name: str
                    alias_name: str
                    name, _, alias_name = part.partition(" as ")
                    alias_stripped: str = alias_name.strip()
                    selected: str = alias_stripped if alias_stripped else name.strip()
                    func_aliases.add(selected)
                violations.append(
                    Violation(
                        file=path,
                        line_no=idx,
                        kind="from-logging-import",
                        line="Use 'from platform_core.logging import get_logger'",
                    )
                )

        return module_aliases, func_aliases, violations

    def _check_line_violations(
        self: LoggingRule,
        path: Path,
        lines: list[str],
        module_aliases: set[str],
        func_aliases: set[str],
        print_lines: set[int],
    ) -> list[Violation]:
        """Check violations for print, basicConfig, and getLogger (aliases included)."""
        violations: list[Violation] = []
        alias_candidates = set(module_aliases)
        alias_candidates.add("logging")

        for idx, raw in enumerate(lines, start=1):
            line = raw.rstrip("\n")
            if idx in print_lines:
                violations.append(Violation(file=path, line_no=idx, kind="print", line=line))
                continue

            for alias in alias_candidates:
                if re.search(rf"\b{alias}\.basicConfig\s*\(", line):
                    violations.append(
                        Violation(file=path, line_no=idx, kind="logging-basicConfig", line=line)
                    )
                    break
                if re.search(rf"\b{alias}\.getLogger\s*\(", line):
                    violations.append(
                        Violation(
                            file=path,
                            line_no=idx,
                            kind="logging-getLogger",
                            line="Use 'from platform_core.logging import get_logger'",
                        )
                    )
                    break

            for func in func_aliases:
                if re.search(rf"\b{func}\s*\(", line):
                    violations.append(
                        Violation(
                            file=path,
                            line_no=idx,
                            kind="logging-getLogger",
                            line="Use 'from platform_core.logging import get_logger'",
                        )
                    )
                    break

        return violations

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

            lines = read_lines(path)
            module_aliases, func_aliases, import_violations = self._extract_logging_aliases(
                path, lines
            )
            out.extend(import_violations)
            print_lines = _bare_print_lines("\n".join(lines), path)
            out.extend(
                self._check_line_violations(path, lines, module_aliases, func_aliases, print_lines)
            )

        return out


__all__ = ["LoggingRule"]
