from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from monorepo_guards import Violation


class LoggingRule:
    name = "logging"

    _pat_print = re.compile(r"\bprint\s*\(")
    _pat_import_logging = re.compile(r"^\s*import\s+logging(\s+as\s+(?P<alias>\w+))?\b")
    _pat_from_logging = re.compile(r"^\s*from\s+logging\s+import\s+(?P<imports>.+)$")

    # Paths that may use low-level stdlib logging for multiprocessing queue handlers.
    # These need access to logging.handlers.QueueHandler/QueueListener for IPC logging.
    _ALLOWED_PATHS: ClassVar[frozenset[str]] = frozenset(
        [
            "services/handwriting-ai/src/handwriting_ai/training/calibration/runner.py",
        ]
    )

    def _should_skip_file(self, path: Path) -> bool:
        """Check if file should be skipped from logging checks."""
        if "platform_core" in path.parts and path.name == "logging.py":
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
    ) -> list[Violation]:
        """Check violations for print, basicConfig, and getLogger (aliases included)."""
        violations: list[Violation] = []
        alias_candidates = set(module_aliases)
        alias_candidates.add("logging")

        for idx, raw in enumerate(lines, start=1):
            line = raw.rstrip("\n")
            if self._pat_print.search(line):
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

            try:
                text = path.read_text(encoding="utf-8", errors="strict")
            except OSError as exc:
                raise RuntimeError(f"failed to read {path}: {exc}") from exc

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

            lines = text.splitlines()
            module_aliases, func_aliases, import_violations = self._extract_logging_aliases(
                path, lines
            )
            out.extend(import_violations)
            out.extend(self._check_line_violations(path, lines, module_aliases, func_aliases))

        return out


__all__ = ["LoggingRule"]
