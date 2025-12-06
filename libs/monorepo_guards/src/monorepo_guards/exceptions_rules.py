from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from re import Match

from monorepo_guards import Violation
from monorepo_guards.util import read_lines


class ExceptionsRule:
    name = "exceptions"

    _except_header = re.compile(r"^(\s*)except(\s+([^:]+))?:\s*$")
    _broad_types = re.compile(r"\b(Exception|BaseException)\b")
    _log_call_named = re.compile(
        r"\b(logging|log|logger)\.(debug|info|warning|error|exception|critical)\("
    )
    _log_call_any = re.compile(r"\.(debug|info|warning|error|exception|critical)\(")
    _raise_re = re.compile(r"\braise\b")

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for path in files:
            lines = read_lines(path)
            if not lines:
                continue
            out.extend(self._scan_excepts(path, lines))
        return out

    def _parse_header(self, raw: str) -> tuple[int, str] | None:
        match: Match[str] | None = self._except_header.match(raw)
        if match is None:
            return None
        indent_group: str | None = match.group(1)
        group3: str | None = match.group(3)
        indent_str = indent_group if indent_group is not None else ""
        types_str = group3 if group3 is not None else ""
        return len(indent_str), types_str.strip()

    def _is_broad(self, types: str) -> bool:
        return types == "" or self._broad_types.search(types) is not None

    def _find_body_start(self, lines: Sequence[str], start: int) -> int | None:
        total = len(lines)
        i = start
        while i < total:
            if lines[i].strip() != "":
                return i
            i += 1
        return None

    def _first_body_is_trivial(self, line: str) -> bool:
        return re.match(r"^\s+(pass|\.\.\.)\s*(#.*)?$", line) is not None

    def _scan_body(
        self,
        lines: Sequence[str],
        start: int,
        header_indent: int,
    ) -> tuple[bool, bool, int]:
        total = len(lines)
        has_log = False
        has_raise = False
        i = start
        while i < total:
            body_line = lines[i]
            if body_line.strip() == "":
                i += 1
                continue
            body_indent = len(body_line) - len(body_line.lstrip(" \t"))
            if body_indent <= header_indent and re.match(
                r"^\s*(except\b|finally\b|else\b|$)", body_line
            ):
                break
            if self._raise_re.search(body_line):
                has_raise = True
            if self._log_call_named.search(body_line) or self._log_call_any.search(body_line):
                has_log = True
            i += 1
        return has_log, has_raise, i

    def _scan_excepts(self, path: Path, lines: Sequence[str]) -> list[Violation]:
        violations: list[Violation] = []
        total = len(lines)
        idx = 0
        while idx < total:
            raw = lines[idx]
            parsed = self._parse_header(raw)
            if parsed is None:
                idx += 1
                continue
            indent, types = parsed
            broad = self._is_broad(types)

            body_start = self._find_body_start(lines, idx + 1)
            if body_start is None:
                violations.append(
                    Violation(
                        file=path,
                        line_no=idx + 1,
                        kind="silent-except-body",
                        line=raw.rstrip("\n"),
                    )
                )
                idx += 1
                continue

            if self._first_body_is_trivial(lines[body_start]):
                violations.append(
                    Violation(
                        file=path,
                        line_no=idx + 1,
                        kind="silent-except-body",
                        line=raw.rstrip("\n"),
                    )
                )

            has_log, has_raise, body_end = self._scan_body(lines, body_start, indent)
            if broad:
                if not (has_log and has_raise):
                    violations.append(
                        Violation(
                            file=path,
                            line_no=idx + 1,
                            kind="broad-except-requires-log-and-raise",
                            line=raw.rstrip("\n"),
                        )
                    )
            else:
                if not (has_log or has_raise):
                    violations.append(
                        Violation(
                            file=path,
                            line_no=idx + 1,
                            kind="except-without-log-or-raise",
                            line=raw.rstrip("\n"),
                        )
                    )

            idx = body_end if body_end > idx else idx + 1
        return violations


__all__ = ["ExceptionsRule"]
