from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from re import Match
from typing import NamedTuple

from monorepo_guards import Violation
from monorepo_guards.util import read_lines


class BodyScan(NamedTuple):
    """What one except body was observed to do.

    Attributes:
        has_log: The body calls a sanctioned logging or output channel.
        has_raise: The body raises.
        transfers: The body transfers control -- return, continue, break,
            raise, or a process exit -- so it decides the outcome rather than
            letting execution fall through as if nothing had happened.
        mentions_error: The body references the alias bound by ``as``.
        end: Index of the first line past the body.
    """

    has_log: bool
    has_raise: bool
    transfers: bool
    mentions_error: bool
    end: int


class ExceptionsRule:
    name = "exceptions"

    _except_header = re.compile(r"^(\s*)except(\s+([^:]+))?:\s*$")
    _broad_types = re.compile(r"\b(Exception|BaseException)\b")
    _log_call_named = re.compile(
        r"\b(logging|log|logger)\.(debug|info|warning|error|exception|critical)\("
    )
    # ``write_line`` and ``emit_error`` are the sanctioned output channels of
    # the stdlib-only clients and CLIs (RustedWarfareBot's
    # ``_test_hooks.write_line``; hpc3's ``cli/_test_hooks.emit_error``, which
    # writes a refusal to stderr at the process boundary). Calling either in an
    # except body surfaces the failure exactly as a log call does -- for a
    # command-line tool it surfaces it more directly, since the operator reads
    # stderr and not the log.
    _log_call_any = re.compile(
        r"\.(debug|info|warning|error|exception|critical|write_line|emit_error)\("
    )
    _raise_re = re.compile(r"\braise\b")
    # A handler that transfers control has decided the outcome: the caller sees
    # a value, the loop moves on, the function ends. That is handling, not
    # swallowing, and it is how typed conversion fallbacks are written --
    # ``except ValueError: return stripped``. The sibling TypeScript rule in the
    # MCPs workspace (no-silent-catch) draws the same line: a catch arm that
    # neither rethrows, returns, nor exits MUST reference the error it caught.
    _transfers_control = re.compile(
        r"^\s*(return\b|continue\b|break\b|raise\b|sys\.exit\(|os\._exit\()"
    )
    # ``except Foo as err`` -- the alias, when the header binds one.
    _alias_re = re.compile(r"\bas\s+([A-Za-z_]\w*)\s*$")

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
        alias: str = "",
    ) -> BodyScan:
        total = len(lines)
        has_log = False
        has_raise = False
        transfers = False
        mentions_error = False
        # Word-bounded: alias ``err`` must not match inside ``error_count``.
        alias_re = re.compile(rf"\b{re.escape(alias)}\b") if alias else None
        i = start
        while i < total:
            body_line = lines[i]
            if body_line.strip() == "":
                i += 1
                continue
            body_indent = len(body_line) - len(body_line.lstrip(" \t"))
            # The body ends at the first dedent, whatever that line is. This
            # used to break only on a dedented except/finally/else, so any
            # other dedented line -- a statement after the try block, the next
            # def -- let the scan run on and credit that code's raise or log
            # to the handler. A silent handler then passed. False negative, so
            # nothing ever went red to reveal it.
            if body_indent <= header_indent:
                break
            if self._raise_re.search(body_line):
                has_raise = True
            if self._log_call_named.search(body_line) or self._log_call_any.search(body_line):
                has_log = True
            if self._transfers_control.match(body_line.strip()):
                transfers = True
            if alias_re is not None and alias_re.search(body_line):
                mentions_error = True
            i += 1
        return BodyScan(
            has_log=has_log,
            has_raise=has_raise,
            transfers=transfers,
            mentions_error=mentions_error,
            end=i,
        )

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

            alias_match = self._alias_re.search(types)
            alias = alias_match.group(1) if alias_match is not None else ""
            scan = self._scan_body(lines, body_start, indent, alias)
            if broad:
                if not (scan.has_log and scan.has_raise):
                    violations.append(
                        Violation(
                            file=path,
                            line_no=idx + 1,
                            kind="broad-except-requires-log-and-raise",
                            line=raw.rstrip("\n"),
                        )
                    )
            # A typed handler that transfers control, or that names the error it
            # caught, has dealt with the condition. Only one that does neither
            # and surfaces nothing has discarded it.
            elif not (scan.has_log or scan.has_raise or scan.transfers or scan.mentions_error):
                violations.append(
                    Violation(
                        file=path,
                        line_no=idx + 1,
                        kind="except-discards-the-error",
                        line=raw.rstrip("\n"),
                    )
                )

            idx = scan.end if scan.end > idx else idx + 1
        return violations


__all__ = ["BodyScan", "ExceptionsRule"]
