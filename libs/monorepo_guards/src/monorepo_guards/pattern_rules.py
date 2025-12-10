from __future__ import annotations

import re
from pathlib import Path
from re import Pattern
from typing import ClassVar

from monorepo_guards import Violation
from monorepo_guards.config import GuardConfig
from monorepo_guards.util import read_lines


class PatternRule:
    name = "patterns"

    # Obfuscate tokens to avoid self-flagging while scanning this file
    _tok_todo = "TO" + "DO"
    _tok_fixme = "FIX" + "ME"
    _tok_hack = "HA" + "CK"
    _tok_xxx = "X" + "XX"
    _tok_wip = "WI" + "P"
    _tok_noqa = "no" + "qa"

    _patterns: ClassVar[dict[str, Pattern[str]]] = {
        _tok_todo: re.compile(r"\b" + _tok_todo + r"\b"),
        _tok_fixme: re.compile(r"\b" + _tok_fixme + r"\b"),
        _tok_hack: re.compile(r"\b" + _tok_hack + r"\b"),
        _tok_xxx: re.compile(r"\b" + _tok_xxx + r"\b"),
        _tok_wip: re.compile(r"\b" + _tok_wip + r"\b"),
        _tok_noqa: re.compile(r"#\s*" + _tok_noqa + r"\b"),
    }

    def __init__(self, config: GuardConfig) -> None:
        self._config = config

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for path in files:
            if self._config.forbid_pyi and path.suffix == ".pyi":
                out.append(
                    Violation(
                        file=path,
                        line_no=1,
                        kind="pyi-disallowed",
                        line=str(path),
                    )
                )
                continue
            lines = read_lines(path)
            for idx, line in enumerate(lines, start=1):
                for name, pattern in self._patterns.items():
                    if pattern.search(line):
                        out.append(
                            Violation(
                                file=path,
                                line_no=idx,
                                kind=f"pattern-{name}",
                                line=line,
                            )
                        )
        return out


__all__ = ["PatternRule"]
