from __future__ import annotations

from pathlib import Path

from monorepo_guards import Rule, Violation
from monorepo_guards.util import read_lines


class PolicyTestsRule(Rule):
    name = "tests"

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for path in files:
            as_posix = path.as_posix()
            if "/tests/" not in as_posix:
                continue
            lines = read_lines(path)
            text = "\n".join(lines)
            if "SimpleNamespace(" in text:
                out.append(
                    Violation(
                        file=path,
                        line_no=1,
                        kind="tests-simplenamespace",
                        line=as_posix,
                    )
                )
        return out


__all__ = ["PolicyTestsRule"]
