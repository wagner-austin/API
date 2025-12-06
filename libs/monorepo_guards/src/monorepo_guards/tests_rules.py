from __future__ import annotations

from pathlib import Path

from monorepo_guards import Rule, Violation


class PolicyTestsRule(Rule):
    name = "tests"

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for path in files:
            as_posix = path.as_posix()
            if "/tests/" not in as_posix:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="strict")
            except OSError as exc:
                raise RuntimeError(f"failed to read {as_posix}: {exc}") from exc
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
