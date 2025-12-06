from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Protocol


class Violation(NamedTuple):
    file: Path
    line_no: int
    kind: str
    line: str


class RuleReport(NamedTuple):
    name: str
    violations: int


class Rule(Protocol):
    @property
    def name(self) -> str: ...

    def run(self, files: list[Path]) -> list[Violation]: ...


__all__ = ["Rule", "RuleReport", "Violation"]
