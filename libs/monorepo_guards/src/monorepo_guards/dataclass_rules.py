from __future__ import annotations

import re
from pathlib import Path

from monorepo_guards import Violation
from monorepo_guards.config import GuardConfig


class DataclassRule:
    name = "dataclass"

    def __init__(self, config: GuardConfig) -> None:
        self._config = config

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for path in files:
            if not self._is_banned_path(path):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="strict")
            except OSError as exc:
                raise RuntimeError(f"failed to read {path}: {exc}") from exc
            for idx, raw in enumerate(text.splitlines(), start=1):
                line = raw.rstrip("\n")
                if re.match(r"^\s*@dataclass\b", line):
                    out.append(
                        Violation(
                            file=path,
                            line_no=idx,
                            kind="dataclass-decorator-forbidden",
                            line=line,
                        )
                    )
                if re.search(r"from\s+dataclasses\s+import\s+dataclass\b", line):
                    out.append(
                        Violation(
                            file=path,
                            line_no=idx,
                            kind="dataclass-import-forbidden",
                            line=line,
                        )
                    )
        return out

    def _is_banned_path(self, path: Path) -> bool:
        parts = path.parts
        for segment in self._config.dataclass_ban_segments:
            if all(elem in parts for elem in segment):
                return True
        return False


__all__ = ["DataclassRule"]
