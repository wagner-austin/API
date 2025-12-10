from __future__ import annotations

from pathlib import Path

from monorepo_guards.config import GuardConfig


def iter_py_files(config: GuardConfig) -> list[Path]:
    roots: list[Path] = []
    for rel in config.directories:
        base = config.root / rel
        if base.exists():
            roots.append(base)
    out: list[Path] = []
    for root in roots:
        for path in root.rglob("*.py"):
            if any(part in config.exclude_parts for part in path.parts):
                continue
            out.append(path)
    return out


def read_lines(path: Path) -> list[str]:
    # utf-8-sig ensures an optional BOM is stripped before parsing.
    text = path.read_text(encoding="utf-8-sig", errors="strict")
    return text.splitlines()


__all__ = ["iter_py_files", "read_lines"]
