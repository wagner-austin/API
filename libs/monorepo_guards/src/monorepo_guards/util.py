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


CONFIG_FILENAME = "monorepo-guards.toml"


def find_monorepo_root(start: Path) -> Path | None:
    """Find the monorepo root by walking up for the guard config.

    The directory holding ``monorepo-guards.toml`` is the monorepo root by
    definition, since that file is what declares the guards for everything
    beneath it.

    Args:
        start: Directory to begin searching from, searched itself first.

    Returns:
        The monorepo root, or None when no ancestor holds the config, which
        means the caller is not inside a guarded monorepo.
    """
    current = start.resolve()
    while True:
        if (current / CONFIG_FILENAME).is_file():
            return current
        if current.parent == current:
            return None
        current = current.parent


__all__ = ["CONFIG_FILENAME", "find_monorepo_root", "iter_py_files", "read_lines"]
