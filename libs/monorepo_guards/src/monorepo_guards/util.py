"""Finding the files to check, and reading each of them exactly once.

Every rule used to read and parse every file for itself. Measured on
covenant-radar-api (367 files, 3.10 MB) before this module cached anything:

    read all once     0.014s
    parse all once    0.371s
    31 rules x a pass 12.0s predicted -- 13.0s measured
    ast.parse calls   5597 over 368 distinct files (15.2 each)

So roughly 92% of a guard run was re-parsing bytes it had already parsed, and
the rules' actual work -- every AST walk, every violation built -- was about
one second of the thirteen. `parse_source` fixes that without changing the
`Rule` protocol, because the redundancy was never inside a rule: it was
between rules, over the same paths.

The cache is keyed on file IDENTITY (path, mtime, size) rather than path
alone, which is what makes it a memoised pure function rather than state.
Two runs in one process see an edited file as a different key, so the
guard-shim test -- which calls the entry point three times -- cannot read a
stale tree.
"""

from __future__ import annotations

import ast
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


_TEXT_CACHE: dict[tuple[str, int, int], str] = {}
_TREE_CACHE: dict[tuple[str, int, int], ast.Module] = {}


def _identity(path: Path) -> tuple[str, int, int]:
    """Key a file by what it IS, not by what it is called.

    Args:
        path: File to identify.

    Returns:
        Path, modification time and size. Keying on the path alone would let
        a second run in the same process read a tree built from bytes that no
        longer exist on disk.
    """
    stat = path.stat()
    return (str(path), stat.st_mtime_ns, stat.st_size)


def read_source(path: Path) -> str:
    """Read a file's text, once per version of that file.

    Args:
        path: File to read.

    Returns:
        The decoded text. ``utf-8-sig`` strips a leading byte-order mark,
        which CPython itself tolerates in a source file -- so a guard that
        choked on one was rejecting a module the interpreter runs happily.

    Raises:
        UnicodeDecodeError: If the bytes are not valid UTF-8. A file the
            interpreter cannot read is a real problem in the tree being
            checked, not something to skip past.
    """
    key = _identity(path)
    cached = _TEXT_CACHE.get(key)
    if cached is not None:
        return cached
    text = path.read_text(encoding="utf-8-sig", errors="strict")
    _TEXT_CACHE[key] = text
    return text


def parse_source(path: Path) -> ast.Module:
    """Parse a file, once per version of that file.

    This is the function that turned 5,597 parses into 368.

    Args:
        path: Python file to parse.

    Returns:
        The module's AST. Callers must not mutate it -- every rule in a run
        receives the same object, which is the entire point.

    Raises:
        SyntaxError: If the file does not parse. Propagated rather than
            skipped: a file the guard cannot read is a file the guard is not
            checking, and silently not checking something is how a rule comes
            to report zero violations it never looked for.
    """
    key = _identity(path)
    cached = _TREE_CACHE.get(key)
    if cached is not None:
        return cached
    tree = ast.parse(read_source(path), filename=str(path))
    _TREE_CACHE[key] = tree
    return tree


def read_lines(path: Path) -> list[str]:
    """Read a file's lines, once per version of that file.

    Args:
        path: File to read.

    Returns:
        The text split into lines, without terminators.
    """
    return read_source(path).splitlines()


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


__all__ = [
    "CONFIG_FILENAME",
    "find_monorepo_root",
    "iter_py_files",
    "parse_source",
    "read_lines",
    "read_source",
]
