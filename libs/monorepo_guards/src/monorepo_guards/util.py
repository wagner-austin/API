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


def package_of(path: Path) -> str:
    """Name the package a file belongs to.

    Args:
        path: The file.

    Returns:
        The directory name immediately above the ``src``/``tests``/``scripts``
        marker where the monorepo's layout puts the package, or the file's own
        parent when the path carries no marker.
    """
    parts = path.as_posix().split("/")
    for marker in ("src", "tests", "scripts"):
        if marker in parts:
            index = parts.index(marker)
            if index > 0:
                return parts[index - 1]
    return path.parent.name


def imported_names(tree: ast.Module) -> set[str]:
    """Collect the names a module BINDS through an import.

    Imports only, deliberately: collecting every ``Name`` node lets a local
    variable that happens to share a symbol's spelling satisfy a rule, and a
    rule that can be passed by accident is worse than no rule because it reads
    as verified.

    Separate from :func:`imported_module_names` rather than a flag over one
    function, because the two answer different questions and a rule needs
    exactly one of them. ``from platform_core.run_record import RunRecord``
    binds ``RunRecord`` and names the module ``run_record``; a rule asking
    "does this module build a record" must not see the second, or every
    module that annotates a record it read back becomes a producer.

    Args:
        tree: Parsed module.

    Returns:
        The bound names, taking ``asname`` where one is given, and the final
        component of a dotted plain import.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.rsplit(".", maxsplit=1)[-1])
    return names


def imported_module_names(tree: ast.Module) -> set[str]:
    """Collect the final component of every module a file imports FROM.

    Args:
        tree: Parsed module.

    Returns:
        The last dotted component of each ``from X.Y import ...`` module. A
        relative import with no module contributes nothing.
    """
    return {
        node.module.rsplit(".", maxsplit=1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }


__all__ = [
    "CONFIG_FILENAME",
    "find_monorepo_root",
    "imported_module_names",
    "imported_names",
    "iter_py_files",
    "package_of",
    "parse_source",
    "read_lines",
    "read_source",
]
