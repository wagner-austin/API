"""Lint: ban semantically-empty field names in wire-protocol TypedDicts.

A TypedDict field named ``unk1`` / ``unk2`` / ``unknown_byte`` /
``padding_3`` documents nothing -- the decoder is admitting it does not
understand the byte. Promoting such fields to their real meaning is the
ongoing Phase 5 work; this guard prevents drift by rejecting new
violations at lint time.

The guard scans every ``TypedDict`` subclass in the wire-format type
modules -- the ``src/tankpit_bot/protocol/types/`` package (every
payload family inside it) and ``src/tankpit_bot/container/types.py`` --
and flags any annotation whose field name matches one of the banned
patterns.

Exit status:
  0 -- no violations.
  1 -- one or more violations; each is printed to stderr with the
       file path, line number, the offending field name, and the
       TypedDict it lives in.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from tankpit_bot import _test_hooks

#: Wire-format type locations scanned by the guard. A directory is
#: expanded to every ``*.py`` inside it, so a new payload family added
#: under ``protocol/types/`` is covered the day it lands rather than the
#: day someone remembers to list it. Other modules (diagnostics, capture
#: stats) intentionally use ``unknown_*`` names to track unknowns and
#: are out of scope.
DEFAULT_TARGETS: tuple[Path, ...] = (
    Path("src/tankpit_bot/protocol/types"),
    Path("src/tankpit_bot/container/types.py"),
)

#: Regexes matching banned field names. Each one represents a different
#: failure mode that historically produced wrong decoder output:
#:
#:  - ``unk\d+`` -- placeholder for a byte whose semantics were never
#:    determined (e.g. ``unk1``, ``unk2`` on ShootEventDict before the
#:    2026-06-20 ``aim_x`` / ``aim_y`` promotion).
#:  - ``unknown_byte\w*`` -- same intent, alternate spelling.
#:  - ``padding\d*`` -- assumes a byte is padding without verifying. Wire
#:    formats rarely have true padding; this name is almost always a
#:    misread of a real field.
#:  - ``reserved\d*`` -- same as ``padding``; usually means "I don't know
#:    what this is."
_BANNED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^unk\d+$"),
    re.compile(r"^unknown_byte\w*$"),
    re.compile(r"^padding\d*$"),
    re.compile(r"^reserved\d*$"),
)


@dataclass(frozen=True)
class Violation:
    """One offending TypedDict field.

    Attributes:
        path: Source file the violation came from.
        line_no: 1-based source line number.
        typed_dict_name: Name of the TypedDict subclass that declared
            the field.
        field_name: The offending field name.
    """

    path: Path
    line_no: int
    typed_dict_name: str
    field_name: str

    def format(self) -> str:
        """Return a one-line stderr-friendly description.

        Returns:
            ``"<path>:<line>: <typed_dict_name>.<field_name> uses a
            banned undecoded-field name"`` style string.
        """
        return (
            f"{self.path}:{self.line_no}: "
            f"{self.typed_dict_name}.{self.field_name} uses a banned "
            f"undecoded-field name"
        )


def _is_typed_dict_base(base: ast.expr) -> bool:
    """Return True when ``base`` references ``TypedDict``.

    Handles both ``TypedDict`` (re-exported) and
    ``typing.TypedDict`` / ``typing_extensions.TypedDict`` style.

    Args:
        base: AST node from a ``ClassDef.bases`` list.

    Returns:
        True when the base resolves to one of the supported TypedDict
        spellings.
    """
    if isinstance(base, ast.Name) and base.id == "TypedDict":
        return True
    return isinstance(base, ast.Attribute) and base.attr == "TypedDict"


def _find_violations_in_classdef(path: Path, node: ast.ClassDef) -> list[Violation]:
    """Return every banned-field violation in one TypedDict subclass.

    Args:
        path: Source file path used in the violation report.
        node: ``ast.ClassDef`` for a TypedDict subclass.

    Returns:
        Zero or more :class:`Violation` instances.
    """
    if not any(_is_typed_dict_base(base) for base in node.bases):
        return []
    violations: list[Violation] = []
    for stmt in node.body:
        if not isinstance(stmt, ast.AnnAssign):
            continue
        if not isinstance(stmt.target, ast.Name):
            continue
        name = stmt.target.id
        if any(pattern.match(name) for pattern in _BANNED_PATTERNS):
            violations.append(
                Violation(
                    path=path,
                    line_no=stmt.lineno,
                    typed_dict_name=node.name,
                    field_name=name,
                )
            )
    return violations


def find_violations_in_source(path: Path, source: str) -> list[Violation]:
    """Scan one source file's text for violations.

    Args:
        path: Source file path (used purely for reporting).
        source: File text.

    Returns:
        All :class:`Violation` instances found.

    Raises:
        SyntaxError: When ``source`` is not valid Python; the caller
            sees the underlying parser error verbatim so the surface
            stays small.
    """
    tree = ast.parse(source, filename=str(path))
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            violations.extend(_find_violations_in_classdef(path, node))
    return violations


def expand_targets(paths: tuple[Path, ...]) -> list[Path]:
    """Resolve each target to the concrete source files to scan.

    A ``.py`` target is one module and is scanned as-is. Any other
    target is a package directory and expands to every ``*.py`` it
    holds, so a payload family added under ``protocol/types/`` is
    covered without editing this module.

    Directory listing and existence both route through
    :mod:`tankpit_bot._test_hooks` so tests inject fakes without
    touching disk.

    Args:
        paths: Configured targets, each a ``.py`` module or a package
            directory.

    Returns:
        Concrete file paths, in target order.

    Raises:
        FileNotFoundError: When a module target does not exist, or a
            package target holds no ``*.py``.
    """
    out: list[Path] = []
    for path in paths:
        if path.suffix == ".py":
            if not _test_hooks.path_exists(path):
                raise FileNotFoundError(path)
            out.append(path)
            continue
        members = _test_hooks.glob_paths(path, "*.py")
        if not members:
            raise FileNotFoundError(path)
        out.extend(members)
    return out


def find_violations(paths: tuple[Path, ...]) -> list[Violation]:
    """Scan each target and aggregate every violation across all files.

    Targets are expanded by :func:`expand_targets`, so a package
    directory contributes each of its modules. File reads route through
    :mod:`tankpit_bot._test_hooks` so tests inject fakes without
    touching disk.

    Args:
        paths: Source targets to scan.

    Returns:
        All :class:`Violation` instances, in scan order.

    Raises:
        FileNotFoundError: When a target does not exist on the
            (real or fake) filesystem.
    """
    out: list[Violation] = []
    for path in expand_targets(paths):
        source = _test_hooks.read_text(path)
        out.extend(find_violations_in_source(path, source))
    return out


def run(targets: tuple[Path, ...] = DEFAULT_TARGETS) -> int:
    """Run the guard and return the CLI exit code.

    Args:
        targets: Source targets to scan, each a file or a package
            directory. Defaults to :data:`DEFAULT_TARGETS`.

    Returns:
        ``0`` when no violations were found; ``1`` otherwise.
    """
    scanned = len(expand_targets(targets))
    violations = find_violations(targets)
    if not violations:
        sys.stdout.write(f"check_undecoded_fields: clean ({scanned} files scanned)\n")
        return 0
    sys.stderr.write(f"check_undecoded_fields: {len(violations)} violation(s):\n")
    for violation in violations:
        sys.stderr.write(f"  {violation.format()}\n")
    return 1


def main() -> None:
    """Entry point for the ``tankpit-check-undecoded-fields`` script."""
    sys.exit(run())


if __name__ == "__main__":
    main()
