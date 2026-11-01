"""The rule that every ``poetry install`` and ``poetry sync`` a recipe runs passes ``--no-cache``.

WHY. Since poetry 2.4 the Authenticator mounts a CacheControl body cache at
``<repository cache>/_default_cache/_http_`` unless ``disable_cache`` is set
(python-poetry PR 10816), so every wheel an install downloads is stored
twice: once in ``artifacts``, the store the installer reads, and once as an
HTTP body nothing reads again. Measured on lavender 2026-10-30T12:52Z (MCPs
board task 22cb1ba7): the fleet's pypoetry generation held 12G of artifacts
and 12G of ``_http_``, its largest bodies the same torch wheels. Only the
command line's ``--no-cache`` sets ``disable_cache``; no config key does, so
the fix is a flag on every install, and one Makefile left without it
refills the generation it shares with every other project on the node.

WHAT THE RULE HOLDS. A recipe line, a tab-led line that is neither a
comment nor an ``echo``, that runs ``poetry install`` or ``poetry sync``
carries ``--no-cache`` on the same line. An ``echo`` is help text naming the
command, never running it, and is outside the rule.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from maketools.makefile_grammar import Violation, tracked_makefiles

#: A poetry command that downloads wheels: ``install`` or ``sync``, as its own word.
POETRY_INSTALL: Final[re.Pattern[str]] = re.compile(r"(?:^|[\s;&|(])poetry\s+(?:install|sync)\b")

#: The flag that sets poetry's ``disable_cache``.
NO_CACHE: Final[str] = "--no-cache"

#: A recipe line that prints text and runs nothing, with or without make's ``@``.
ECHO_LINE: Final[re.Pattern[str]] = re.compile(r"^@?echo(\s|$)")

#: The rule's name, as :func:`maketools.makefile_grammar.render_violation` prints it.
RULE_CACHED_INSTALL: Final[str] = f"poetry install and sync must pass {NO_CACHE}"


def check_poetry_cache_rule(path: Path, text: str) -> list[Violation]:
    """Apply the rule to one Makefile.

    Args:
        path: The Makefile.
        text: Its contents.

    Returns:
        One violation per recipe line that runs ``poetry install`` or
        ``poetry sync`` without :data:`NO_CACHE`, naming the line.
    """
    found: list[Violation] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.startswith("\t"):
            continue
        body = line[1:].strip()
        if body.startswith("#") or ECHO_LINE.match(body) is not None:
            continue
        if POETRY_INSTALL.search(body) is not None and NO_CACHE not in body.split():
            found.append(
                Violation(path=path, line_number=number, rule=RULE_CACHED_INSTALL, text=body)
            )
    return found


def lint_poetry_caches(repo_root: Path) -> list[Violation]:
    """Apply the rule to every tracked Makefile.

    Args:
        repo_root: The repository root.

    Returns:
        Every violation across the tree, in git's order.
    """
    found: list[Violation] = []
    for makefile in tracked_makefiles(repo_root):
        found.extend(check_poetry_cache_rule(makefile, makefile.read_text(encoding="utf-8")))
    return found


__all__ = [
    "ECHO_LINE",
    "NO_CACHE",
    "POETRY_INSTALL",
    "RULE_CACHED_INSTALL",
    "check_poetry_cache_rule",
    "lint_poetry_caches",
]
