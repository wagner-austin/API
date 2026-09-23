"""The rule that every ``check`` target prints the pass banner last.

WHY A BANNER AND NOT THE EXIT CODE ALONE. The review gate that re-runs a
cited commit's check (``packages/session-audit/src/session_audit/review/
shapes.py`` in ~/PROJECTS/MCPs) rules a check PASSED only when make exits 0
AND the output carries ``=== ALL CHECKS PASSED ===``, because either signal
alone was measured to lie. Every MCPs package printed it; no API package
did, so ``tools/fleet`` at ca196dda16cb ended ``750 passed`` at 100 percent
coverage, exited 0, and was ruled FAILED, banner absent (MCPs board task
e725c56a). The operator chose, 2026-09-23, that API adopt the banner rather
than the gate weaken: "go with option (a) for e725c56a, add the banner".

WHAT THE RULE HOLDS. A Makefile that defines ``check:`` carries the exact
recipe line :data:`BANNER_RECIPE`, and no command other than an ``@echo``
follows it, so the banner can only print once everything before it has
succeeded. make stops at the first failing recipe line or prerequisite, so
a banner in that position is proof the whole target ran. A Makefile with no
``check:`` target (the root, which has none) is outside the rule.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from maketools.makefile_grammar import Violation, tracked_makefiles

#: The line the review gate reads as positive evidence of a passed check.
CHECK_BANNER: Final[str] = "=== ALL CHECKS PASSED ==="

#: The exact recipe line that prints it, tab and nothing else before it.
BANNER_RECIPE: Final[str] = f'@echo "{CHECK_BANNER}"'

#: A recipe line allowed after the banner: output only, never a command.
ECHO_LINE: Final[re.Pattern[str]] = re.compile(r"^@echo(\s|$)")

#: The rule names, as :func:`maketools.makefile_grammar.render_violation` prints them.
RULE_MISSING: Final[str] = "check: must print the pass banner"
RULE_NOT_LAST: Final[str] = "check: runs a command after the pass banner"


def recipe_of_check_target(text: str) -> tuple[int, list[tuple[int, str]]] | None:
    """The ``check:`` target's line and its recipe lines.

    Args:
        text: The Makefile's contents.

    Returns:
        The target's 1-based line and its ``(line number, recipe)`` pairs,
        each recipe without its leading tab and surrounding space, blank and
        comment lines left out; ``None`` when the Makefile has no ``check:``.
    """
    target: int | None = None
    recipe: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if target is not None:
            if line.startswith("\t"):
                body = line[1:].strip()
                if body != "" and not body.startswith("#"):
                    recipe.append((number, body))
                continue
            if line.strip() == "":
                continue
            break
        if re.match(r"^check:", line):
            target = number
    if target is None:
        return None
    return target, recipe


def check_banner_rule(path: Path, text: str) -> list[Violation]:
    """Apply the banner rule to one Makefile.

    Args:
        path: The Makefile.
        text: Its contents.

    Returns:
        No violation without a ``check:`` target; one naming the target's
        line when the recipe never prints :data:`BANNER_RECIPE`; one per
        non-``@echo`` command after the banner otherwise.
    """
    found = recipe_of_check_target(text)
    if found is None:
        return []
    target, recipe = found
    banners = [index for index, (_, body) in enumerate(recipe) if body == BANNER_RECIPE]
    if not banners:
        return [Violation(path=path, line_number=target, rule=RULE_MISSING, text=BANNER_RECIPE)]
    return [
        Violation(path=path, line_number=number, rule=RULE_NOT_LAST, text=body)
        for number, body in recipe[banners[-1] + 1 :]
        if ECHO_LINE.match(body) is None
    ]


def lint_banners(repo_root: Path) -> list[Violation]:
    """Apply the banner rule to every tracked Makefile.

    Args:
        repo_root: The repository root.

    Returns:
        Every violation across the tree, in git's order.
    """
    found: list[Violation] = []
    for makefile in tracked_makefiles(repo_root):
        found.extend(check_banner_rule(makefile, makefile.read_text(encoding="utf-8")))
    return found


__all__ = [
    "BANNER_RECIPE",
    "CHECK_BANNER",
    "ECHO_LINE",
    "RULE_MISSING",
    "RULE_NOT_LAST",
    "check_banner_rule",
    "lint_banners",
    "recipe_of_check_target",
]
