"""The rule that every package's ``check`` runs under the five-minute budget.

The operator, 2026-10-04: "ideally make check should be like 5 min for
every repo" (MCPs board task 1b152218). ``check-budget``
(:mod:`maketools.budget_run`) is what holds a package to it, and a budget
that a Makefile can leave out is no budget, so every ``check:`` here is
read for the one shape that applies it (each recipe line tab-indented):

    check:
        $(PYTHON) ../../tools/maketools/scripts/run.py check-budget
        @echo "=== ALL CHECKS PASSED ==="

    _check-unbudgeted: lint | test

``check:`` takes NO prerequisites, because make runs a prerequisite before
the recipe and so before the clock starts: a ``check: lint`` would leave
lint uncounted. Its first recipe line is the budget call at the
Makefile's own depth (computed, never assumed, like the shell include).
``_check-unbudgeted`` exists and is declared ``.PHONY``, since it is what
the call runs. The banner rule (:mod:`maketools.makefile_banner_rule`)
still holds the banner last.

A FAN-OUT IS NOT A PACKAGE. ``libs/``, ``services/`` and ``clients/``
spell ``check:`` as ``fan-out check .``, which runs each package's own
``make check``, and so each package's own budget. Budgeting the fan-out
would charge one package for all of them, so a ``check:`` whose recipe
runs ``fan-out`` is outside the rule; that is the shape of the recipe,
read every time, not a list of exempt paths.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from maketools.check_budget import UNBUDGETED_TARGET
from maketools.makefile_banner_rule import recipe_of_check_target
from maketools.makefile_grammar import Violation, makefile_depth, tracked_makefiles

#: The launcher, from the repository root.
LAUNCHER: Final[str] = "tools/maketools/scripts/run.py"

#: A ``check:`` recipe line that marks a fan-out.
FAN_OUT: Final[re.Pattern[str]] = re.compile(r"\brun\.py fan-out check\b")

#: The rule names, as :func:`maketools.makefile_grammar.render_violation` prints them.
RULE_PREREQUISITES: Final[str] = "check: takes prerequisites the budget would not count"
RULE_CALL: Final[str] = "check: must start with the budget call"
RULE_TARGET: Final[str] = f"{UNBUDGETED_TARGET} is not defined"
RULE_PHONY: Final[str] = f"{UNBUDGETED_TARGET} is not declared .PHONY"


def expected_budget_line(depth: int) -> str:
    """The exact first recipe line of a package's ``check:``.

    Args:
        depth: From :func:`maketools.makefile_grammar.makefile_depth`.

    Returns:
        For example ``$(PYTHON) ../../tools/maketools/scripts/run.py check-budget``.
    """
    return f"$(PYTHON) {'../' * depth}{LAUNCHER} check-budget"


def phony_names(text: str) -> set[str]:
    """Every name some ``.PHONY`` declaration lists, across continued lines.

    Args:
        text: The Makefile's contents.

    Returns:
        The declared names.
    """
    names: set[str] = set()
    for match in re.finditer(r"^\.PHONY:(.*?)(?=\n[^\s\\]|\Z)", text, re.MULTILINE | re.DOTALL):
        declaration: str = text[match.start(1) : match.end(1)]
        names.update(declaration.replace("\\\n", " ").split())
    return names


def check_budget_rule(repo_root: Path, path: Path, text: str) -> list[Violation]:
    """Apply the budget rule to one Makefile.

    Args:
        repo_root: The repository root.
        path: The Makefile.
        text: Its contents.

    Returns:
        No violation without a ``check:`` target or for a fan-out; one per
        broken half of the shape otherwise.
    """
    found = recipe_of_check_target(text)
    if found is None:
        return []
    target, recipe = found
    if any(FAN_OUT.search(body) is not None for _, body in recipe):
        return []
    violations: list[Violation] = []
    header = text.splitlines()[target - 1]
    if header.strip() != "check:":
        violations.append(
            Violation(path=path, line_number=target, rule=RULE_PREREQUISITES, text=header)
        )
    expected = expected_budget_line(makefile_depth(repo_root, path))
    if not recipe or recipe[0][1] != expected:
        where = recipe[0][0] if recipe else target
        violations.append(Violation(path=path, line_number=where, rule=RULE_CALL, text=expected))
    if re.search(rf"^{re.escape(UNBUDGETED_TARGET)}:", text, re.MULTILINE) is None:
        violations.append(Violation(path=path, line_number=target, rule=RULE_TARGET, text=""))
    if UNBUDGETED_TARGET not in phony_names(text):
        violations.append(Violation(path=path, line_number=1, rule=RULE_PHONY, text=""))
    return violations


def lint_budgets(repo_root: Path) -> list[Violation]:
    """Apply the budget rule to every tracked Makefile.

    Args:
        repo_root: The repository root.

    Returns:
        Every violation across the tree, in git's order.
    """
    found: list[Violation] = []
    for makefile in tracked_makefiles(repo_root):
        found.extend(check_budget_rule(repo_root, makefile, makefile.read_text(encoding="utf-8")))
    return found


__all__ = [
    "FAN_OUT",
    "LAUNCHER",
    "RULE_CALL",
    "RULE_PHONY",
    "RULE_PREREQUISITES",
    "RULE_TARGET",
    "check_budget_rule",
    "expected_budget_line",
    "lint_budgets",
    "phony_names",
]
