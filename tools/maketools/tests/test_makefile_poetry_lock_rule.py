"""The rule that no recipe runs ``poetry lock`` (MCPs board task 972ea464).

Each case is shown FIRING on the recipe shape every package carried before
this rule, which re-resolved its lock on every ``lint`` and ``test``, and
silent on the shape that verifies it; the whole tree is linted for real, so
a Makefile added later that re-resolves fails ``make lint-makefiles``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maketools.cli import repository_root
from maketools.makefile_grammar import Violation
from maketools.makefile_poetry_lock_rule import (
    RULE_RECIPE_LOCK,
    check_poetry_lock_rule,
    lint_poetry_locks,
)
from tests.conftest import World

#: ``libs/covenant_nn/Makefile``'s lint and test openers before this rule.
RESOLVING = """include ../../scripts/make/shell.mk

.PHONY: help lint test

help:
\t@echo "  make lock    poetry lock"
\techo poetry lock

lint:
\t# poetry lock re-resolved torch's index here
\tpoetry lock
\tpoetry sync --no-cache --with dev

test:
\t-poetry lock --regenerate
\tpoetry sync --no-cache --with dev
"""

#: The recipe line that replaces each resolution.
LOCK_CHECK = "\t$(PYTHON) ../../tools/maketools/scripts/run.py lock-check\n"

#: The same recipes once they verify the lock instead.
VERIFYING = RESOLVING.replace("\tpoetry lock\n", LOCK_CHECK).replace(
    "\t-poetry lock --regenerate\n", LOCK_CHECK
)


def test_the_rule_names_the_command_that_replaces_the_resolution() -> None:
    assert RULE_RECIPE_LOCK == (
        "a recipe verifies poetry.lock with run.py lock-check and never runs poetry lock"
    )


def test_each_recipe_line_that_resolves_fires_with_makes_prefix_stripped(tmp_path: Path) -> None:
    path = tmp_path / "Makefile"
    assert check_poetry_lock_rule(path, RESOLVING) == [
        Violation(path=path, line_number=11, rule=RULE_RECIPE_LOCK, text="poetry lock"),
        Violation(
            path=path, line_number=15, rule=RULE_RECIPE_LOCK, text="poetry lock --regenerate"
        ),
    ]


def test_verifying_recipes_echoes_and_comments_pass(tmp_path: Path) -> None:
    assert check_poetry_lock_rule(tmp_path / "Makefile", VERIFYING) == []


@pytest.mark.parametrize(
    "recipe",
    [
        "poetry locks",
        "poetry check --lock",
        "poetrylock",
        "echo; poetry",
        "poetry -C lock sync",
        "poetry run lock",
    ],
)
def test_other_poetry_commands_and_lookalikes_pass(tmp_path: Path, recipe: str) -> None:
    assert check_poetry_lock_rule(tmp_path / "Makefile", f"x:\n\t{recipe}\n") == []


@pytest.mark.parametrize(
    "recipe",
    [
        "poetry -C sub lock",
        "poetry --directory sub lock",
        "poetry -P sub --no-interaction lock",
        "poetry --project=sub lock",
    ],
)
def test_lock_behind_poetrys_global_options_fires(tmp_path: Path, recipe: str) -> None:
    text = f"x:\n\t{recipe}\n"
    assert [v["text"] for v in check_poetry_lock_rule(tmp_path / "Makefile", text)] == [recipe]


@pytest.mark.parametrize("separator", [" ", ";", "&", "|", "("])
def test_lock_after_any_command_separator_fires(tmp_path: Path, separator: str) -> None:
    text = f"x:\n\ttrue{separator}poetry lock\n"
    assert [v["line_number"] for v in check_poetry_lock_rule(tmp_path / "Makefile", text)] == [2]


def test_a_line_outside_any_recipe_is_outside_the_rule(tmp_path: Path) -> None:
    assert check_poetry_lock_rule(tmp_path / "Makefile", "LOCK = poetry lock\n") == []


def test_every_tracked_makefile_verifies_its_lock() -> None:
    """The real tree, read through git with the default hooks: the guard
    that keeps a Makefile added later from re-resolving on every check."""
    assert lint_poetry_locks(repository_root()) == []


def test_lint_poetry_locks_reports_each_offending_makefile(world: World) -> None:
    root = repository_root()
    bad = root / "tools" / "maketools" / "runs" / "poetry-lock" / "Makefile"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("lint:\n\tpoetry lock\n", encoding="utf-8")
    world.tracked = [Path("tools/maketools/Makefile"), bad.relative_to(root)]
    violations = lint_poetry_locks(root)
    bad.unlink()
    assert [(v["path"], v["line_number"], v["rule"]) for v in violations] == [
        (bad, 2, RULE_RECIPE_LOCK)
    ]
