"""The rule that every package's ``check`` runs under the five-minute budget.

Each half of the shape is shown FIRING on a Makefile that breaks it and
silent on the form every package now carries and on a fan-out, and the
whole tree is linted for real, so a package added later without the
budget fails ``make lint-makefiles`` (MCPs board task 1b152218).
"""

from __future__ import annotations

from pathlib import Path

from maketools.cli import repository_root
from maketools.makefile_budget_rule import (
    RULE_CALL,
    RULE_PHONY,
    RULE_PREREQUISITES,
    RULE_TARGET,
    check_budget_rule,
    expected_budget_line,
    lint_budgets,
    phony_names,
)
from maketools.makefile_grammar import Violation
from tests.conftest import World

PACKAGE = """include ../../scripts/make/shell.mk

.PHONY: lint test check \\
\t_check-unbudgeted

lint:
\tpoetry run mypy src tests scripts

test:
\t$(PYTHON) ../../tools/maketools/scripts/run.py test

check:
\t$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
\t@echo "=== ALL CHECKS PASSED ==="

_check-unbudgeted: lint | test
"""

FAN_OUT = """include ../scripts/make/shell.mk

check:
\t$(PYTHON) ../tools/maketools/scripts/run.py fan-out check .
\t@echo "=== ALL CHECKS PASSED ==="
"""

OLD_FORM = """include ../../scripts/make/shell.mk

.PHONY: lint test check

check: lint | test
\t@echo "=== ALL CHECKS PASSED ==="
"""


def _package(repo_root: Path) -> Path:
    """A Makefile path two directories below the root.

    Args:
        repo_root: The root.

    Returns:
        ``libs/pkg/Makefile`` under it.
    """
    return repo_root / "libs" / "pkg" / "Makefile"


def test_the_budget_call_is_spelled_at_the_makefile_s_depth() -> None:
    assert expected_budget_line(2) == (
        "$(PYTHON) ../../tools/maketools/scripts/run.py check-budget"
    )
    assert expected_budget_line(3) == (
        "$(PYTHON) ../../../tools/maketools/scripts/run.py check-budget"
    )


def test_phony_names_are_read_across_continued_lines() -> None:
    assert phony_names(PACKAGE) == {"lint", "test", "check", "_check-unbudgeted"}


def test_the_budgeted_form_and_a_fan_out_pass(tmp_path: Path) -> None:
    assert check_budget_rule(tmp_path, _package(tmp_path), PACKAGE) == []
    assert check_budget_rule(tmp_path, tmp_path / "libs" / "Makefile", FAN_OUT) == []
    assert check_budget_rule(tmp_path, _package(tmp_path), "up:\n\tx\n") == []


def test_the_old_form_fires_every_half_of_the_rule(tmp_path: Path) -> None:
    path = _package(tmp_path)
    assert check_budget_rule(tmp_path, path, OLD_FORM) == [
        Violation(path=path, line_number=5, rule=RULE_PREREQUISITES, text="check: lint | test"),
        Violation(
            path=path,
            line_number=6,
            rule=RULE_CALL,
            text="$(PYTHON) ../../tools/maketools/scripts/run.py check-budget",
        ),
        Violation(path=path, line_number=5, rule=RULE_TARGET, text=""),
        Violation(path=path, line_number=1, rule=RULE_PHONY, text=""),
    ]


def test_a_check_with_no_recipe_is_told_the_call_it_needs(tmp_path: Path) -> None:
    path = _package(tmp_path)
    found = check_budget_rule(tmp_path, path, "check:\n\n_check-unbudgeted:\n")
    assert [violation["rule"] for violation in found] == [RULE_CALL, RULE_PHONY]
    assert found[0]["line_number"] == 1


def test_a_call_at_the_wrong_depth_fires(tmp_path: Path) -> None:
    shallow = PACKAGE.replace("../../tools/maketools/scripts/run.py check-budget", "x")
    (violation,) = check_budget_rule(tmp_path, _package(tmp_path), shallow)
    assert violation["rule"] == RULE_CALL
    assert violation["line_number"] == 13


def test_every_tracked_makefile_holds_the_rule() -> None:
    """The real tree, read through git with the default hooks: the guard
    that keeps a package added later from checking outside the budget."""
    assert lint_budgets(repository_root()) == []


def test_lint_budgets_reports_each_offending_makefile(world: World) -> None:
    root = repository_root()
    world.tracked = [Path("tools/maketools/Makefile"), Path("libs/Makefile")]
    assert lint_budgets(root) == []
    bad = root / "tools" / "maketools" / "runs" / "budget" / "Makefile"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text(OLD_FORM, encoding="utf-8")
    world.tracked = [Path("tools/maketools/Makefile"), bad.relative_to(root)]
    violations = lint_budgets(root)
    bad.unlink()
    assert [(v["path"], v["rule"]) for v in violations] == [
        (bad, RULE_PREREQUISITES),
        (bad, RULE_CALL),
        (bad, RULE_TARGET),
        (bad, RULE_PHONY),
    ]
