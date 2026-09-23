"""The rule that every ``check`` target prints the pass banner last.

Each case is shown FIRING on a Makefile that breaks the rule and silent on
the form every package now carries, and the whole tree is linted for real,
so a package added later without the banner fails ``make lint-makefiles``.
"""

from __future__ import annotations

from pathlib import Path

from maketools.cli import repository_root
from maketools.makefile_banner_rule import (
    BANNER_RECIPE,
    CHECK_BANNER,
    RULE_MISSING,
    RULE_NOT_LAST,
    check_banner_rule,
    lint_banners,
    recipe_of_check_target,
)
from maketools.makefile_grammar import Violation
from tests.conftest import World

PACKAGE = """include ../../scripts/make/shell.mk

.PHONY: lint test check

lint:
\tpoetry run mypy src tests scripts

test:
\t$(PYTHON) ../../tools/maketools/scripts/run.py test

check: lint | test
\t@echo "=== ALL CHECKS PASSED ==="
"""

FAN_OUT = """include ../scripts/make/shell.mk

check:
\t$(PYTHON) ../tools/maketools/scripts/run.py fan-out check .
\t@echo "=== ALL CHECKS PASSED ==="
"""

SPACED = """check: lint sources | test agent-selftest
\t@echo ""
\t# the banner the review gate reads
\t@echo "=== ALL CHECKS PASSED ==="

\t@echo ""
"""


def test_the_banner_is_the_one_the_review_gate_reads() -> None:
    assert CHECK_BANNER == "=== ALL CHECKS PASSED ==="
    assert BANNER_RECIPE == '@echo "=== ALL CHECKS PASSED ==="'


def test_the_check_recipe_is_read_to_the_next_target() -> None:
    assert recipe_of_check_target(PACKAGE) == (11, [(12, BANNER_RECIPE)])
    assert recipe_of_check_target(SPACED) == (
        1,
        [(2, '@echo ""'), (4, BANNER_RECIPE), (6, '@echo ""')],
    )
    assert recipe_of_check_target("lint:\n\tx\n") is None


def test_a_package_a_fan_out_and_an_echo_after_the_banner_all_pass(tmp_path: Path) -> None:
    path = tmp_path / "Makefile"
    assert check_banner_rule(path, PACKAGE) == []
    assert check_banner_rule(path, FAN_OUT) == []
    assert check_banner_rule(path, SPACED) == []


def test_a_makefile_without_a_check_target_is_outside_the_rule(tmp_path: Path) -> None:
    no_check = "include scripts/make/shell.mk\n\nup:\n\tx\n"
    assert check_banner_rule(tmp_path / "Makefile", no_check) == []


def test_a_check_without_the_banner_fires_on_the_target_line(tmp_path: Path) -> None:
    path = tmp_path / "Makefile"
    silent = PACKAGE.replace('\t@echo "=== ALL CHECKS PASSED ==="\n', "")
    assert check_banner_rule(path, silent) == [
        Violation(path=path, line_number=11, rule=RULE_MISSING, text=BANNER_RECIPE)
    ]
    misspelt = PACKAGE.replace("ALL CHECKS PASSED", "ALL CHECKS PASS")
    assert [v["rule"] for v in check_banner_rule(path, misspelt)] == [RULE_MISSING]


def test_a_command_after_the_banner_fires_on_its_own_line(tmp_path: Path) -> None:
    path = tmp_path / "Makefile"
    late = PACKAGE + "\t$(PYTHON) scripts/late.py\n\t@echo done\n"
    assert check_banner_rule(path, late) == [
        Violation(path=path, line_number=13, rule=RULE_NOT_LAST, text="$(PYTHON) scripts/late.py")
    ]


def test_only_the_last_banner_is_the_one_that_must_end_the_recipe(tmp_path: Path) -> None:
    path = tmp_path / "Makefile"
    twice = FAN_OUT.replace("\t$(PYTHON)", '\t@echo "=== ALL CHECKS PASSED ==="\n\t$(PYTHON)')
    assert check_banner_rule(path, twice) == []


def test_every_tracked_makefile_prints_the_banner_last() -> None:
    """The real tree, read through git with the default hooks: the guard
    that keeps a package added later from falling outside the rule."""
    assert lint_banners(repository_root()) == []


def test_lint_banners_reports_each_offending_makefile(world: World) -> None:
    root = repository_root()
    bad = root / "tools" / "maketools" / "runs" / "banner" / "Makefile"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text(PACKAGE.replace(f"\t{BANNER_RECIPE}\n", ""), encoding="utf-8")
    world.tracked = [Path("tools/maketools/Makefile"), bad.relative_to(root)]
    violations = lint_banners(root)
    bad.unlink()
    assert [(v["path"], v["rule"]) for v in violations] == [(bad, RULE_MISSING)]
