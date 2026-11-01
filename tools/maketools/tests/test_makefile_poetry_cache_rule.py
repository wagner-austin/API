"""The rule that every ``poetry install`` and ``poetry sync`` a recipe runs passes ``--no-cache``.

Each case is shown FIRING on a recipe that would store every wheel twice and
silent on the form the packages carry, and the whole tree is linted for
real, so a Makefile added later without the flag fails ``make lint-makefiles``
(MCPs board task 22cb1ba7).
"""

from __future__ import annotations

from pathlib import Path

from maketools.cli import repository_root
from maketools.makefile_grammar import Violation
from maketools.makefile_poetry_cache_rule import (
    NO_CACHE,
    RULE_CACHED_INSTALL,
    check_poetry_cache_rule,
    lint_poetry_caches,
)
from tests.conftest import World

UNCACHED = """include ../../scripts/make/shell.mk

.PHONY: help install lint

help:
\t@echo "  make install    poetry install"
\techo poetry sync --with dev

install:
\t# poetry install would cache every wheel twice
\tpoetry lock
\tpoetry sync --no-cache --with dev
\tcd sub && poetry install --no-cache --with dev

lint:
\tpoetry run ruff check .
"""


def test_the_flag_is_the_one_that_disables_poetrys_http_cache() -> None:
    assert NO_CACHE == "--no-cache"
    assert RULE_CACHED_INSTALL == "poetry install and sync must pass --no-cache"


def test_installs_with_the_flag_echoes_comments_and_other_poetry_commands_pass(
    tmp_path: Path,
) -> None:
    assert check_poetry_cache_rule(tmp_path / "Makefile", UNCACHED) == []


def test_an_install_or_sync_without_the_flag_fires_on_its_own_line(tmp_path: Path) -> None:
    path = tmp_path / "Makefile"
    cached = UNCACHED.replace(" --no-cache", "")
    assert check_poetry_cache_rule(path, cached) == [
        Violation(
            path=path, line_number=12, rule=RULE_CACHED_INSTALL, text="poetry sync --with dev"
        ),
        Violation(
            path=path,
            line_number=13,
            rule=RULE_CACHED_INSTALL,
            text="cd sub && poetry install --with dev",
        ),
    ]


def test_the_flag_must_be_a_word_of_its_own(tmp_path: Path) -> None:
    path = tmp_path / "Makefile"
    spelt = "install:\n\tpoetry install --no-cache-dir\n"
    assert [v["line_number"] for v in check_poetry_cache_rule(path, spelt)] == [2]


def test_a_line_outside_any_recipe_is_outside_the_rule(tmp_path: Path) -> None:
    path = tmp_path / "Makefile"
    assert check_poetry_cache_rule(path, "INSTALL = poetry install\n") == []


def test_every_tracked_makefile_installs_without_the_http_cache() -> None:
    """The real tree, read through git with the default hooks: the guard
    that keeps a Makefile added later from refilling the cache."""
    assert lint_poetry_caches(repository_root()) == []


def test_lint_poetry_caches_reports_each_offending_makefile(world: World) -> None:
    root = repository_root()
    bad = root / "tools" / "maketools" / "runs" / "poetry-cache" / "Makefile"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("install:\n\tpoetry install --with dev\n", encoding="utf-8")
    world.tracked = [Path("tools/maketools/Makefile"), bad.relative_to(root)]
    violations = lint_poetry_caches(root)
    bad.unlink()
    assert [(v["path"], v["line_number"], v["rule"]) for v in violations] == [
        (bad, 2, RULE_CACHED_INSTALL)
    ]
