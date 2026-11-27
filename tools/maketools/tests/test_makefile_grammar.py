"""The portable recipe grammar.

Every rule is shown FIRING on a line that breaks it and silent on the
portable form, because a grammar only ever seen passing clean text is a
grammar nobody has watched catch anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maketools.makefile_grammar import (
    Violation,
    check_makefile,
    check_recipe_line,
    command_lines,
    expected_include,
    lint_grammar,
    makefile_depth,
    recipe_body,
    render_violation,
)
from tests.conftest import World

PORTABLE = """include ../scripts/make/shell.mk

.PHONY: lint test check

lint:
\t$(PYTHON) ../tools/maketools/scripts/run.py venv-check
\tpoetry run mypy src tests scripts

test:
\t$(PYTHON) ../tools/maketools/scripts/run.py test

check: lint | test
\t@echo "=== ALL CHECKS PASSED ==="
"""


def rules_of(path: Path, line: str) -> list[str]:
    return [v["rule"] for v in check_recipe_line(path, 1, "\t" + line)]


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / "libs" / "platform_core").mkdir(parents=True)
    (tmp_path / "tools").mkdir()
    return tmp_path


def test_the_depth_is_counted_from_the_root(repo: Path) -> None:
    assert makefile_depth(repo, repo / "Makefile") == 0
    assert makefile_depth(repo, repo / "libs" / "Makefile") == 1
    assert makefile_depth(repo, repo / "libs" / "platform_core" / "Makefile") == 2


def test_the_include_line_is_computed_per_depth() -> None:
    assert expected_include(0) == "include scripts/make/shell.mk"
    assert expected_include(2) == "include ../../scripts/make/shell.mk"


def test_the_recipe_body_drops_make_prefixes() -> None:
    assert recipe_body("\t@-echo hi") == "echo hi"
    assert recipe_body("\t  poetry run x  ") == "poetry run x"
    assert recipe_body("\t+echo hi") == "echo hi"
    assert recipe_body("\t@ -\t+echo hi") == "echo hi"


def test_command_lines_are_the_recipe_lines_that_run_something() -> None:
    text = (
        "VAR = poetry sync\n"
        "x:\n"
        "\t\n"
        "\t# a comment\n"
        "\t@-echo help\n"
        "\techo\n"
        "\techoes x\n"
        "\t@poetry sync --no-cache\n"
        "\t+make -C sub\n"
    )
    assert command_lines(text) == [
        (7, "echoes x"),
        (8, "poetry sync --no-cache"),
        (9, "make -C sub"),
    ]


@pytest.mark.parametrize(
    ("line", "rule"),
    [
        ("npm run a && npm run b", "&& or || chaining"),
        ("npm run a || true", "&& or || chaining"),
        ("echo $$HOME", "shell variable or substitution ($$)"),
        ("echo `date`", "backtick"),
        ("[ -f x ]", "[ test"),
        ("test -f x || [ -d y ]", "[ test"),
        ("if test -f x", "sh control flow"),
        ("for f in *", "sh control flow"),
        ("case x in", "sh control flow"),
        ("while true", "sh control flow"),
        ("until false", "sh control flow"),
        ("a; b", "; separator (one command per line)"),
        ("cmd > out.txt", "redirection or pipe"),
        ("cmd 2>/dev/null", "redirection or pipe"),
        ("cmd < in.txt", "redirection or pipe"),
        ("cmd | tail", "redirection or pipe"),
        ("cd libs/platform_core", "cd (use make -C or a script)"),
        ("+cd libs/platform_core", "cd (use make -C or a script)"),
        ("@ -cd libs/platform_core", "cd (use make -C or a script)"),
        ("-+@cd libs/platform_core", "cd (use make -C or a script)"),
        ("+if test -f x", "sh control flow"),
    ],
)
def test_each_banned_form_fires_its_rule(tmp_path: Path, line: str, rule: str) -> None:
    assert rule in rules_of(tmp_path / "Makefile", line)


@pytest.mark.parametrize(
    "line",
    [
        "poetry run mypy src tests scripts",
        '@echo "=== ALL CHECKS PASSED ==="',
        '@echo ""',
        "@echo done",
        "@echo",
        '+echo "left ran"',
        "@ -echo 'left ran'",
        "$(MAKE) -C web test",
        "@echoes-tool --flag",
        "poetry run ruff check src \\",
        'docker ps --format "table {{.Names}}\\t{{.Status}}"',
        "iffy-tool --case",
        "$(PYTHON) ../../tools/maketools/scripts/run.py test",
        "poetry run pytest -n auto -v --cov-branch --cov=src",
        "make -C libs/platform_core check",
        "docker compose --project-directory services/qr-api up -d --build",
        "@# a recipe comment",
        "",
    ],
)
def test_the_portable_forms_pass(tmp_path: Path, line: str) -> None:
    assert rules_of(tmp_path / "Makefile", line) == []


@pytest.mark.parametrize("condition", ["ifdef VERBOSE", "ifeq ($(CI),1)", "ifneq ($(CI),1)"])
def test_a_conditional_exempts_neither_arms_recipes(repo: Path, condition: str) -> None:
    text = PORTABLE + f"{condition}\nx:\n\tnpm run a && npm run b\nelse\nx:\n\tcmd | tail\nendif\n"
    rules = [v["rule"] for v in check_makefile(repo, repo / "libs" / "Makefile", text)]
    assert rules == ["&& or || chaining", "redirection or pipe"]


def test_shell_functions_are_checked_in_both_conditional_arms(repo: Path) -> None:
    text = PORTABLE + "ifdef FAST\nX := $(shell pwd)\nelse\nX := $(shell ls)\nendif\n"
    violations = check_makefile(repo, repo / "libs" / "Makefile", text)
    assert [v["text"] for v in violations] == ["X := $(shell pwd)", "X := $(shell ls)"]


def test_the_shell_variable_stays_banned_inside_conditionals(repo: Path) -> None:
    text = PORTABLE + "ifdef FAST\nSHELL := /bin/bash\nendif\n"
    (violation,) = check_makefile(repo, repo / "libs" / "Makefile", text)
    assert violation["rule"] == "SHELL is set only in scripts/make/shell.mk"


def test_a_stray_else_or_endif_outside_any_conditional_is_consumed_harmlessly(repo: Path) -> None:
    text = PORTABLE + "else\nendif\n"
    assert check_makefile(repo, repo / "libs" / "Makefile", text) == []


def test_a_portable_makefile_has_no_violations(repo: Path) -> None:
    assert check_makefile(repo, repo / "libs" / "Makefile", PORTABLE) == []


def test_a_missing_prologue_fires_on_the_first_code_line(repo: Path) -> None:
    text = "# comment\n\n.PHONY: lint\nlint:\n\tpoetry run mypy\n"
    (violation,) = check_makefile(repo, repo / "libs" / "Makefile", text)
    assert violation["rule"] == "first line must be the shell prologue"
    assert violation["line_number"] == 3
    assert violation["text"] == "include ../scripts/make/shell.mk"


def test_a_prologue_at_the_wrong_depth_fires(repo: Path) -> None:
    violations = check_makefile(repo, repo / "libs" / "platform_core" / "Makefile", PORTABLE)
    assert [v["rule"] for v in violations] == ["first line must be the shell prologue"]


def test_an_empty_makefile_fires_on_line_one(repo: Path) -> None:
    (violation,) = check_makefile(repo, repo / "libs" / "Makefile", "")
    assert violation["line_number"] == 1


def test_setting_the_shell_outside_the_prologue_fires(repo: Path) -> None:
    text = PORTABLE + "SHELL := /bin/bash\n.SHELLFLAGS := -c\n"
    rules = [v["rule"] for v in check_makefile(repo, repo / "libs" / "Makefile", text)]
    assert rules == [
        "SHELL is set only in scripts/make/shell.mk",
        ".SHELLFLAGS is set only in scripts/make/shell.mk",
    ]


def test_the_shell_function_fires_anywhere_in_the_file(repo: Path) -> None:
    text = PORTABLE + "FILE := $(shell ls)\n"
    (violation,) = check_makefile(repo, repo / "libs" / "Makefile", text)
    assert violation["rule"] == "$(shell ...) runs the shell while make parses"
    assert violation["text"] == "FILE := $(shell ls)"


def test_a_comment_mentioning_the_shell_does_not_fire(repo: Path) -> None:
    text = PORTABLE + "# SHELL := is set in the prologue; $(shell) is banned\n"
    assert check_makefile(repo, repo / "libs" / "Makefile", text) == []


def test_a_violation_renders_relative_to_the_root(repo: Path) -> None:
    violation = Violation(
        path=repo / "libs" / "platform_core" / "Makefile", line_number=4, rule="rule", text="text"
    )
    assert render_violation(violation, repo) == "libs/platform_core/Makefile:4: rule: text"


def test_lint_grammar_reads_every_tracked_makefile(repo: Path, world: World) -> None:
    (repo / "libs" / "Makefile").write_text(PORTABLE, encoding="utf-8")
    (repo / "libs" / "platform_core" / "Makefile").write_text(PORTABLE, encoding="utf-8")
    world.tracked = [Path("libs/Makefile"), Path("libs/platform_core/Makefile")]
    examined, violations = lint_grammar(repo)
    assert examined == 2
    assert [(v["path"].relative_to(repo).as_posix(), v["rule"]) for v in violations] == [
        ("libs/platform_core/Makefile", "first line must be the shell prologue")
    ]
