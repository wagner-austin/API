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


@pytest.mark.parametrize(
    ("line", "rule"),
    [
        ('Write-Host "done" -ForegroundColor Green', "cmdlet"),
        ("Set-Location services/x", "cmdlet"),
        ("Resolve-Path .", "cmdlet"),
        ("$$env:GIT_COMMIT = (git rev-parse HEAD)", "$env: variable"),
        ("powershell -NoProfile -File x", "powershell invocation"),
        ("powershell -File scripts/x.ps1", "PowerShell script"),
        ("if (-not $$?) { exit 1 }", "$? status variable"),
        ("if (Test-Path x) { y }", "PowerShell control flow"),
        ("$$args = @(1, 2)", "PowerShell array literal"),
        ('& "..\\..\\scripts\\run-tests.ps1"', "backslash path separator"),
        ("echo `n", "backtick"),
        ("npm run a && npm run b", "&& or || chaining"),
        ("echo $$HOME", "shell variable or substitution ($$)"),
        ("cmd 2>/dev/null", "null device"),
        ("[ -f x ]", "[ test"),
        ("if test -f x", "sh control flow"),
        ("for f in *", "sh control flow"),
        ("a; b", "; separator (one command per line)"),
        ("cmd > out.txt", "redirection or pipe"),
        ("cmd | tail", "redirection or pipe"),
        ("cd libs/platform_core", "cd (use make -C or a script)"),
    ],
)
def test_each_banned_form_fires_its_rule(tmp_path: Path, line: str, rule: str) -> None:
    assert rule in rules_of(tmp_path / "Makefile", line)


@pytest.mark.parametrize(
    "line",
    [
        "poetry run mypy src tests scripts",
        '@echo "=== ALL CHECKS PASSED ==="',
        "$(PYTHON) ../../tools/maketools/scripts/run.py test",
        "poetry run pytest -n auto -v --cov-branch --cov=src",
        "$(MAKE) -C libs/platform_core check",
        "docker compose --project-directory services/qr-api up -d --build",
        "@# a recipe comment",
        "",
    ],
)
def test_the_portable_forms_pass(tmp_path: Path, line: str) -> None:
    assert rules_of(tmp_path / "Makefile", line) == []


def test_a_line_continuation_backslash_is_not_a_path_separator(tmp_path: Path) -> None:
    assert rules_of(tmp_path / "Makefile", "poetry run ruff check src \\") == []


def test_a_backslash_escape_inside_a_docker_argument_is_not_a_path(tmp_path: Path) -> None:
    line = 'docker ps --format "table {{.Names}}\\t{{.Status}}"'
    assert rules_of(tmp_path / "Makefile", line) == []


FENCED = (
    PORTABLE
    + """
ifeq ($(OS),Windows_NT)
register:
\tpowershell -NoProfile -File scripts\\register.ps1
\t@Write-Host "done" -ForegroundColor Green
ifdef VERBOSE
\t@Write-Host "verbose" -ForegroundColor Green
endif
else
register:
\t@echo "register is Windows-only"
\t@exit 1
endif
"""
)


def test_the_windows_arm_of_the_platform_fence_may_use_powershell(repo: Path) -> None:
    assert check_makefile(repo, repo / "libs" / "Makefile", FENCED) == []


def test_the_else_arm_of_the_platform_fence_is_checked(repo: Path) -> None:
    text = FENCED.replace(
        '\t@echo "register is Windows-only"', '\t@Write-Host "nope" -ForegroundColor Red'
    )
    violations = check_makefile(repo, repo / "libs" / "Makefile", text)
    assert [v["rule"] for v in violations] == ["cmdlet", "cmdlet parameter"]
    assert violations[0]["text"] == 'Write-Host "nope" -ForegroundColor Red'


def test_an_ifneq_fence_exempts_its_else_arm_instead(repo: Path) -> None:
    text = PORTABLE + (
        "ifneq ($(OS),Windows_NT)\nx:\n\t@echo portable\nelse\nx:\n\t@Write-Host win\nendif\n"
    )
    assert check_makefile(repo, repo / "libs" / "Makefile", text) == []
    swapped = PORTABLE + (
        "ifneq ($(OS),Windows_NT)\nx:\n\t@Write-Host win\nelse\nx:\n\t@echo portable\nendif\n"
    )
    rules = [v["rule"] for v in check_makefile(repo, repo / "libs" / "Makefile", swapped)]
    assert rules == ["cmdlet"]


def test_a_conditional_that_is_not_the_platform_fence_exempts_nothing(repo: Path) -> None:
    text = PORTABLE + "ifdef FAST\nx:\n\t@Write-Host fast\nelse\nx:\n\t@Write-Host slow\nendif\n"
    rules = [v["rule"] for v in check_makefile(repo, repo / "libs" / "Makefile", text)]
    assert rules == ["cmdlet", "cmdlet"]


def test_the_shell_function_is_allowed_only_inside_the_windows_arm(repo: Path) -> None:
    text = PORTABLE + "ifeq ($(OS),Windows_NT)\nX := $(shell dir)\nelse\nX := $(shell ls)\nendif\n"
    (violation,) = check_makefile(repo, repo / "libs" / "Makefile", text)
    assert violation["text"] == "X := $(shell ls)"


def test_the_shell_variable_stays_banned_inside_the_windows_arm(repo: Path) -> None:
    text = PORTABLE + "ifeq ($(OS),Windows_NT)\nSHELL := pwsh\nendif\n"
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
    text = PORTABLE + "SHELL := powershell.exe\n.SHELLFLAGS := -Command\n"
    rules = [v["rule"] for v in check_makefile(repo, repo / "libs" / "Makefile", text)]
    assert rules == [
        "SHELL is set only in scripts/make/shell.mk",
        ".SHELLFLAGS is set only in scripts/make/shell.mk",
    ]


def test_the_shell_function_fires_anywhere_in_the_file(repo: Path) -> None:
    text = PORTABLE + "FILE := $(shell ls)\n"
    (violation,) = check_makefile(repo, repo / "libs" / "Makefile", text)
    assert violation["rule"] == "$(shell ...) runs the platform shell"
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
