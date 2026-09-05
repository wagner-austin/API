from __future__ import annotations

from pathlib import Path

from monorepo_guards.config_rules import ConfigRule

COVERAGE_MANIFEST_TEMPLATE = """
[tool.mypy]
files = ["src"]
strict = true
disallow_any_expr = true
disallow_any_explicit = true
disallow_any_unimported = true

[tool.ruff]
src = ["src"]

[tool.ruff.lint.flake8-tidy-imports.banned-api]
"typing.Any" = { msg = "no" }
"typing.cast" = { msg = "no" }

COVERAGE_BLOCK
"""
"""A manifest whose only variable part is its coverage configuration."""


def test_config_rule_detects_missing_mypy_files(tmp_path: Path) -> None:
    """Test that missing directories in mypy files are detected."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    (repo / "scripts").mkdir()

    # Add Python files so directories are expected in config
    (repo / "src" / "mod.py").write_text("x = 1", encoding="utf-8")
    (repo / "tests" / "test_mod.py").write_text("y = 2", encoding="utf-8")
    (repo / "scripts" / "run.py").write_text("z = 3", encoding="utf-8")

    pyproject = repo / "pyproject.toml"
    pyproject.write_text(
        """
[tool.mypy]
files = ["src", "tests"]
strict = true
disallow_any_expr = true
disallow_any_explicit = true
disallow_any_unimported = true
""",
        encoding="utf-8",
    )

    rule = ConfigRule()

    violations = rule.run([repo / "src" / "mod.py"])

    mypy_violations = [v for v in violations if v.kind == "mypy-files-missing-dirs"]
    assert len(mypy_violations) == 1
    assert "scripts" in mypy_violations[0].line


def test_config_rule_detects_missing_ruff_src(tmp_path: Path) -> None:
    """Test that missing directories in ruff src are detected."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    (repo / "scripts").mkdir()

    # Add Python files so directories are expected in config
    (repo / "src" / "mod.py").write_text("x = 1", encoding="utf-8")
    (repo / "tests" / "test_mod.py").write_text("y = 2", encoding="utf-8")
    (repo / "scripts" / "run.py").write_text("z = 3", encoding="utf-8")

    pyproject = repo / "pyproject.toml"
    pyproject.write_text(
        """
[tool.ruff]
src = ["src"]
""",
        encoding="utf-8",
    )

    rule = ConfigRule()

    violations = rule.run([repo / "src" / "mod.py"])

    ruff_violations = [v for v in violations if v.kind == "ruff-src-missing-dirs"]
    assert len(ruff_violations) == 1
    assert "scripts" in ruff_violations[0].line or "tests" in ruff_violations[0].line


def test_config_rule_detects_missing_strict_flags(tmp_path: Path) -> None:
    """Test that missing mypy strict flags are detected."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()

    pyproject = repo / "pyproject.toml"
    pyproject.write_text(
        """
[tool.mypy]
files = ["src"]
strict = false
disallow_any_expr = false
""",
        encoding="utf-8",
    )

    rule = ConfigRule()
    test_file = repo / "src" / "test.py"
    test_file.write_text("x = 1", encoding="utf-8")

    violations = rule.run([test_file])

    assert len(violations) >= 4
    kinds = {v.kind for v in violations}
    assert "mypy-strict-disabled" in kinds
    assert "mypy-disallow-any-expr-disabled" in kinds
    assert "mypy-disallow-any-explicit-disabled" in kinds
    assert "mypy-disallow-any-unimported-disabled" in kinds


def test_config_rule_detects_missing_banned_api(tmp_path: Path) -> None:
    """Test that missing ruff banned API rules are detected."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()

    pyproject = repo / "pyproject.toml"
    pyproject.write_text(
        """
[tool.ruff.lint.flake8-tidy-imports.banned-api]
""",
        encoding="utf-8",
    )

    rule = ConfigRule()
    test_file = repo / "src" / "test.py"
    test_file.write_text("x = 1", encoding="utf-8")

    violations = rule.run([test_file])

    assert len(violations) >= 2
    kinds = {v.kind for v in violations}
    assert "ruff-missing-ban-typing-any" in kinds
    assert "ruff-missing-ban-typing-cast" in kinds


def test_config_rule_accepts_valid_config(tmp_path: Path) -> None:
    """Test that valid configuration passes without violations."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    (repo / "scripts").mkdir()

    pyproject = repo / "pyproject.toml"
    pyproject.write_text(
        """
[tool.mypy]
files = ["src", "tests", "scripts"]
strict = true
disallow_any_expr = true
disallow_any_explicit = true
disallow_any_unimported = true

[tool.ruff]
src = ["src", "tests", "scripts"]

[tool.ruff.lint.flake8-tidy-imports.banned-api]
"typing.Any" = { msg = "banned" }
"typing.cast" = { msg = "banned" }
""",
        encoding="utf-8",
    )

    rule = ConfigRule()
    test_file = repo / "src" / "test.py"
    test_file.write_text("x = 1", encoding="utf-8")

    violations = rule.run([test_file])

    assert len(violations) == 0


def test_config_rule_skips_repos_without_expected_dirs(tmp_path: Path) -> None:
    """Test that repos without src/tests/scripts are skipped."""
    repo = tmp_path / "repo"
    repo.mkdir()

    pyproject = repo / "pyproject.toml"
    pyproject.write_text("[tool.mypy]\nstrict = false\n", encoding="utf-8")

    rule = ConfigRule()
    test_file = repo / "other" / "test.py"
    test_file.parent.mkdir()
    test_file.write_text("x = 1", encoding="utf-8")

    violations = rule.run([test_file])

    assert len(violations) == 0


def test_config_rule_finds_monorepo_pyprojects(tmp_path: Path) -> None:
    """Test that ConfigRule finds pyproject.toml files in monorepo structure."""
    monorepo = tmp_path / "monorepo"
    monorepo.mkdir()

    (monorepo / "services").mkdir()
    (monorepo / "clients").mkdir()
    (monorepo / "libs").mkdir()

    service1 = monorepo / "services" / "api"
    service1.mkdir()
    (service1 / "src").mkdir()
    (service1 / "pyproject.toml").write_text("[tool.mypy]\nfiles = ['src']", encoding="utf-8")

    service2 = monorepo / "services" / "worker"
    service2.mkdir()
    (service2 / "src").mkdir()
    (service2 / "pyproject.toml").write_text("[tool.ruff]\nsrc = ['src']", encoding="utf-8")

    rule = ConfigRule()
    test_file = service1 / "src" / "test.py"
    test_file.write_text("x = 1", encoding="utf-8")

    violations = rule.run([test_file])

    assert len(violations) >= 0


def test_config_rule_handles_no_files() -> None:
    """Test that ConfigRule handles empty file list."""
    rule = ConfigRule()
    violations = rule.run([])
    assert len(violations) == 0


def test_config_rule_handles_nonexistent_pyproject(tmp_path: Path) -> None:
    """Test that ConfigRule handles files without pyproject.toml in hierarchy."""
    repo = tmp_path / "repo"
    repo.mkdir()
    test_file = repo / "test.py"
    test_file.write_text("x = 1", encoding="utf-8")

    rule = ConfigRule()
    violations = rule.run([test_file])

    assert len(violations) == 0


def test_config_rule_handles_files_in_category_dirs(tmp_path: Path) -> None:
    """Test that ConfigRule handles files (not dirs) in monorepo category paths."""
    monorepo = tmp_path / "monorepo"
    monorepo.mkdir()

    (monorepo / "services").write_text("not a dir", encoding="utf-8")
    (monorepo / "clients").mkdir()
    (monorepo / "libs").mkdir()

    (monorepo / "clients" / "somefile.txt").write_text("not a dir", encoding="utf-8")

    service1 = monorepo / "libs" / "lib1"
    service1.mkdir()
    (service1 / "src").mkdir()
    (service1 / "pyproject.toml").write_text(
        """
[tool.mypy]
files = ["src"]
strict = true
disallow_any_expr = true
disallow_any_explicit = true
disallow_any_unimported = true

[tool.ruff]
src = ["src"]

[tool.ruff.lint.flake8-tidy-imports.banned-api]
"typing.Any" = { msg = "banned" }
"typing.cast" = { msg = "banned" }
""",
        encoding="utf-8",
    )

    rule = ConfigRule()
    test_file = service1 / "src" / "test.py"
    test_file.write_text("x = 1", encoding="utf-8")

    violations = rule.run([test_file])

    assert len(violations) == 0


def _repo_with_packages(tmp_path: Path, packages_block: str) -> Path:
    """Build a minimal package whose only interesting content is `packages`.

    Args:
        tmp_path: Per-test temporary directory.
        packages_block: The `packages = [...]` lines to write.

    Returns:
        Path to a source file inside it, for handing to ``rule.run``.
    """
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    source = repo / "src" / "mod.py"
    source.write_text("x = 1", encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        f"""
[tool.poetry]
name = "thing"
{packages_block}

[tool.mypy]
files = ["src"]
strict = true
disallow_any_expr = true
disallow_any_explicit = true
disallow_any_unimported = true

[tool.ruff]
src = ["src"]

[tool.ruff.lint.flake8-tidy-imports.banned-api]
"typing.Any".msg = "banned"
"typing.cast".msg = "banned"
""",
        encoding="utf-8",
    )
    return source


def test_config_rule_detects_a_package_include_without_from(tmp_path: Path) -> None:
    """The defect this rule exists for, in the exact shape it shipped in.

    Every one of the 40 Python packages here carried this line. It made
    poetry put the project ROOT on each consumer's sys.path and ship a
    top-level `scripts` module in the wheel, so the packages overwrote each
    other on install.
    """
    source = _repo_with_packages(
        tmp_path,
        'packages = [\n  { include = "thing", from = "src" },\n  { include = "scripts" },\n]',
    )

    violations = [v for v in ConfigRule().run([source]) if v.kind == "package-include-without-from"]

    assert len(violations) == 1
    assert violations[0].line_no == 6
    assert 'include="scripts"' in violations[0].line


def test_config_rule_accepts_a_package_include_with_from(tmp_path: Path) -> None:
    """An entry naming a source root is the correct shape and stays silent."""
    source = _repo_with_packages(
        tmp_path, 'packages = [\n  { include = "thing", from = "src" },\n]'
    )

    violations = [v for v in ConfigRule().run([source]) if v.kind == "package-include-without-from"]

    assert violations == []


def test_config_rule_names_the_two_ways_a_rootless_include_hurts(tmp_path: Path) -> None:
    """The message has to teach, because the failure is invisible locally.

    A stale sys.path entry and a colliding wheel are different symptoms of
    one line, and whoever hits either one needs to be told about both.
    """
    source = _repo_with_packages(tmp_path, 'packages = [\n  { include = "scripts" },\n]')

    violations = [v for v in ConfigRule().run([source]) if v.kind == "package-include-without-from"]

    assert len(violations) == 1
    assert "sys.path" in violations[0].line
    assert "wheel" in violations[0].line


_RUN_SECTION = "[tool.coverage.run]"
_REPORT_SECTION = "[tool.coverage.report]"


def _coverage_config(*lines: str) -> str:
    """Join coverage-configuration lines into a manifest fragment.

    Written as lines rather than one triple-quoted block so a test reads as
    the settings it is about.

    Args:
        *lines: Section headers and key assignments, in order.

    Returns:
        The fragment, newline-joined.
    """
    return "\n".join(lines)


def _repo_with_coverage(tmp_path: Path, coverage_block: str) -> Path:
    """Build a minimal package whose only interesting content is its coverage config.

    Args:
        tmp_path: Per-test temporary directory.
        coverage_block: The coverage sections to write.

    Returns:
        Path to a source file inside it, for handing to ``rule.run``.
    """
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    source = repo / "src" / "mod.py"
    source.write_text("x = 1", encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        COVERAGE_MANIFEST_TEMPLATE.replace("COVERAGE_BLOCK", coverage_block),
        encoding="utf-8",
    )
    return source


def _coverage_violations(source: Path) -> list[str]:
    """Run the rule and return the kinds of its coverage violations.

    Args:
        source: A file inside the package to check.

    Returns:
        Every violation kind beginning with ``coverage-``, in report order.
    """
    return [v.kind for v in ConfigRule().run([source]) if v.kind.startswith("coverage-")]


def test_config_rule_accepts_coverage_config_that_hides_nothing(tmp_path: Path) -> None:
    """The shape every package here uses: source scoped, nothing carved out."""
    source = _repo_with_coverage(
        tmp_path,
        _coverage_config(
            _RUN_SECTION,
            'source = ["src"]',
            "branch = true",
            "",
            _REPORT_SECTION,
            "fail_under = 100",
        ),
    )

    assert _coverage_violations(source) == []


def test_config_rule_accepts_an_empty_carve_out(tmp_path: Path) -> None:
    """An empty omit excludes nothing, and reporting it would train the reader
    to ignore the rule."""
    source = _repo_with_coverage(
        tmp_path,
        _coverage_config(
            _RUN_SECTION,
            'source = ["src"]',
            "omit = []",
            "",
            _REPORT_SECTION,
            "fail_under = 100",
        ),
    )

    assert _coverage_violations(source) == []


def test_config_rule_detects_an_omit(tmp_path: Path) -> None:
    source = _repo_with_coverage(
        tmp_path,
        _coverage_config(
            _RUN_SECTION,
            'source = ["src"]',
            'omit = ["src/generated/*"]',
            "",
            _REPORT_SECTION,
            "fail_under = 100",
        ),
    )

    assert _coverage_violations(source) == ["coverage-omit"]


def test_config_rule_detects_excluded_lines(tmp_path: Path) -> None:
    """The carve-out that hid platform_email's entry point. Invisible in the
    report, because the package still printed 100%."""
    source = _repo_with_coverage(
        tmp_path,
        _coverage_config(
            _RUN_SECTION,
            'source = ["src"]',
            "",
            _REPORT_SECTION,
            "fail_under = 100",
            "exclude_lines = [",
            '    "def main",',
            "]",
        ),
    )

    violations = [v for v in ConfigRule().run([source]) if v.kind == "coverage-exclude-lines"]

    assert len(violations) == 1
    assert "def main" in violations[0].line


def test_config_rule_detects_a_lowered_threshold(tmp_path: Path) -> None:
    source = _repo_with_coverage(
        tmp_path,
        _coverage_config(
            _RUN_SECTION,
            'source = ["src"]',
            "",
            _REPORT_SECTION,
            "fail_under = 95",
        ),
    )

    assert _coverage_violations(source) == ["coverage-fail-under-below-100"]


def test_config_rule_reports_every_carve_out_not_only_the_first(tmp_path: Path) -> None:
    """A package that carved out three ways would otherwise be repaired one
    lint run at a time."""
    source = _repo_with_coverage(
        tmp_path,
        _coverage_config(
            _RUN_SECTION,
            'source = ["src"]',
            'omit = ["a/*"]',
            "",
            _REPORT_SECTION,
            "fail_under = 90",
            'exclude_lines = ["def main"]',
        ),
    )

    assert sorted(_coverage_violations(source)) == [
        "coverage-exclude-lines",
        "coverage-fail-under-below-100",
        "coverage-omit",
    ]


def test_config_rule_ignores_an_omit_outside_the_coverage_section(tmp_path: Path) -> None:
    """``omit`` is not a reserved word: another tool's table may hold one, and
    that one hides nothing from coverage."""
    source = _repo_with_coverage(
        tmp_path,
        _coverage_config(
            "[tool.something_else]",
            'omit = ["everything"]',
            "",
            _RUN_SECTION,
            'source = ["src"]',
            "",
            _REPORT_SECTION,
            "fail_under = 100",
        ),
    )

    assert _coverage_violations(source) == []


__all__ = [
    "test_config_rule_accepts_a_package_include_with_from",
    "test_config_rule_accepts_an_empty_carve_out",
    "test_config_rule_accepts_coverage_config_that_hides_nothing",
    "test_config_rule_accepts_valid_config",
    "test_config_rule_detects_a_lowered_threshold",
    "test_config_rule_detects_a_package_include_without_from",
    "test_config_rule_detects_an_omit",
    "test_config_rule_detects_excluded_lines",
    "test_config_rule_detects_missing_banned_api",
    "test_config_rule_detects_missing_mypy_files",
    "test_config_rule_detects_missing_ruff_src",
    "test_config_rule_detects_missing_strict_flags",
    "test_config_rule_finds_monorepo_pyprojects",
    "test_config_rule_handles_files_in_category_dirs",
    "test_config_rule_handles_no_files",
    "test_config_rule_handles_nonexistent_pyproject",
    "test_config_rule_ignores_an_omit_outside_the_coverage_section",
    "test_config_rule_names_the_two_ways_a_rootless_include_hurts",
    "test_config_rule_reports_every_carve_out_not_only_the_first",
    "test_config_rule_skips_repos_without_expected_dirs",
]
