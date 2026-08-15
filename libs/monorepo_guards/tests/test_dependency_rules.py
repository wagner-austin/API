"""Tests for the escaping-path-dependency rule."""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.config import GuardConfig
from monorepo_guards.dependency_rules import (
    KIND_ESCAPING_PATH,
    KIND_UNROOTED,
    EscapingPathDependencyRule,
)
from monorepo_guards.toml_reader import extract_path_dependencies
from monorepo_guards.util import CONFIG_FILENAME, find_monorepo_root


def _config(root: Path) -> GuardConfig:
    """Build a configuration rooted at one project."""
    return GuardConfig(
        root=root,
        directories=("src",),
        exclude_parts=(),
        forbid_pyi=True,
        allow_print_in_tests=False,
        dataclass_ban_segments=(),
    )


def _monorepo(root: Path) -> Path:
    """Create a monorepo root, marked by the guard config file."""
    root.mkdir(parents=True, exist_ok=True)
    (root / CONFIG_FILENAME).write_text("[guards]\n", encoding="utf-8")
    return root


def _project(monorepo: Path, name: str, pyproject: str) -> Path:
    """Create a project inside the monorepo with the given pyproject."""
    root = monorepo / "services" / name
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    return root


def test_sibling_path_dependency_inside_the_monorepo_is_allowed(tmp_path: Path) -> None:
    monorepo = _monorepo(tmp_path)
    (monorepo / "libs" / "shared").mkdir(parents=True)
    project = _project(
        monorepo,
        "api",
        '[tool.poetry.dependencies]\nshared = { path = "../../libs/shared", develop = true }\n',
    )

    assert EscapingPathDependencyRule(_config(project)).run([]) == []


def test_path_dependency_climbing_out_of_the_monorepo_is_a_violation(tmp_path: Path) -> None:
    monorepo = _monorepo(tmp_path / "repo")
    (tmp_path / "outside-library").mkdir(parents=True)
    project = _project(
        monorepo,
        "api",
        "[tool.poetry.dependencies]\n"
        'outside-library = { path = "../../../outside-library", develop = true }\n',
    )

    violations = EscapingPathDependencyRule(_config(project)).run([])

    assert len(violations) == 1
    assert violations[0].kind == KIND_ESCAPING_PATH
    assert violations[0].line_no == 2
    assert "outside-library" in violations[0].line
    assert str(monorepo) in violations[0].line


def test_a_dev_group_cannot_hide_an_escaping_dependency(tmp_path: Path) -> None:
    monorepo = _monorepo(tmp_path / "repo")
    (tmp_path / "elsewhere").mkdir(parents=True)
    project = _project(
        monorepo,
        "api",
        "[tool.poetry.dependencies]\npython = '^3.11'\n\n"
        "[tool.poetry.group.dev.dependencies]\n"
        'elsewhere = { path = "../../../elsewhere", develop = true }\n',
    )

    violations = EscapingPathDependencyRule(_config(project)).run([])

    assert [v.kind for v in violations] == [KIND_ESCAPING_PATH]
    assert violations[0].line_no == 5


def test_every_escaping_dependency_is_reported(tmp_path: Path) -> None:
    monorepo = _monorepo(tmp_path / "repo")
    project = _project(
        monorepo,
        "api",
        "[tool.poetry.dependencies]\n"
        'alpha = { path = "../../../alpha" }\n'
        'beta = { path = "../../../beta" }\n',
    )

    violations = EscapingPathDependencyRule(_config(project)).run([])

    assert [v.line_no for v in violations] == [2, 3]
    assert all(v.kind == KIND_ESCAPING_PATH for v in violations)


def test_version_dependencies_are_not_path_dependencies(tmp_path: Path) -> None:
    monorepo = _monorepo(tmp_path)
    project = _project(
        monorepo,
        "api",
        '[tool.poetry.dependencies]\npython = "^3.11"\nfastapi = "^0.124"\n',
    )

    assert EscapingPathDependencyRule(_config(project)).run([]) == []


def test_a_project_with_no_pyproject_declares_nothing(tmp_path: Path) -> None:
    monorepo = _monorepo(tmp_path)
    root = monorepo / "services" / "api"
    root.mkdir(parents=True)

    assert EscapingPathDependencyRule(_config(root)).run([]) == []


def test_a_project_outside_any_guarded_monorepo_is_a_violation(tmp_path: Path) -> None:
    root = tmp_path / "loose"
    root.mkdir()
    (root / "pyproject.toml").write_text("[tool.poetry.dependencies]\n", encoding="utf-8")

    violations = EscapingPathDependencyRule(_config(root)).run([])

    assert [v.kind for v in violations] == [KIND_UNROOTED]


def test_rule_name_is_stable() -> None:
    assert EscapingPathDependencyRule(_config(Path("."))).name == "dependency-escape"


def test_extractor_reports_name_path_and_line() -> None:
    content = '[tool.poetry.dependencies]\nshared = { path = "../shared", develop = true }\n'

    assert extract_path_dependencies(content) == [("shared", "../shared", 2)]


def test_extractor_accepts_a_quoted_dependency_name() -> None:
    content = '[tool.poetry.dependencies]\n"odd.name" = { path = "../x" }\n'

    assert extract_path_dependencies(content) == [("odd.name", "../x", 2)]


def test_extractor_ignores_version_dependencies() -> None:
    content = '[tool.poetry.dependencies]\npython = "^3.11"\n'

    assert extract_path_dependencies(content) == []


def test_extractor_ignores_paths_outside_a_dependency_table() -> None:
    """A ``path`` key in some other section is not a dependency."""
    content = '[tool.something-else]\nshared = { path = "../shared" }\n'

    assert extract_path_dependencies(content) == []

    leaving = (
        '[tool.poetry.dependencies]\nshared = { path = "../shared" }\n'
        '[tool.other]\nescapee = { path = "../../../elsewhere" }\n'
    )
    assert [d.name for d in extract_path_dependencies(leaving)] == ["shared"]


def test_extractor_reads_every_dependency_group() -> None:
    content = (
        '[tool.poetry.dependencies]\na = { path = "../a" }\n'
        '[tool.poetry.group.dev.dependencies]\nb = { path = "../b" }\n'
        '[tool.poetry.group.docs.dependencies]\nc = { path = "../c" }\n'
    )

    assert [d.name for d in extract_path_dependencies(content)] == ["a", "b", "c"]


def test_extractor_ignores_an_inline_table_without_a_path() -> None:
    content = '[tool.poetry.dependencies]\nextras = { version = "^1.0", optional = true }\n'

    assert extract_path_dependencies(content) == []


def test_find_monorepo_root_returns_none_at_the_filesystem_root(tmp_path: Path) -> None:
    assert find_monorepo_root(tmp_path) is None


def test_find_monorepo_root_finds_the_directory_holding_the_config(tmp_path: Path) -> None:
    (tmp_path / CONFIG_FILENAME).write_text("[guards]\n", encoding="utf-8")
    nested = tmp_path / "services" / "api"
    nested.mkdir(parents=True)

    assert find_monorepo_root(nested) == tmp_path.resolve()
