"""Tests for Rust source-quality rules: file iteration, tests, cargo lints.

The serialization, coverage, proptest, and explicit-match rules are
pinned in ``test_rust_project_rules.py``.
"""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.config import GuardConfig
from monorepo_guards.rust_rules import (
    RustCargoLintRule,
    RustTestRule,
    iter_rs_files,
)


def _write(path: Path, text: str) -> None:
    """Write text to file, creating parent directories.

    Args:
        path: Path to file.
        text: Content to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_config(root: Path) -> GuardConfig:
    """Create standard GuardConfig for Rust tests.

    Args:
        root: Project root directory.

    Returns:
        GuardConfig with standard settings.
    """
    return GuardConfig(
        root=root,
        monorepo_root=root,
        directories=("src",),
        exclude_parts=(".venv", "__pycache__", "target"),
        forbid_pyi=False,
        allow_print_in_tests=False,
        dataclass_ban_segments=(),
    )


class TestIterRsFiles:
    """Tests for iter_rs_files helper function."""

    def test_returns_empty_when_no_src_dir(self, tmp_path: Path) -> None:
        """Test that empty list is returned when src/ doesn't exist."""
        config = _make_config(tmp_path)
        result = iter_rs_files(config)
        assert result == []

    def test_finds_rs_files_in_src(self, tmp_path: Path) -> None:
        """Test that .rs files in src/ are found."""
        _write(tmp_path / "src" / "lib.rs", "fn main() {}")
        _write(tmp_path / "src" / "utils.rs", "fn helper() {}")
        config = _make_config(tmp_path)

        result = iter_rs_files(config)

        assert len(result) == 2
        names = {p.name for p in result}
        assert names == {"lib.rs", "utils.rs"}

    def test_excludes_target_directory(self, tmp_path: Path) -> None:
        """Test that files in excluded directories are skipped."""
        _write(tmp_path / "src" / "lib.rs", "fn main() {}")
        _write(tmp_path / "src" / "target" / "debug.rs", "fn debug() {}")
        config = _make_config(tmp_path)

        result = iter_rs_files(config)

        assert len(result) == 1
        assert result[0].name == "lib.rs"

    def test_finds_nested_rs_files(self, tmp_path: Path) -> None:
        """Test that nested .rs files are found."""
        _write(tmp_path / "src" / "lib.rs", "mod utils;")
        _write(tmp_path / "src" / "utils" / "mod.rs", "fn util() {}")
        _write(tmp_path / "src" / "utils" / "helpers.rs", "fn help() {}")
        config = _make_config(tmp_path)

        result = iter_rs_files(config)

        assert len(result) == 3


class TestRustTestRule:
    """Tests for RustTestRule."""

    def test_no_cargo_toml_returns_empty(self, tmp_path: Path) -> None:
        """Test that rule returns empty when no Cargo.toml exists."""
        _write(tmp_path / "src" / "lib.rs", "#[test]\nfn test_foo() {}")
        config = _make_config(tmp_path)
        rule = RustTestRule(config)

        violations = rule.run([])

        assert violations == []

    def test_flags_test_without_result_return(self, tmp_path: Path) -> None:
        """Test that test functions without Result return are flagged."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        _write(
            tmp_path / "src" / "lib.rs",
            "#[test]\nfn test_something() {\n    assert!(true);\n}\n",
        )
        config = _make_config(tmp_path)
        rule = RustTestRule(config)

        violations = rule.run([])

        assert len(violations) == 1
        assert violations[0].kind == "rust-test-no-result"
        assert "test_something" in violations[0].line

    def test_allows_test_with_result_return(self, tmp_path: Path) -> None:
        """Test that test functions with Result return pass."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        _write(
            tmp_path / "src" / "lib.rs",
            "#[test]\nfn test_ok() -> Result<(), Error> {\n    Ok(())\n}\n",
        )
        config = _make_config(tmp_path)
        rule = RustTestRule(config)

        violations = rule.run([])

        no_result = [v for v in violations if v.kind == "rust-test-no-result"]
        assert no_result == []

    def test_flags_unwrap_in_test_module(self, tmp_path: Path) -> None:
        """Test that .unwrap() in #[cfg(test)] module is flagged."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        code = """
fn main() {}

#[cfg(test)]
mod tests {
    #[test]
    fn test_it() -> Result<(), ()> {
        let x = Some(1).unwrap();
        Ok(())
    }
}
"""
        _write(tmp_path / "src" / "lib.rs", code)
        config = _make_config(tmp_path)
        rule = RustTestRule(config)

        violations = rule.run([])

        unwrap_violations = [v for v in violations if v.kind == "rust-test-unwrap"]
        assert len(unwrap_violations) == 1
        assert ".unwrap()" in unwrap_violations[0].line

    def test_flags_expect_in_test_module(self, tmp_path: Path) -> None:
        """Test that .expect() in #[cfg(test)] module is flagged."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        code = """
#[cfg(test)]
mod tests {
    #[test]
    fn test_it() -> Result<(), ()> {
        let x = Some(1).expect("msg");
        Ok(())
    }
}
"""
        _write(tmp_path / "src" / "lib.rs", code)
        config = _make_config(tmp_path)
        rule = RustTestRule(config)

        violations = rule.run([])

        expect_violations = [v for v in violations if v.kind == "rust-test-unwrap"]
        assert len(expect_violations) == 1
        assert ".expect(" in expect_violations[0].line

    def test_handles_tokio_test_attribute(self, tmp_path: Path) -> None:
        """Test that #[tokio::test] is recognized as test."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        _write(
            tmp_path / "src" / "lib.rs",
            "#[tokio::test]\nasync fn test_async() {\n    assert!(true);\n}\n",
        )
        config = _make_config(tmp_path)
        rule = RustTestRule(config)

        violations = rule.run([])

        assert len(violations) == 1
        assert violations[0].kind == "rust-test-no-result"

    def test_handles_pub_fn_test(self, tmp_path: Path) -> None:
        """Test that pub fn test functions are detected."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        _write(
            tmp_path / "src" / "lib.rs",
            "#[test]\npub fn test_public() {\n    assert!(true);\n}\n",
        )
        config = _make_config(tmp_path)
        rule = RustTestRule(config)

        violations = rule.run([])

        assert len(violations) == 1
        assert "test_public" in violations[0].line

    def test_ignores_non_fn_after_test_attr(self, tmp_path: Path) -> None:
        """Test that non-function lines after #[test] reset state."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        code = """
#[test]
// some comment
struct NotAFunction;
"""
        _write(tmp_path / "src" / "lib.rs", code)
        config = _make_config(tmp_path)
        rule = RustTestRule(config)

        violations = rule.run([])

        assert violations == []

    def test_handles_multiple_attributes_before_fn(self, tmp_path: Path) -> None:
        """Test that additional attributes after #[test] are handled."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        code = """
#[test]
#[should_panic]
fn test_with_extra_attr() {
    panic!("expected");
}
"""
        _write(tmp_path / "src" / "lib.rs", code)
        config = _make_config(tmp_path)
        rule = RustTestRule(config)

        violations = rule.run([])

        assert len(violations) == 1
        assert violations[0].kind == "rust-test-no-result"
        assert "test_with_extra_attr" in violations[0].line


class TestRustCargoLintRule:
    """Tests for RustCargoLintRule."""

    def test_no_cargo_toml_returns_empty(self, tmp_path: Path) -> None:
        """Test that rule returns empty when no Cargo.toml exists."""
        config = _make_config(tmp_path)
        rule = RustCargoLintRule(config)

        violations = rule.run([])

        assert violations == []

    def test_flags_missing_forbid_lints(self, tmp_path: Path) -> None:
        """Test that missing required forbid lints are flagged."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"\n')
        config = _make_config(tmp_path)
        rule = RustCargoLintRule(config)

        violations = rule.run([])

        kinds = {v.kind for v in violations}
        assert "rust-lint-not-forbid" in kinds
        lint_violations = [v for v in violations if v.kind == "rust-lint-not-forbid"]
        assert len(lint_violations) == 6  # All required lints missing

    def test_allows_proper_forbid_lints(self, tmp_path: Path) -> None:
        """Test that properly configured forbid lints pass."""
        cargo = """
[package]
name = "test"

[lints.clippy]
question_mark_used = "forbid"
unwrap_used = "forbid"
expect_used = "forbid"
panic = "forbid"
todo = "forbid"
unimplemented = "forbid"
"""
        _write(tmp_path / "Cargo.toml", cargo)
        config = _make_config(tmp_path)
        rule = RustCargoLintRule(config)

        violations = rule.run([])

        not_forbid = [v for v in violations if v.kind == "rust-lint-not-forbid"]
        assert not_forbid == []

    def test_flags_deny_instead_of_forbid(self, tmp_path: Path) -> None:
        """Test that deny instead of forbid is flagged."""
        cargo = """
[package]
name = "test"

[lints.clippy]
question_mark_used = "forbid"
unwrap_used = "forbid"
expect_used = "forbid"
panic = "forbid"
todo = "forbid"
unimplemented = "forbid"
some_other_lint = "deny"
"""
        _write(tmp_path / "Cargo.toml", cargo)
        config = _make_config(tmp_path)
        rule = RustCargoLintRule(config)

        violations = rule.run([])

        deny_violations = [v for v in violations if v.kind == "rust-lint-deny-not-forbid"]
        assert len(deny_violations) == 1
        assert "some_other_lint" in deny_violations[0].line

    def test_allows_deny_for_all_and_cargo(self, tmp_path: Path) -> None:
        """Test that deny is allowed for 'all' and 'cargo' lints."""
        cargo = """
[package]
name = "test"

[lints.clippy]
question_mark_used = "forbid"
unwrap_used = "forbid"
expect_used = "forbid"
panic = "forbid"
todo = "forbid"
unimplemented = "forbid"
all = "deny"
cargo = "deny"
"""
        _write(tmp_path / "Cargo.toml", cargo)
        config = _make_config(tmp_path)
        rule = RustCargoLintRule(config)

        violations = rule.run([])

        deny_violations = [v for v in violations if v.kind == "rust-lint-deny-not-forbid"]
        assert deny_violations == []


__all__ = [
    "TestIterRsFiles",
    "TestRustCargoLintRule",
    "TestRustTestRule",
]
