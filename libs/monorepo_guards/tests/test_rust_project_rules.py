"""Tests for Rust project-shape rules: serialize, coverage, proptest, match.

The file-iteration, test, and cargo-lint rules are pinned in
``test_rust_source_rules.py``.
"""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.config import GuardConfig
from monorepo_guards.rust_rules import (
    RustCoverageRule,
    RustExplicitMatchRule,
    RustManualSerializeRule,
    RustProptestRule,
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
        directories=("src",),
        exclude_parts=(".venv", "__pycache__", "target"),
        forbid_pyi=False,
        allow_print_in_tests=False,
        dataclass_ban_segments=(),
    )


class TestRustManualSerializeRule:
    """Tests for RustManualSerializeRule."""

    def test_no_cargo_toml_returns_empty(self, tmp_path: Path) -> None:
        """Test that rule returns empty when no Cargo.toml exists."""
        _write(tmp_path / "src" / "lib.rs", "#[derive(Serialize)]\nstruct Foo;")
        config = _make_config(tmp_path)
        rule = RustManualSerializeRule(config)

        violations = rule.run([])

        assert violations == []

    def test_flags_derive_serialize(self, tmp_path: Path) -> None:
        """Test that derive(Serialize) is flagged."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        _write(tmp_path / "src" / "lib.rs", "#[derive(Serialize)]\nstruct Foo {\n    x: i32,\n}")
        config = _make_config(tmp_path)
        rule = RustManualSerializeRule(config)

        violations = rule.run([])

        assert len(violations) == 1
        assert violations[0].kind == "rust-derive-serde-banned"

    def test_flags_derive_deserialize(self, tmp_path: Path) -> None:
        """Test that derive(Deserialize) is flagged."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        _write(tmp_path / "src" / "lib.rs", "#[derive(Deserialize)]\nstruct Foo;")
        config = _make_config(tmp_path)
        rule = RustManualSerializeRule(config)

        violations = rule.run([])

        assert len(violations) == 1
        assert violations[0].kind == "rust-derive-serde-banned"

    def test_flags_derive_with_both(self, tmp_path: Path) -> None:
        """Test that derive with both Serialize and Deserialize is flagged."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        _write(
            tmp_path / "src" / "lib.rs",
            "#[derive(Clone, Serialize, Deserialize, Debug)]\nstruct Foo;",
        )
        config = _make_config(tmp_path)
        rule = RustManualSerializeRule(config)

        violations = rule.run([])

        assert len(violations) == 1

    def test_allows_other_derives(self, tmp_path: Path) -> None:
        """Test that other derive macros are allowed."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        _write(tmp_path / "src" / "lib.rs", "#[derive(Debug, Clone, PartialEq)]\nstruct Foo;")
        config = _make_config(tmp_path)
        rule = RustManualSerializeRule(config)

        violations = rule.run([])

        assert violations == []


class TestRustCoverageRule:
    """Tests for RustCoverageRule."""

    def test_no_makefile_returns_empty(self, tmp_path: Path) -> None:
        """Test that rule returns empty when no Makefile exists."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        config = _make_config(tmp_path)
        rule = RustCoverageRule(config)

        violations = rule.run([])

        assert violations == []

    def test_no_cargo_toml_returns_empty(self, tmp_path: Path) -> None:
        """Test that rule returns empty when no Cargo.toml exists."""
        _write(tmp_path / "Makefile", "test:\n\tcargo test")
        config = _make_config(tmp_path)
        rule = RustCoverageRule(config)

        violations = rule.run([])

        assert violations == []

    def test_flags_missing_coverage_enforcement(self, tmp_path: Path) -> None:
        """Test that missing coverage enforcement is flagged."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        _write(tmp_path / "Makefile", "test:\n\tcargo test")
        config = _make_config(tmp_path)
        rule = RustCoverageRule(config)

        violations = rule.run([])

        kinds = {v.kind for v in violations}
        assert "rust-coverage-missing" in kinds

    def test_allows_segment_coverage_config(self, tmp_path: Path) -> None:
        """Test that the segment-coverage binary invocation passes."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        makefile = """
test:
\tcargo test
\tcargo llvm-cov --all-features --json --output-path coverage.json
\tcargo run --bin check_segment_coverage -- --threshold 100
"""
        _write(tmp_path / "Makefile", makefile)
        config = _make_config(tmp_path)
        rule = RustCoverageRule(config)

        violations = rule.run([])

        assert violations == []

    def test_rejects_llvm_cov_line_and_region_gates(self, tmp_path: Path) -> None:
        """Test that llvm-cov's own line/region gates are not accepted.

        They report phantom misses for generic instantiations, so they cannot
        express the 100% requirement the segment checker enforces.
        """
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        makefile = """
test:
\tcargo test
\tcargo llvm-cov --fail-under-lines 100 --fail-under-regions 100
"""
        _write(tmp_path / "Makefile", makefile)
        config = _make_config(tmp_path)
        rule = RustCoverageRule(config)

        violations = rule.run([])

        kinds = {v.kind for v in violations}
        assert "rust-coverage-missing" in kinds

    def test_rejects_threshold_below_one_hundred(self, tmp_path: Path) -> None:
        """Test that a lowered threshold is flagged."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        _write(
            tmp_path / "Makefile",
            "test:\n\tcargo run --bin check_segment_coverage -- --threshold 95",
        )
        config = _make_config(tmp_path)
        rule = RustCoverageRule(config)

        violations = rule.run([])

        kinds = {v.kind for v in violations}
        assert "rust-coverage-missing" in kinds


class TestRustProptestRule:
    """Tests for RustProptestRule."""

    def test_no_cargo_toml_returns_empty(self, tmp_path: Path) -> None:
        """Test that rule returns empty when no Cargo.toml exists."""
        config = _make_config(tmp_path)
        rule = RustProptestRule(config)

        violations = rule.run([])

        assert violations == []

    def test_flags_missing_proptest_dep(self, tmp_path: Path) -> None:
        """Test that missing proptest in dev-dependencies is flagged."""
        cargo = """
[package]
name = "test"

[dev-dependencies]
tokio = "1.0"
"""
        _write(tmp_path / "Cargo.toml", cargo)
        _write(tmp_path / "src" / "lib.rs", "fn main() {}")
        config = _make_config(tmp_path)
        rule = RustProptestRule(config)

        violations = rule.run([])

        kinds = {v.kind for v in violations}
        assert "rust-proptest-missing-dep" in kinds

    def test_flags_no_proptest_tests(self, tmp_path: Path) -> None:
        """Test that missing proptest tests are flagged."""
        cargo = """
[package]
name = "test"

[dev-dependencies]
proptest = "1.0"
"""
        _write(tmp_path / "Cargo.toml", cargo)
        _write(tmp_path / "src" / "lib.rs", "fn main() {}")
        config = _make_config(tmp_path)
        rule = RustProptestRule(config)

        violations = rule.run([])

        kinds = {v.kind for v in violations}
        assert "rust-proptest-no-tests" in kinds

    def test_allows_proptest_macro_usage(self, tmp_path: Path) -> None:
        """Test that proptest! macro usage passes."""
        cargo = """
[package]
name = "test"

[dev-dependencies]
proptest = "1.0"
"""
        _write(tmp_path / "Cargo.toml", cargo)
        code = """
proptest! {
    fn test_prop(x in 0..100) {
        assert!(x < 100);
    }
}
"""
        _write(tmp_path / "src" / "lib.rs", code)
        config = _make_config(tmp_path)
        rule = RustProptestRule(config)

        violations = rule.run([])

        no_tests = [v for v in violations if v.kind == "rust-proptest-no-tests"]
        assert no_tests == []

    def test_allows_proptest_attribute_usage(self, tmp_path: Path) -> None:
        """Test that #[proptest] attribute usage passes."""
        cargo = """
[package]
name = "test"

[dev-dependencies]
proptest = "1.0"
"""
        _write(tmp_path / "Cargo.toml", cargo)
        code = """
#[proptest]
fn test_prop(x: u32) {
    assert!(x >= 0);
}
"""
        _write(tmp_path / "src" / "lib.rs", code)
        config = _make_config(tmp_path)
        rule = RustProptestRule(config)

        violations = rule.run([])

        no_tests = [v for v in violations if v.kind == "rust-proptest-no-tests"]
        assert no_tests == []

    def test_no_violation_when_no_rs_files(self, tmp_path: Path) -> None:
        """Test that no proptest-no-tests violation when no .rs files exist."""
        cargo = """
[package]
name = "test"

[dev-dependencies]
proptest = "1.0"
"""
        _write(tmp_path / "Cargo.toml", cargo)
        config = _make_config(tmp_path)
        rule = RustProptestRule(config)

        violations = rule.run([])

        no_tests = [v for v in violations if v.kind == "rust-proptest-no-tests"]
        assert no_tests == []


class TestRustExplicitMatchRule:
    """Tests for RustExplicitMatchRule."""

    def test_no_cargo_toml_returns_empty(self, tmp_path: Path) -> None:
        """Test that rule returns empty when no Cargo.toml exists."""
        _write(tmp_path / "src" / "lib.rs", "let x = foo()?;")
        config = _make_config(tmp_path)
        rule = RustExplicitMatchRule(config)

        violations = rule.run([])

        assert violations == []

    def test_flags_question_mark_before_semicolon(self, tmp_path: Path) -> None:
        """Test that ? before semicolon is flagged."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        code = "fn foo() -> Result<(), ()> {\n    bar()?;\n    Ok(())\n}"
        _write(tmp_path / "src" / "lib.rs", code)
        config = _make_config(tmp_path)
        rule = RustExplicitMatchRule(config)

        violations = rule.run([])

        assert len(violations) == 1
        assert violations[0].kind == "rust-question-mark-banned"

    def test_flags_question_mark_before_comma(self, tmp_path: Path) -> None:
        """Test that ? before comma is flagged."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        _write(tmp_path / "src" / "lib.rs", "fn foo() {\n    call(bar()?, baz());\n}")
        config = _make_config(tmp_path)
        rule = RustExplicitMatchRule(config)

        violations = rule.run([])

        assert len(violations) == 1
        assert violations[0].kind == "rust-question-mark-banned"

    def test_flags_question_mark_before_paren(self, tmp_path: Path) -> None:
        """Test that ? before closing paren is flagged."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        _write(tmp_path / "src" / "lib.rs", "fn foo() {\n    Ok(bar()?)\n}")
        config = _make_config(tmp_path)
        rule = RustExplicitMatchRule(config)

        violations = rule.run([])

        assert len(violations) == 1

    def test_ignores_question_mark_in_comments(self, tmp_path: Path) -> None:
        """Test that ? in comments is ignored."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        _write(tmp_path / "src" / "lib.rs", "// What about this?;\nfn foo() {}")
        config = _make_config(tmp_path)
        rule = RustExplicitMatchRule(config)

        violations = rule.run([])

        assert violations == []

    def test_allows_code_without_question_mark(self, tmp_path: Path) -> None:
        """Test that code without ? operator passes."""
        _write(tmp_path / "Cargo.toml", '[package]\nname = "test"')
        code = """
fn foo() -> Result<(), Error> {
    match bar() {
        Ok(v) => Ok(v),
        Err(e) => Err(e),
    }
}
"""
        _write(tmp_path / "src" / "lib.rs", code)
        config = _make_config(tmp_path)
        rule = RustExplicitMatchRule(config)

        violations = rule.run([])

        assert violations == []


__all__ = [
    "TestRustCoverageRule",
    "TestRustExplicitMatchRule",
    "TestRustManualSerializeRule",
    "TestRustProptestRule",
]
