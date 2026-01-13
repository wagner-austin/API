"""Rust code quality rules.

Enforces strict patterns in Rust code:
- All #[test] functions must return Result<(), ...>
- No .unwrap() or .expect() in test code
- No ? operator (explicit match required for full coverage)
- No derive(Serialize, Deserialize) (manual impl required)
- 100% region coverage enforcement in Makefile
- proptest in dev-dependencies for property-based testing
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from monorepo_guards import Violation
from monorepo_guards.config import GuardConfig


def iter_rs_files(config: GuardConfig) -> list[Path]:
    """Iterate over all .rs files in the project.

    Args:
        config: Guard configuration with root and exclude patterns.

    Returns:
        List of paths to Rust source files.
    """
    src_dir = config.root / "src"
    if not src_dir.exists():
        return []

    out: list[Path] = []
    for path in src_dir.rglob("*.rs"):
        if any(part in config.exclude_parts for part in path.parts):
            continue
        out.append(path)
    return out


def _read_lines(path: Path) -> list[str]:
    """Read file lines with UTF-8 encoding.

    Args:
        path: Path to the file.

    Returns:
        List of lines.
    """
    text = path.read_text(encoding="utf-8", errors="strict")
    return text.splitlines()


def _read_file(path: Path) -> str:
    """Read entire file with UTF-8 encoding.

    Args:
        path: Path to the file.

    Returns:
        File contents as string.
    """
    return path.read_text(encoding="utf-8", errors="strict")


class RustTestRule:
    """Rule enforcing Rust test function patterns.

    Checks:
    - All #[test] functions must return Result<(), ...>
    - No .unwrap() calls in test functions

    Args:
        config: Guard configuration with project root.
    """

    name = "rust-test"

    _TEST_ATTR_PATTERN: re.Pattern[str] = re.compile(r"#\[(test|tokio::test)\]")
    _FN_PATTERN: re.Pattern[str] = re.compile(r"^\s*(pub\s+)?(async\s+)?fn\s+(\w+)")
    _RESULT_RETURN_PATTERN: re.Pattern[str] = re.compile(r"->\s*Result\s*<")
    _UNWRAP_PATTERN: re.Pattern[str] = re.compile(r"\.unwrap\(\)")
    _EXPECT_PATTERN: re.Pattern[str] = re.compile(r"\.expect\(")

    def __init__(self, config: GuardConfig) -> None:
        """Initialize RustTestRule with configuration.

        Args:
            config: Guard configuration with project root.
        """
        self._config = config

    def _find_test_functions(self, lines: list[str]) -> list[tuple[int, str, bool]]:
        """Find test functions and check if they return Result.

        Args:
            lines: File content as lines.

        Returns:
            List of (line_no, fn_name, has_result_return) tuples.
        """
        tests: list[tuple[int, str, bool]] = []
        in_test_attr = False

        for i, line in enumerate(lines):
            line_no = i + 1

            if self._TEST_ATTR_PATTERN.search(line):
                in_test_attr = True
                continue

            if in_test_attr:
                fn_match = self._FN_PATTERN.match(line)
                if fn_match:
                    fn_name = fn_match.group(3)
                    has_result = bool(self._RESULT_RETURN_PATTERN.search(line))
                    tests.append((line_no, fn_name, has_result))
                    in_test_attr = False
                elif line.strip() and not line.strip().startswith("#"):
                    in_test_attr = False

        return tests

    def _find_unwrap_in_tests(self, lines: list[str]) -> list[tuple[int, str]]:
        """Find .unwrap() and .expect() calls in test module.

        Args:
            lines: File content as lines.

        Returns:
            List of (line_no, line_content) tuples.
        """
        violations: list[tuple[int, str]] = []
        in_test_module = False

        for i, line in enumerate(lines):
            line_no = i + 1

            if "#[cfg(test)]" in line:
                in_test_module = True

            if in_test_module:
                if self._UNWRAP_PATTERN.search(line):
                    violations.append((line_no, line.strip()))
                if self._EXPECT_PATTERN.search(line):
                    violations.append((line_no, line.strip()))

        return violations

    def run(self, files: list[Path]) -> list[Violation]:
        """Run the Rust test rule.

        Args:
            files: List of Python files (ignored - uses config.root for Rust files).

        Returns:
            List of violations found.
        """
        violations: list[Violation] = []

        cargo_toml = self._config.root / "Cargo.toml"
        if not cargo_toml.exists():
            return violations

        rs_files = iter_rs_files(self._config)

        for path in rs_files:
            lines = _read_lines(path)

            tests = self._find_test_functions(lines)
            for line_no, fn_name, has_result in tests:
                if not has_result:
                    violations.append(
                        Violation(
                            file=path,
                            line_no=line_no,
                            kind="rust-test-no-result",
                            line=f"fn {fn_name}() must return Result<(), ...>",
                        )
                    )

            unwraps = self._find_unwrap_in_tests(lines)
            for line_no, line_content in unwraps:
                violations.append(
                    Violation(
                        file=path,
                        line_no=line_no,
                        kind="rust-test-unwrap",
                        line=line_content,
                    )
                )

        return violations


class RustCargoLintRule:
    """Rule enforcing strict Cargo.toml lint configuration.

    Verifies:
    - question_mark_used = "forbid" (no lazy error propagation)
    - unwrap_used = "forbid"
    - expect_used = "forbid"
    - All lints use "forbid" not "deny" or weaker

    Args:
        config: Guard configuration with project root.
    """

    name = "rust-cargo-lint"

    _REQUIRED_FORBID_LINTS: ClassVar[list[str]] = [
        "question_mark_used",
        "unwrap_used",
        "expect_used",
        "panic",
        "todo",
        "unimplemented",
    ]

    def __init__(self, config: GuardConfig) -> None:
        """Initialize RustCargoLintRule with configuration.

        Args:
            config: Guard configuration with project root.
        """
        self._config = config

    def run(self, files: list[Path]) -> list[Violation]:
        """Run the Cargo lint rule.

        Args:
            files: List of files (ignored - uses config.root).

        Returns:
            List of violations found.
        """
        violations: list[Violation] = []

        cargo_toml = self._config.root / "Cargo.toml"
        if not cargo_toml.exists():
            return violations

        content = _read_file(cargo_toml)

        for lint in self._REQUIRED_FORBID_LINTS:
            pattern = rf'{lint}\s*=\s*"forbid"'
            if not re.search(pattern, content):
                violations.append(
                    Violation(
                        file=cargo_toml,
                        line_no=1,
                        kind="rust-lint-not-forbid",
                        line=f'{lint} must be set to "forbid" in [lints.clippy]',
                    )
                )

        deny_pattern = re.compile(r'(\w+)\s*=\s*"deny"')
        for match in deny_pattern.finditer(content):
            lint_name = match.group(1)
            if lint_name not in ("all", "cargo"):
                violations.append(
                    Violation(
                        file=cargo_toml,
                        line_no=1,
                        kind="rust-lint-deny-not-forbid",
                        line=f'{lint_name} uses "deny" but should use "forbid"',
                    )
                )

        return violations


class RustManualSerializeRule:
    """Rule banning derive(Serialize, Deserialize) macros.

    Requires manual implementation of Serialize/Deserialize for full
    control and testability. Derive macros generate hidden code that
    cannot be directly tested or audited.

    Args:
        config: Guard configuration with project root.
    """

    name = "rust-manual-serialize"

    _DERIVE_SERDE_PATTERN: re.Pattern[str] = re.compile(
        r"#\[derive\([^)]*(?:Serialize|Deserialize)[^)]*\)\]"
    )

    def __init__(self, config: GuardConfig) -> None:
        """Initialize RustManualSerializeRule with configuration.

        Args:
            config: Guard configuration with project root.
        """
        self._config = config

    def run(self, files: list[Path]) -> list[Violation]:
        """Run the manual serialize rule.

        Args:
            files: List of files (ignored - uses config.root).

        Returns:
            List of violations found.
        """
        violations: list[Violation] = []

        cargo_toml = self._config.root / "Cargo.toml"
        if not cargo_toml.exists():
            return violations

        rs_files = iter_rs_files(self._config)

        for path in rs_files:
            lines = _read_lines(path)
            for i, line in enumerate(lines):
                line_no = i + 1
                if self._DERIVE_SERDE_PATTERN.search(line):
                    violations.append(
                        Violation(
                            file=path,
                            line_no=line_no,
                            kind="rust-derive-serde-banned",
                            line="derive(Serialize/Deserialize) banned; implement manually",
                        )
                    )

        return violations


class RustCoverageRule:
    """Rule enforcing 100% region coverage in Makefile.

    Verifies that the Makefile uses:
    - --fail-under-lines 100
    - --fail-under-regions 100

    Args:
        config: Guard configuration with project root.
    """

    name = "rust-coverage"

    def __init__(self, config: GuardConfig) -> None:
        """Initialize RustCoverageRule with configuration.

        Args:
            config: Guard configuration with project root.
        """
        self._config = config

    def run(self, files: list[Path]) -> list[Violation]:
        """Run the coverage rule.

        Args:
            files: List of files (ignored - uses config.root).

        Returns:
            List of violations found.
        """
        violations: list[Violation] = []

        makefile = self._config.root / "Makefile"
        if not makefile.exists():
            return violations

        cargo_toml = self._config.root / "Cargo.toml"
        if not cargo_toml.exists():
            return violations

        content = _read_file(makefile)

        if "--fail-under-lines 100" not in content:
            violations.append(
                Violation(
                    file=makefile,
                    line_no=1,
                    kind="rust-coverage-lines",
                    line="Makefile must include --fail-under-lines 100",
                )
            )

        if "--fail-under-regions 100" not in content:
            violations.append(
                Violation(
                    file=makefile,
                    line_no=1,
                    kind="rust-coverage-regions",
                    line="Makefile must include --fail-under-regions 100",
                )
            )

        return violations


class RustProptestRule:
    """Rule enforcing property-based testing with proptest.

    Verifies:
    - proptest is in dev-dependencies
    - At least one proptest test exists (proptest! macro or #[proptest])

    Args:
        config: Guard configuration with project root.
    """

    name = "rust-proptest"

    # Matches proptest! macro, #[proptest] attribute, or TestRunner API (explicit proptest usage)
    _PROPTEST_PATTERN: re.Pattern[str] = re.compile(
        r"(?:proptest!\s*\{|#\[proptest\]|proptest::test_runner::TestRunner)"
    )

    def __init__(self, config: GuardConfig) -> None:
        """Initialize RustProptestRule with configuration.

        Args:
            config: Guard configuration with project root.
        """
        self._config = config

    def run(self, files: list[Path]) -> list[Violation]:
        """Run the proptest rule.

        Args:
            files: List of files (ignored - uses config.root).

        Returns:
            List of violations found.
        """
        violations: list[Violation] = []

        cargo_toml = self._config.root / "Cargo.toml"
        if not cargo_toml.exists():
            return violations

        cargo_content = _read_file(cargo_toml)

        if "[dev-dependencies]" in cargo_content and "proptest" not in cargo_content:
            violations.append(
                Violation(
                    file=cargo_toml,
                    line_no=1,
                    kind="rust-proptest-missing-dep",
                    line="proptest must be in [dev-dependencies]",
                )
            )

        rs_files = iter_rs_files(self._config)
        has_proptest = False

        for path in rs_files:
            content = _read_file(path)
            if self._PROPTEST_PATTERN.search(content):
                has_proptest = True
                break

        if not has_proptest and rs_files:
            violations.append(
                Violation(
                    file=cargo_toml,
                    line_no=1,
                    kind="rust-proptest-no-tests",
                    line="No proptest tests found; add property-based tests",
                )
            )

        return violations


class RustExplicitMatchRule:
    """Rule banning the ? operator in Rust code.

    The ? operator creates hidden error propagation branches that
    are difficult to test. Explicit match statements ensure every
    error path is visible and testable.

    Args:
        config: Guard configuration with project root.
    """

    name = "rust-explicit-match"

    _QUESTION_MARK_PATTERN: re.Pattern[str] = re.compile(r"\?\s*[;,\)]")

    def __init__(self, config: GuardConfig) -> None:
        """Initialize RustExplicitMatchRule with configuration.

        Args:
            config: Guard configuration with project root.
        """
        self._config = config

    def run(self, files: list[Path]) -> list[Violation]:
        """Run the explicit match rule.

        Args:
            files: List of files (ignored - uses config.root).

        Returns:
            List of violations found.
        """
        violations: list[Violation] = []

        cargo_toml = self._config.root / "Cargo.toml"
        if not cargo_toml.exists():
            return violations

        rs_files = iter_rs_files(self._config)

        for path in rs_files:
            lines = _read_lines(path)
            for i, line in enumerate(lines):
                line_no = i + 1
                is_question_mark = "?" in line and self._QUESTION_MARK_PATTERN.search(line)
                is_comment = line.strip().startswith("//")
                if is_question_mark and not is_comment:
                    violations.append(
                        Violation(
                            file=path,
                            line_no=line_no,
                            kind="rust-question-mark-banned",
                            line="? operator banned; use explicit match",
                        )
                    )

        return violations


__all__ = [
    "RustCargoLintRule",
    "RustCoverageRule",
    "RustExplicitMatchRule",
    "RustManualSerializeRule",
    "RustProptestRule",
    "RustTestRule",
    "iter_rs_files",
]
