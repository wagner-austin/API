from __future__ import annotations

from pathlib import Path

from monorepo_guards import Violation
from monorepo_guards.toml_reader import (
    check_banned_api,
    extract_mypy_bool,
    extract_mypy_files,
    extract_ruff_src,
    read_pyproject,
)


class ConfigRule:
    """Validates pyproject.toml configurations for strict typing enforcement."""

    name = "config"

    def _get_expected_dirs(self, repo_root: Path) -> set[str]:
        """Determine which directories should be checked based on what exists."""
        expected = set()
        for dirname in ("src", "scripts", "tests"):
            if (repo_root / dirname).is_dir():
                expected.add(dirname)
        return expected

    def _check_mypy_files(
        self,
        repo_root: Path,
        toml_content: str,
        expected_dirs: set[str],
    ) -> list[Violation]:
        """Check that mypy 'files' includes all expected directories."""
        violations: list[Violation] = []

        files_list = extract_mypy_files(toml_content)
        if files_list is None:
            return violations

        configured_dirs = set(files_list)
        missing_dirs = expected_dirs - configured_dirs

        if missing_dirs:
            missing_str = ", ".join(sorted(missing_dirs))
            has_str = ", ".join(sorted(configured_dirs))
            violations.append(
                Violation(
                    file=repo_root / "pyproject.toml",
                    line_no=0,
                    kind="mypy-files-missing-dirs",
                    line=f"mypy files missing: [{missing_str}], has: [{has_str}]",
                )
            )

        return violations

    def _check_ruff_src(
        self,
        repo_root: Path,
        toml_content: str,
        expected_dirs: set[str],
    ) -> list[Violation]:
        """Check that ruff 'src' includes all expected directories."""
        violations: list[Violation] = []

        src_list = extract_ruff_src(toml_content)
        if src_list is None:
            return violations

        configured_dirs = set(src_list)
        missing_dirs = expected_dirs - configured_dirs

        if missing_dirs:
            missing_str = ", ".join(sorted(missing_dirs))
            has_str = ", ".join(sorted(configured_dirs))
            violations.append(
                Violation(
                    file=repo_root / "pyproject.toml",
                    line_no=0,
                    kind="ruff-src-missing-dirs",
                    line=f"ruff src missing: [{missing_str}], has: [{has_str}]",
                )
            )

        return violations

    def _check_mypy_strict_flags(
        self,
        repo_root: Path,
        toml_content: str,
    ) -> list[Violation]:
        """Check that critical mypy strict flags are enabled."""
        violations: list[Violation] = []

        strict_value = extract_mypy_bool(toml_content, "strict")
        if strict_value is not True:
            violations.append(
                Violation(
                    file=repo_root / "pyproject.toml",
                    line_no=0,
                    kind="mypy-strict-disabled",
                    line=f"Strict mode must be enabled (currently: {strict_value})",
                )
            )

        disallow_any_expr = extract_mypy_bool(toml_content, "disallow_any_expr")
        if disallow_any_expr is not True:
            violations.append(
                Violation(
                    file=repo_root / "pyproject.toml",
                    line_no=0,
                    kind="mypy-disallow-any-expr-disabled",
                    line=f"disallow_any_expr must be True (currently: {disallow_any_expr})",
                )
            )

        disallow_any_explicit = extract_mypy_bool(toml_content, "disallow_any_explicit")
        if disallow_any_explicit is not True:
            violations.append(
                Violation(
                    file=repo_root / "pyproject.toml",
                    line_no=0,
                    kind="mypy-disallow-any-explicit-disabled",
                    line=f"disallow_any_explicit must be True (currently: {disallow_any_explicit})",
                )
            )

        disallow_any_unimported = extract_mypy_bool(toml_content, "disallow_any_unimported")
        if disallow_any_unimported is not True:
            msg = f"disallow_any_unimported must be True (currently: {disallow_any_unimported})"
            violations.append(
                Violation(
                    file=repo_root / "pyproject.toml",
                    line_no=0,
                    kind="mypy-disallow-any-unimported-disabled",
                    line=msg,
                )
            )

        return violations

    def _check_ruff_banned_api(
        self,
        repo_root: Path,
        toml_content: str,
    ) -> list[Violation]:
        """Check that ruff has banned API rules for typing.Any and typing.cast."""
        violations: list[Violation] = []

        if not check_banned_api(toml_content, "typing.Any"):
            violations.append(
                Violation(
                    file=repo_root / "pyproject.toml",
                    line_no=0,
                    kind="ruff-missing-ban-typing-any",
                    line="ruff must ban typing.Any via flake8-tidy-imports",
                )
            )

        if not check_banned_api(toml_content, "typing.cast"):
            violations.append(
                Violation(
                    file=repo_root / "pyproject.toml",
                    line_no=0,
                    kind="ruff-missing-ban-typing-cast",
                    line="ruff must ban typing.cast via flake8-tidy-imports",
                )
            )

        return violations

    def _check_pyproject(self, pyproject_path: Path) -> list[Violation]:
        """Check a single pyproject.toml file."""
        violations: list[Violation] = []
        repo_root = pyproject_path.parent

        toml_content = read_pyproject(pyproject_path)
        expected_dirs = self._get_expected_dirs(repo_root)

        if expected_dirs:
            violations.extend(self._check_mypy_files(repo_root, toml_content, expected_dirs))
            violations.extend(self._check_ruff_src(repo_root, toml_content, expected_dirs))
            violations.extend(self._check_mypy_strict_flags(repo_root, toml_content))
            violations.extend(self._check_ruff_banned_api(repo_root, toml_content))

        return violations

    def _find_pyprojects_from_files(self, files: list[Path]) -> set[Path]:
        """Find pyproject.toml files by traversing up from provided files."""
        checked: set[Path] = set()

        for file_path in files:
            current = file_path.parent
            while current.parent != current:
                pyproject = current / "pyproject.toml"
                if pyproject.exists() and pyproject not in checked:
                    checked.add(pyproject)
                    break
                current = current.parent

        return checked

    def _find_monorepo_root(self, files: list[Path]) -> Path | None:
        """Find the monorepo root by looking for services/clients/libs structure."""
        if not files:
            return None

        search_root = files[0].resolve()
        while search_root.parent != search_root:
            has_monorepo = any(
                (search_root / dirname).is_dir() for dirname in ("services", "clients", "libs")
            )
            if has_monorepo:
                return search_root
            search_root = search_root.parent

        return None

    def _scan_monorepo_pyprojects(self, monorepo_root: Path) -> set[Path]:
        """Scan all subdirectories in monorepo for pyproject.toml files."""
        found: set[Path] = set()

        for category in ("services", "clients", "libs"):
            category_path = monorepo_root / category
            if category_path.is_dir():
                for repo_dir in category_path.iterdir():
                    if repo_dir.is_dir():
                        pyproject = repo_dir / "pyproject.toml"
                        if pyproject.exists():
                            found.add(pyproject)

        return found

    def run(self, files: list[Path]) -> list[Violation]:
        """Find and validate all pyproject.toml files in the monorepo."""
        violations: list[Violation] = []

        checked = self._find_pyprojects_from_files(files)

        monorepo_root = self._find_monorepo_root(files)
        if monorepo_root is not None:
            checked.update(self._scan_monorepo_pyprojects(monorepo_root))

        for pyproject in checked:
            violations.extend(self._check_pyproject(pyproject))

        return violations


__all__ = ["ConfigRule"]
