"""What the rules read from outside the package under check.

THIS IS THE ANSWER TO A QUESTION SOMEBODY ELSE ASKS. A caller assembling a
partial tree -- the fleet dispatcher sending one project to another machine --
needs to know which outside paths to carry, and deriving that list a second
time would drift towards carrying too little. The tests here pin the answer
against the SAME constants the rules use, so a new registered literal set or a
fourth category directory is reflected without anybody editing two places.
"""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.external_inputs import (
    CATEGORY_DIRECTORIES,
    GUARD_CONFIG_NAME,
    declaring_modules,
    external_inputs,
    package_manifests,
)
from monorepo_guards.literal_set_rules import REGISTERED_SETS


def _package(root: Path, category: str, name: str) -> Path:
    """Create a package directory with a manifest.

    Args:
        root: The synthetic monorepo root.
        category: One of the category directories.
        name: The package's directory name.

    Returns:
        The package directory.
    """
    directory = root / category / name
    (directory / "src").mkdir(parents=True)
    (directory / "pyproject.toml").write_text('[tool.poetry]\nname = "x"\n', encoding="utf-8")
    return directory


class TestPackageManifests:
    def test_every_category_is_scanned(self, tmp_path: Path) -> None:
        for index, category in enumerate(CATEGORY_DIRECTORIES):
            _package(tmp_path, category, f"pkg{index}")

        found = package_manifests(tmp_path)

        assert len(found) == len(CATEGORY_DIRECTORIES)

    def test_an_absent_category_is_skipped(self, tmp_path: Path) -> None:
        _package(tmp_path, CATEGORY_DIRECTORIES[0], "only")

        assert package_manifests(tmp_path) == (
            tmp_path / CATEGORY_DIRECTORIES[0] / "only" / "pyproject.toml",
        )

    def test_a_package_without_a_manifest_is_skipped(self, tmp_path: Path) -> None:
        (tmp_path / CATEGORY_DIRECTORIES[0] / "bare").mkdir(parents=True)

        assert package_manifests(tmp_path) == ()

    def test_a_file_where_a_package_should_be_is_skipped(self, tmp_path: Path) -> None:
        (tmp_path / CATEGORY_DIRECTORIES[0]).mkdir(parents=True)
        (tmp_path / CATEGORY_DIRECTORIES[0] / "stray.txt").write_text("x", encoding="utf-8")

        assert package_manifests(tmp_path) == ()


class TestDeclaringModules:
    def test_each_registered_set_is_found_where_the_rule_looks(self, tmp_path: Path) -> None:
        source = tmp_path / "services" / "thing" / "src"
        for declared in REGISTERED_SETS:
            module = source / declared.defining_module
            module.parent.mkdir(parents=True, exist_ok=True)
            module.write_text("X = ()\n", encoding="utf-8")

        found = declaring_modules(tmp_path)

        assert len(found) == len(REGISTERED_SETS)

    def test_a_set_whose_module_is_absent_contributes_nothing(self, tmp_path: Path) -> None:
        (tmp_path / "services" / "thing" / "src").mkdir(parents=True)

        assert declaring_modules(tmp_path) == ()

    def test_an_empty_monorepo_yields_nothing(self, tmp_path: Path) -> None:
        assert declaring_modules(tmp_path) == ()


class TestExternalInputs:
    def test_the_guard_config_is_included(self, tmp_path: Path) -> None:
        (tmp_path / GUARD_CONFIG_NAME).write_text("[guards]\n", encoding="utf-8")

        assert tmp_path / GUARD_CONFIG_NAME in external_inputs(tmp_path)

    def test_an_absent_guard_config_is_not_invented(self, tmp_path: Path) -> None:
        assert external_inputs(tmp_path) == ()

    def test_manifests_and_declaring_modules_are_both_carried(self, tmp_path: Path) -> None:
        (tmp_path / GUARD_CONFIG_NAME).write_text("[guards]\n", encoding="utf-8")
        package = _package(tmp_path, CATEGORY_DIRECTORIES[0], "thing")
        module = package / "src" / REGISTERED_SETS[0].defining_module
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text("X = ()\n", encoding="utf-8")

        found = external_inputs(tmp_path)

        assert package / "pyproject.toml" in found
        assert module in found

    def test_the_result_is_sorted_and_deduplicated(self, tmp_path: Path) -> None:
        (tmp_path / GUARD_CONFIG_NAME).write_text("[guards]\n", encoding="utf-8")
        for index, category in enumerate(CATEGORY_DIRECTORIES):
            _package(tmp_path, category, f"pkg{index}")

        found = external_inputs(tmp_path)

        assert list(found) == sorted(found)
        assert len(set(found)) == len(found)


class TestAgainstTheRealRepository:
    def repo_root(self) -> Path:
        """Locate the monorepo this package lives in.

        Returns:
            Its absolute path.
        """
        return Path(__file__).resolve().parents[3]

    def test_all_three_registered_sets_resolve_here(self) -> None:
        """If one stops resolving, the rule that owns it is already failing
        with `<subject>-declaration-unresolved` -- this says so first."""
        assert len(declaring_modules(self.repo_root())) == len(REGISTERED_SETS)

    def test_this_repository_has_a_guard_config(self) -> None:
        assert self.repo_root() / GUARD_CONFIG_NAME in external_inputs(self.repo_root())

    def test_it_finds_this_package_s_own_manifest(self) -> None:
        found = package_manifests(self.repo_root())

        assert self.repo_root() / "libs" / "monorepo_guards" / "pyproject.toml" in found
