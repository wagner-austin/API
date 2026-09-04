"""What a dispatch has to carry, read from the manifests that declare it.

THESE TESTS USE THE REAL MONOREPO WHERE THEY CAN. ``tools/fleet`` genuinely
declares ``platform-core`` at ``../../libs/platform_core``, and the assertion
that the walk finds it is worth more than the same assertion against a manifest
this file wrote: a fixture built from my own understanding of poetry's syntax
would agree with the code for the same wrong reason if that understanding were
off. The synthetic trees below cover the shapes the real repo does not have --
a diamond, a group dependency, an escape.
"""

from __future__ import annotations

import pathlib

import pytest
from monorepo_guards.external_inputs import declaring_modules, external_inputs
from monorepo_guards.literal_set_rules import REGISTERED_SETS
from platform_core.errors import AppError, FleetErrorCode

from fleet.core import manifest


def write_manifest(root: pathlib.Path, project: str, body: str) -> None:
    """Write a minimal poetry manifest for a synthetic project.

    Args:
        root: The synthetic monorepo root.
        project: Repo-relative project path.
        body: TOML to place under ``[tool.poetry]``, already indented as it
            should appear.
    """
    directory = root / project
    directory.mkdir(parents=True, exist_ok=True)
    (directory / manifest.MANIFEST_NAME).write_text(body, encoding="utf-8")


def shared(root: pathlib.Path) -> None:
    """Create everything every dispatch carries, in the right kind.

    Args:
        root: The synthetic monorepo root.
    """
    for directory in manifest.SHARED_DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)
    for name in manifest.SHARED_FILES:
        (root / name).write_text("# present\n", encoding="utf-8")


class TestAgainstTheRealRepository:
    """The walk, run against manifests nobody wrote for a test."""

    def repo_root(self) -> pathlib.Path:
        """Locate the monorepo this package lives in.

        Returns:
            Its absolute path.
        """
        return pathlib.Path(__file__).resolve().parents[3]

    def test_fleet_carries_the_library_its_lockfile_resolves_against(self) -> None:
        members = manifest.build_tree(self.repo_root(), "tools/fleet")
        assert "libs/platform_core" in members

    def test_the_project_comes_first(self) -> None:
        members = manifest.build_tree(self.repo_root(), "tools/fleet")
        assert members[0] == "tools/fleet"

    def test_every_shared_path_is_carried(self) -> None:
        members = manifest.build_tree(self.repo_root(), "tools/fleet")
        for path in manifest.SHARED_PATHS:
            assert path in members

    def test_the_shared_paths_are_really_there_and_the_right_kind(self) -> None:
        # SHARED_DIRECTORIES and SHARED_FILES quote assertions the Makefiles,
        # the guard shim and the guard config loader make about this
        # repository's layout. If a rename ever makes one wrong, this is where
        # it is caught -- not on a node.
        for directory in manifest.SHARED_DIRECTORIES:
            assert (self.repo_root() / directory).is_dir()
        for name in manifest.SHARED_FILES:
            assert (self.repo_root() / name).is_file()

    def test_the_guard_config_the_rules_load_is_carried(self) -> None:
        # The rules raise FileNotFoundError without it, which is how the
        # second real dispatch failed after both directories were staged.
        loader = (
            self.repo_root() / "libs/monorepo_guards/src/monorepo_guards/config_loader.py"
        ).read_text(encoding="utf-8")
        assert '"monorepo-guards.toml"' in loader
        assert "monorepo-guards.toml" in manifest.SHARED_FILES

    def test_the_guard_shim_still_imports_from_the_path_that_is_carried(self) -> None:
        # scripts/guard.py hard-codes parents[3] / "libs" / "monorepo_guards".
        # That literal is why the directory is in SHARED_PATHS, so the tie
        # between the two is asserted rather than assumed.
        shim = (self.repo_root() / "tools/fleet/scripts/guard.py").read_text(encoding="utf-8")
        assert '"libs" / "monorepo_guards"' in shim

    def test_the_launcher_the_makefile_calls_is_carried(self) -> None:
        recipe = (self.repo_root() / "tools/fleet/Makefile").read_text(encoding="utf-8")
        assert "scripts\\\\run-tests.ps1" in recipe
        assert "scripts" in manifest.build_tree(self.repo_root(), "tools/fleet")


class TestGuardInputs:
    """What ``make check`` reads from outside the project being built."""

    def repo_root(self) -> pathlib.Path:
        """Locate the monorepo this package lives in.

        Returns:
            Its absolute path.
        """
        return pathlib.Path(__file__).resolve().parents[3]

    def test_every_declaring_module_the_rules_look_for_is_carried(self) -> None:
        """THE REGRESSION, measured on sedona 2026-09-04. A dispatch that
        carried the project, its dependencies and both shared directories got
        through poetry sync and then failed on corpus-format-, risk-tier- and
        strategy-name-declaration-unresolved, because the modules declaring
        those sets had not travelled with it.

        DERIVED FROM THE REGISTRY, NOT NAMED. The first version of this test
        listed the three paths, and one of them moved within the hour --
        RISK_TIERS went from covenant_domain/features.py to
        platform_core/risk_tiers.py while this was being written. A test that
        spells the paths is the same second copy the code exists to avoid,
        and it drifts the same way.
        """
        root = self.repo_root()
        members = manifest.build_tree(root, "tools/fleet")

        wanted = [
            module.resolve().relative_to(root.resolve()).as_posix()
            for module in declaring_modules(root)
        ]
        assert len(wanted) == len(REGISTERED_SETS)
        for path in wanted:
            assert any(path == member or path.startswith(f"{member}/") for member in members), (
                f"{path} is neither staged nor inside anything staged"
            )

    def test_the_list_comes_from_the_rules_and_not_from_here(self) -> None:
        """Asked of monorepo_guards rather than restated. A second copy would
        drift towards naming too little, and too little surfaces on a remote
        node as three guard failures that read as the project's fault."""
        root = self.repo_root()
        asked = external_inputs(root)

        assert set(manifest.guard_inputs(root)) == {
            path.resolve().relative_to(root.resolve()).as_posix() for path in asked
        }

    def test_package_manifests_are_carried_as_files_not_packages(self) -> None:
        """A dispatch of tools/fleet must not drag an ML service with it."""
        members = manifest.build_tree(self.repo_root(), "tools/fleet")

        assert "services/Model-Trainer/pyproject.toml" in members
        assert "services/Model-Trainer" not in members

    def test_a_manifest_inside_a_staged_package_is_not_named_twice(self) -> None:
        """platform_core is staged whole, so its pyproject is already there
        and naming it again would put the file in the archive twice."""
        members = manifest.build_tree(self.repo_root(), "tools/fleet")

        assert "libs/platform_core" in members
        assert "libs/platform_core/pyproject.toml" not in members


class TestDeclaredExternalPaths:
    """What a project's own SUITE reads from outside itself."""

    def test_a_declared_path_is_carried(self, tmp_path: pathlib.Path) -> None:
        """THE hpc3 CASE, measured on lavender 2026-09-04. Four of its tests
        failed on a staged tree because its suite reads docs/RESEARCH.md at
        the monorepo root. The guards' outside reads can be discovered by
        asking monorepo_guards; a project's tests can only be declared."""
        shared(tmp_path)
        write_manifest(tmp_path, "apps/a", "[tool.poetry.dependencies]\n")
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "RESEARCH.md").write_text("# index\n", encoding="utf-8")

        members = manifest.build_tree(tmp_path, "apps/a", external=("docs",))

        assert "docs" in members

    def test_a_declared_file_is_carried(self, tmp_path: pathlib.Path) -> None:
        shared(tmp_path)
        write_manifest(tmp_path, "apps/a", "[tool.poetry.dependencies]\n")
        (tmp_path / "VERSION").write_text("1\n", encoding="utf-8")

        assert "VERSION" in manifest.build_tree(tmp_path, "apps/a", external=("VERSION",))

    def test_a_declared_path_already_inside_the_tree_is_not_named_twice(
        self, tmp_path: pathlib.Path
    ) -> None:
        shared(tmp_path)
        write_manifest(tmp_path, "apps/a", "[tool.poetry.dependencies]\n")

        members = manifest.build_tree(tmp_path, "apps/a", external=("apps/a",))

        assert members.count("apps/a") == 1

    def test_a_shared_file_inside_a_staged_member_is_not_named_twice(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Staging the monorepo root carries every shared file already, and
        naming one again would put it in the archive twice."""
        shared(tmp_path)
        write_manifest(tmp_path, "apps/a", "[tool.poetry.dependencies]\n")
        write_manifest(
            tmp_path, "apps/b", '[tool.poetry.dependencies]\nwhole = { path = "../.." }\n'
        )
        write_manifest(tmp_path, manifest.ROOT_MEMBER, "[tool.poetry.dependencies]\n")

        members = manifest.build_tree(tmp_path, "apps/b")

        assert members == ("apps/b", manifest.ROOT_MEMBER)

    def test_a_declared_path_that_is_gone_is_refused(self, tmp_path: pathlib.Path) -> None:
        """tar would say `Cannot stat`, which names a path and not a
        declaration -- and the reader's fix is to edit the workspace."""
        shared(tmp_path)
        write_manifest(tmp_path, "apps/a", "[tool.poetry.dependencies]\n")

        with pytest.raises(AppError) as refusal:
            manifest.build_tree(tmp_path, "apps/a", external=("docs",))

        assert refusal.value.code is FleetErrorCode.PROJECT_MANIFEST_MISSING
        assert "external_paths" in refusal.value.message

    def test_a_declared_path_outside_the_root_is_refused(self, tmp_path: pathlib.Path) -> None:
        shared(tmp_path)
        write_manifest(tmp_path, "apps/a", "[tool.poetry.dependencies]\n")

        with pytest.raises(AppError) as refusal:
            manifest.build_tree(tmp_path, "apps/a", external=("../elsewhere",))

        assert refusal.value.code is FleetErrorCode.PROJECT_DEPENDENCY_ESCAPES_ROOT

    def test_declaring_nothing_changes_nothing(self, tmp_path: pathlib.Path) -> None:
        shared(tmp_path)
        write_manifest(tmp_path, "apps/a", "[tool.poetry.dependencies]\n")

        assert manifest.build_tree(tmp_path, "apps/a", external=()) == manifest.build_tree(
            tmp_path, "apps/a"
        )


class TestWalkingSyntheticTrees:
    """The shapes the real repository does not happen to contain."""

    def test_a_diamond_is_carried_once(self, tmp_path: pathlib.Path) -> None:
        shared(tmp_path)
        write_manifest(
            tmp_path,
            "apps/top",
            '[tool.poetry.dependencies]\nleft = { path = "../left" }\n'
            'right = { path = "../right" }\n',
        )
        write_manifest(
            tmp_path, "apps/left", '[tool.poetry.dependencies]\nbase = { path = "../base" }\n'
        )
        write_manifest(
            tmp_path, "apps/right", '[tool.poetry.dependencies]\nbase = { path = "../base" }\n'
        )
        write_manifest(tmp_path, "apps/base", "[tool.poetry.dependencies]\n")

        members = manifest.build_tree(tmp_path, "apps/top")

        assert members.count("apps/base") == 1
        assert set(members) == {
            "apps/top",
            "apps/left",
            "apps/right",
            "apps/base",
            *manifest.SHARED_PATHS,
        }

    def test_a_cycle_terminates(self, tmp_path: pathlib.Path) -> None:
        shared(tmp_path)
        write_manifest(tmp_path, "apps/a", '[tool.poetry.dependencies]\nb = { path = "../b" }\n')
        write_manifest(tmp_path, "apps/b", '[tool.poetry.dependencies]\na = { path = "../a" }\n')

        members = manifest.build_tree(tmp_path, "apps/a")

        assert members == ("apps/a", "apps/b", *manifest.SHARED_PATHS)

    def test_a_dev_group_dependency_is_carried(self, tmp_path: pathlib.Path) -> None:
        # make check runs `poetry sync --with dev`, so a dev-group path
        # dependency is as required for a build as a runtime one.
        shared(tmp_path)
        write_manifest(
            tmp_path,
            "apps/a",
            '[tool.poetry.dependencies]\npytest = "^9.1.1"\n'
            '\n[tool.poetry.group.dev.dependencies]\nharness = { path = "../harness" }\n',
        )
        write_manifest(tmp_path, "apps/harness", "[tool.poetry.dependencies]\n")

        assert "apps/harness" in manifest.build_tree(tmp_path, "apps/a")

    def test_a_version_constraint_is_not_a_path(self, tmp_path: pathlib.Path) -> None:
        shared(tmp_path)
        write_manifest(
            tmp_path, "apps/a", '[tool.poetry.dependencies]\npython = "^3.11"\nrich = "^13"\n'
        )

        assert manifest.build_tree(tmp_path, "apps/a") == ("apps/a", *manifest.SHARED_PATHS)

    def test_a_table_without_a_path_is_not_a_path(self, tmp_path: pathlib.Path) -> None:
        shared(tmp_path)
        write_manifest(
            tmp_path,
            "apps/a",
            '[tool.poetry.dependencies]\ntorch = { version = "^2.5", source = "pytorch" }\n',
        )

        assert manifest.build_tree(tmp_path, "apps/a") == ("apps/a", *manifest.SHARED_PATHS)

    def test_a_path_that_is_not_a_string_is_not_a_path(self, tmp_path: pathlib.Path) -> None:
        # poetry would reject this manifest long before a dispatch read it,
        # and re-validating poetry is not this module's job.
        shared(tmp_path)
        write_manifest(tmp_path, "apps/a", "[tool.poetry.dependencies]\nweird = { path = 7 }\n")

        assert manifest.build_tree(tmp_path, "apps/a") == ("apps/a", *manifest.SHARED_PATHS)

    def test_a_manifest_with_no_poetry_table_carries_only_itself(
        self, tmp_path: pathlib.Path
    ) -> None:
        shared(tmp_path)
        write_manifest(tmp_path, "apps/a", '[project]\nname = "a"\n')

        assert manifest.build_tree(tmp_path, "apps/a") == ("apps/a", *manifest.SHARED_PATHS)

    def test_a_dependency_that_is_also_a_shared_path_is_carried_once(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Four packages here declare monorepo_guards as a real dependency
        while every package reaches it through the guard shim's hard-coded
        path, so the two ways of needing it genuinely overlap."""
        shared(tmp_path)
        guards = "libs/monorepo_guards"
        assert guards in manifest.SHARED_DIRECTORIES
        write_manifest(tmp_path, guards, "[tool.poetry.dependencies]\n")
        write_manifest(
            tmp_path,
            "libs/a",
            f'[tool.poetry.dependencies]\nguards = {{ path = "../{guards.rsplit("/", 1)[-1]}" }}\n',
        )

        members = manifest.build_tree(tmp_path, "libs/a")

        assert members.count(guards) == 1
        assert members == ("libs/a", guards, "scripts", *manifest.SHARED_FILES)

    def test_a_dependency_directly_at_the_root_is_allowed(self, tmp_path: pathlib.Path) -> None:
        # The root itself is inside the root, which `resolved.parents` alone
        # would say it is not.
        shared(tmp_path)
        write_manifest(
            tmp_path, "apps/a", '[tool.poetry.dependencies]\nwhole = { path = "../.." }\n'
        )
        write_manifest(tmp_path, ".", "[tool.poetry.dependencies]\n")

        # Nothing shared is named after it: the root member already carries
        # every one of them, and naming them again would duplicate the files.
        assert manifest.build_tree(tmp_path, "apps/a") == ("apps/a", manifest.ROOT_MEMBER)


class TestRefusals:
    """Everything that stops a dispatch rather than staging a broken tree."""

    def test_a_project_without_a_manifest_is_refused(self, tmp_path: pathlib.Path) -> None:
        shared(tmp_path)
        (tmp_path / "apps/a").mkdir(parents=True)

        with pytest.raises(AppError) as refusal:
            manifest.build_tree(tmp_path, "apps/a")

        assert refusal.value.code is FleetErrorCode.PROJECT_MANIFEST_MISSING
        assert "pyproject.toml" in refusal.value.message

    def test_a_dependency_without_a_manifest_is_refused(self, tmp_path: pathlib.Path) -> None:
        shared(tmp_path)
        write_manifest(tmp_path, "apps/a", '[tool.poetry.dependencies]\nb = { path = "../b" }\n')
        (tmp_path / "apps/b").mkdir(parents=True)

        with pytest.raises(AppError) as refusal:
            manifest.build_tree(tmp_path, "apps/a")

        assert refusal.value.code is FleetErrorCode.PROJECT_MANIFEST_MISSING
        assert "apps/b" in refusal.value.message

    def test_a_dependency_outside_the_root_is_refused(self, tmp_path: pathlib.Path) -> None:
        shared(tmp_path)
        write_manifest(
            tmp_path,
            "apps/a",
            '[tool.poetry.dependencies]\nelsewhere = { path = "../../../elsewhere" }\n',
        )

        with pytest.raises(AppError) as refusal:
            manifest.build_tree(tmp_path, "apps/a")

        assert refusal.value.code is FleetErrorCode.PROJECT_DEPENDENCY_ESCAPES_ROOT
        assert "elsewhere" in refusal.value.message

    def test_a_missing_shared_directory_is_refused(self, tmp_path: pathlib.Path) -> None:
        # No `shared(tmp_path)` here: this is what a rename of scripts/ or
        # libs/monorepo_guards would look like from inside a dispatch.
        write_manifest(tmp_path, "apps/a", "[tool.poetry.dependencies]\n")

        with pytest.raises(AppError) as refusal:
            manifest.build_tree(tmp_path, "apps/a")

        assert refusal.value.code is FleetErrorCode.PROJECT_MANIFEST_MISSING
        assert manifest.SHARED_DIRECTORIES[0] in refusal.value.message
        assert "directory" in refusal.value.message

    def test_a_missing_shared_file_is_refused(self, tmp_path: pathlib.Path) -> None:
        """The second real dispatch's failure, as a regression. Both shared
        directories were staged and monorepo-guards.toml was not, so the
        rules arrived without the file naming which of them to run."""
        shared(tmp_path)
        for name in manifest.SHARED_FILES:
            (tmp_path / name).unlink()
        write_manifest(tmp_path, "apps/a", "[tool.poetry.dependencies]\n")

        with pytest.raises(AppError) as refusal:
            manifest.build_tree(tmp_path, "apps/a")

        assert refusal.value.code is FleetErrorCode.PROJECT_MANIFEST_MISSING
        assert manifest.SHARED_FILES[0] in refusal.value.message
        assert "file" in refusal.value.message

    def test_a_shared_file_that_became_a_directory_is_refused(self, tmp_path: pathlib.Path) -> None:
        """The kinds are checked separately so a rename that turns one into
        the other cannot stage a tree that will not build."""
        shared(tmp_path)
        for name in manifest.SHARED_FILES:
            (tmp_path / name).unlink()
            (tmp_path / name).mkdir()
        write_manifest(tmp_path, "apps/a", "[tool.poetry.dependencies]\n")

        with pytest.raises(AppError) as refusal:
            manifest.build_tree(tmp_path, "apps/a")

        assert refusal.value.code is FleetErrorCode.PROJECT_MANIFEST_MISSING

    def test_a_dependency_table_that_is_not_a_table_is_refused(
        self, tmp_path: pathlib.Path
    ) -> None:
        shared(tmp_path)
        write_manifest(tmp_path, "apps/a", '[tool.poetry]\ndependencies = "everything"\n')

        with pytest.raises(AppError) as refusal:
            manifest.build_tree(tmp_path, "apps/a")

        assert refusal.value.code is FleetErrorCode.PROJECT_MANIFEST_UNREADABLE
        assert "'dependencies'" in refusal.value.message

    def test_a_group_that_is_not_a_table_is_refused(self, tmp_path: pathlib.Path) -> None:
        shared(tmp_path)
        write_manifest(tmp_path, "apps/a", '[tool.poetry.group]\ndev = "yes please"\n')

        with pytest.raises(AppError) as refusal:
            manifest.build_tree(tmp_path, "apps/a")

        assert refusal.value.code is FleetErrorCode.PROJECT_MANIFEST_UNREADABLE
        assert "'dev'" in refusal.value.message

    def test_a_tool_table_that_is_not_a_table_is_refused(self, tmp_path: pathlib.Path) -> None:
        shared(tmp_path)
        write_manifest(tmp_path, "apps/a", 'tool = "not a table"\n')

        with pytest.raises(AppError) as refusal:
            manifest.build_tree(tmp_path, "apps/a")

        assert refusal.value.code is FleetErrorCode.PROJECT_MANIFEST_UNREADABLE
        assert "'tool'" in refusal.value.message


class TestPathDependencies:
    """The single-project read the walk is built on."""

    def test_it_reports_only_direct_dependencies(self, tmp_path: pathlib.Path) -> None:
        write_manifest(tmp_path, "apps/a", '[tool.poetry.dependencies]\nb = { path = "../b" }\n')
        write_manifest(tmp_path, "apps/b", '[tool.poetry.dependencies]\nc = { path = "../c" }\n')
        write_manifest(tmp_path, "apps/c", "[tool.poetry.dependencies]\n")

        assert manifest.path_dependencies(tmp_path, "apps/a") == ("apps/b",)
