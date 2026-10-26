"""A loopback-only project's check runs in a network of its own (MCPs board task 649fcc7a).

Chromium fails every request it has in flight, to localhost too, with
ERR_NETWORK_CHANGED whenever the host's addresses change, and the GitHub
Actions jobs' service containers on lavender change them all day: idle's
browser test files failed to load in fleet jobs dee131c3 and 1a8e13bc on
2026-10-25, each beside a container's bridge going down. A project that
declares ``loopback_only`` runs its install steps in the host's network and
its ``make check`` through ``fleet-check-unit loopback``
(:mod:`fleet.core.linux_loopback`), which MCPs' execution suite runs for real
on each Linux node (``execution/tests/check-unit-loopback.test.ts``).

These cases hold the declaration, the recipe, the rendered build and the
shipped registry. The one shell case runs the rendered command with the
helper's prefix taken off, under a real ``/bin/sh`` and ``make``, so what the
loopback check is handed is read from a process and not from the script's
text.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue

from fleet.contracts.check_unit import CHECK_UNIT
from fleet.contracts.project import decode_loopback_only, decode_project_config
from fleet.contracts.source import InstallStep
from fleet.contracts.workspace import decode_fleet_workspace
from fleet.core import dialect, names, recipe
from fleet.core.linux_isolated_build import EXEC_PATH
from fleet.core.linux_loopback import BUILD_EXPORTS, recipe_command
from tests._registry import shipped_document

#: The projects that declare ``loopback_only``, and why each check reads only lo.
LOOPBACK_REASONS = {
    "idle": (
        "its browser tests import their modules from vitest's Vite server on localhost:63315, "
        "and it needs gpu, which lavender alone of the enabled nodes that run checks carries, "
        "beside three GitHub Actions runners; its check reads nothing else since idle b5f133d"
    ),
}

#: What a declaration that is no boolean is refused with.
REFUSED = r"project\.loopback_only must be true, false or absent"

#: A run of idle on lavender, and the unit its launch starts it under.
RUN_ID = "idle-lavender-1792906110"
UNIT = names.task_name(RUN_ID)
TARGET = f"/home/corvis/fleet/stage/{RUN_ID}"

#: idle's install steps, as its registry row declares them.
INSTALL = (
    InstallStep(phase="install", argv=("npm", "ci")),
    InstallStep(phase="install", argv=("npx", "playwright", "install", "chromium")),
)


def _project(**extra: JSONValue) -> JSONObject:
    """A project row like idle's, with the keys given added to it."""
    row: JSONObject = {
        "worker_ram_gb": 0.5,
        "minimum_workers": 4,
        "expected_minutes": 15,
        "required_tags": ["gpu"],
        "source": None,
    }
    return {**row, **extra}


def _build(loopback_unit: str | None) -> str:
    """idle's Linux build for :data:`RUN_ID`.

    Args:
        loopback_unit: The unit for a loopback-only build, else None.

    Returns:
        The script's text.
    """
    return dialect.build_script(
        target=TARGET,
        path="",
        workers=4,
        install=INSTALL,
        cache_root="/home/corvis/fleet/stage/cache",
        isolated_docker=False,
        agent="opus-mcps-1024-a771",
        loopback_unit=loopback_unit,
    )


class TestTheDeclaration:
    def test_absent_reads_as_false_and_a_boolean_as_itself(self) -> None:
        assert decode_project_config(_project())["loopback_only"] is False
        assert decode_project_config(_project(loopback_only=True))["loopback_only"] is True
        assert decode_project_config(_project(loopback_only=False))["loopback_only"] is False

    @pytest.mark.parametrize("value", ["true", 1, ["lo"]])
    def test_any_other_value_is_refused_by_name(self, value: JSONValue) -> None:
        with pytest.raises(JSONTypeError, match=REFUSED):
            decode_loopback_only(value)
        with pytest.raises(JSONTypeError, match=REFUSED):
            decode_project_config(_project(loopback_only=value))

    def test_the_projects_declaring_it_are_exactly_those_with_a_reason(self) -> None:
        workspace = decode_fleet_workspace(shipped_document())
        declared = sorted(
            name for name, project in workspace["projects"].items() if project["loopback_only"]
        )
        assert declared == sorted(LOOPBACK_REASONS)


class TestTheRecipe:
    def test_it_carries_the_declaration_and_names_the_check_s_own_unit(self) -> None:
        plan = decode_project_config(_project(loopback_only=True))
        built = recipe.recipe_for(plan, path="", install=INSTALL)

        assert built["loopback_only"] is True
        assert recipe.loopback_unit(built, RUN_ID) == "fleet-idle-lavender-1792906110"
        assert recipe.working_tree_recipe("idle", plan)["loopback_only"] is True

    def test_a_project_that_does_not_declare_it_names_no_unit(self) -> None:
        built = recipe.recipe_for(decode_project_config(_project()), path="", install=())

        assert built["loopback_only"] is False
        assert recipe.loopback_unit(built, RUN_ID) is None


class TestTheBuild:
    def test_the_install_steps_run_in_the_host_s_network_and_the_check_through_the_helper(
        self,
    ) -> None:
        lines = _build(UNIT).splitlines()
        log = names.log_path(TARGET)
        passed = " ".join(f'{name}="${name}"' for name in ("PATH", *BUILD_EXPORTS))

        assert f"npm ci >> '{log}' 2>&1" in lines
        assert f"npx playwright install chromium >> '{log}' 2>&1" in lines
        assert (
            f"{CHECK_UNIT} loopback '{UNIT}' /usr/bin/env {passed} make check >> '{log}' 2>&1"
        ) in lines
        assert f"make check >> '{log}' 2>&1" not in lines
        # Everything else is the build a host-network check renders.
        host = _build(None).splitlines()
        assert [line for line in lines if "make check" not in line] == [
            line for line in host if "make check" not in line
        ]

    def test_every_variable_the_build_exports_is_passed_on(self) -> None:
        exported = [line for line in _build(UNIT).splitlines() if line.startswith("export ")]

        # The prologue's PATH, then the build's own, which env(1) passes on with it.
        assert exported == ["export PATH", f"export {' '.join(BUILD_EXPORTS)}"]
        assert recipe_command(None) == "make check"

    def test_the_command_hands_make_the_values_the_build_exported(
        self, tmp_path: pathlib.Path
    ) -> None:
        (tmp_path / "Makefile").write_text(
            "check:\n\t@printenv " + " ".join(BUILD_EXPORTS) + "\n", encoding="utf-8"
        )
        command = recipe_command(UNIT).removeprefix(f"{CHECK_UNIT} loopback '{UNIT}' ")
        exports = " ".join(f"{name}=value-of-{name}" for name in BUILD_EXPORTS)
        # env -i first, so a value can only reach make through the command,
        # on the system PATH an isolated build starts from (make is in the
        # system profile on NixOS, MCPs board task c2e3f31f).
        script = f"export {exports}\ncd '{tmp_path}'\n{command}\n"
        ran = subprocess.run(
            ["/usr/bin/env", "-i", f"PATH={EXEC_PATH}", "/bin/sh", "-c", script],
            text=True,
            capture_output=True,
            check=True,
        )

        assert ran.stdout.splitlines() == [f"value-of-{name}" for name in BUILD_EXPORTS]
