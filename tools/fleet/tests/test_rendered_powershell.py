"""The committed PowerShell renders: current, complete, and written by one command.

The drift gate is the first class: every script
:func:`fleet.core.rendered_powershell.render_all` produces from the real
roster must be committed under ``rendered/`` byte for byte (line endings
aside, which git normalises), and every committed ``.ps1`` there must have an
entry. Pester executes the committed copies (``tests/pester/rendered-*``), so
a stale copy would test a script no host receives.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest

from fleet.cli import render_powershell
from fleet.cli.runners import load_runner_spec
from fleet.contracts.runners import RunnerSpec
from fleet.core import rendered_powershell, runner_rebuild
from fleet.core.powershell_text import STRICT_HEADER, system32_parameter

#: tools/fleet.
_PACKAGE = pathlib.Path(__file__).resolve().parent.parent

#: The committed copies.
_RENDERED = _PACKAGE / rendered_powershell.RENDERED_DIRECTORY

#: The real roster the host scripts are rendered from.
_ROSTER = _PACKAGE / "runners.json"


def _roster() -> RunnerSpec:
    """The real roster.

    Returns:
        ``runners.json``, decoded.
    """
    return load_runner_spec(str(_ROSTER))


class TestTheCommittedCopies:
    """``make render-powershell`` has been run since the last renderer change."""

    def test_every_render_is_committed_as_rendered(self) -> None:
        stale = [
            script["name"]
            for script in rendered_powershell.render_all(_roster())
            if not (_RENDERED / f"{script['name']}.ps1").is_file()
            or (_RENDERED / f"{script['name']}.ps1").read_text(encoding="utf-8") != script["text"]
        ]
        assert stale == [], f"run `make render-powershell` in tools/fleet; stale: {stale}"

    def test_every_committed_copy_has_a_renderer(self) -> None:
        registered = {script["name"] for script in rendered_powershell.render_all(_roster())}
        committed = [path.stem for path in _RENDERED.glob("*.ps1")]
        orphans = sorted(stem for stem in committed if stem not in registered)
        assert orphans == [], f"delete the copies no renderer writes any more: {orphans}"

    def test_the_host_scripts_are_rendered_once_per_roster_host(self) -> None:
        roster = _roster()
        names = [script["name"] for script in rendered_powershell.render_all(roster)]
        assert names == [
            "rebuild-boot-instant",
            "rebuild-restart",
            "dialect-capacity-probe",
            "dialect-toolchain-probe",
            "dialect-observe-sessions",
            "dialect-make-directory",
            "dialect-reset-directory",
            "dialect-digest",
            "dialect-result",
            "dialect-log-tail",
            "dialect-launch",
            "dialect-stop",
            "dialect-extract",
            "dialect-init-repository",
            "dialect-companion-repository",
            "dialect-build",
            *(f"rebuild-terminate-{host['name']}" for host in roster["hosts"]),
        ]


class TestTheRegistry:
    def test_two_scripts_sharing_a_file_name_are_refused(self) -> None:
        host = _roster()["hosts"][0]
        shared = f"share a file name: rebuild-terminate-{host['name']}"
        with pytest.raises(ValueError, match=shared):
            rendered_powershell.render_all(RunnerSpec(hosts=[host, host]))


class TestTheSharedLines:
    def test_every_render_opens_with_the_strict_header_after_its_parameters(self) -> None:
        for script in rendered_powershell.render_all(_roster()):
            lines = script["text"].splitlines()
            start = lines.index(")") + 1 if lines[0] == "param(" else 0
            assert tuple(lines[start : start + 2]) == STRICT_HEADER, script["name"]

    def test_a_system32_parameter_names_the_binary_by_its_absolute_path(self) -> None:
        assert (
            system32_parameter("Wsl", "wsl.exe")
            == '[string]$Wsl = "$env:SystemRoot\\System32\\wsl.exe"'
        )

    @pytest.mark.parametrize(
        ("name", "binary", "message"),
        [
            ("wsl", "wsl.exe", "PascalCase"),
            ("Wsl", "C:/wsl.exe", "lowercase name ending .exe"),
            ("Wsl", "wsl", "lowercase name ending .exe"),
        ],
    )
    def test_a_value_that_is_not_plain_is_refused(
        self, name: str, binary: str, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            system32_parameter(name, binary)

    def test_a_distro_name_that_cannot_be_embedded_is_refused(self) -> None:
        with pytest.raises(ValueError, match="wsl_distro"):
            runner_rebuild.render_terminate_script("Ubu'ntu")


class TestTheCommand:
    def test_it_writes_every_render_to_the_directory_named(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "rendered"
        assert render_powershell.main(["--spec", str(_ROSTER), "--out", str(out)]) == 0
        written = {path.name: path.read_text(encoding="utf-8") for path in out.glob("*.ps1")}
        assert written == {
            f"{script['name']}.ps1": script["text"]
            for script in rendered_powershell.render_all(_roster())
        }

    def test_the_entrypoint_exits_with_mains_status(self, tmp_path: pathlib.Path) -> None:
        saved = sys.argv
        sys.argv = ["x", "--spec", str(_ROSTER), "--out", str(tmp_path)]
        try:
            with pytest.raises(SystemExit) as raised:
                render_powershell.entrypoint()
        finally:
            sys.argv = saved
        assert raised.value.code == 0

    def test_running_as_a_module_actually_runs(self, tmp_path: pathlib.Path) -> None:
        saved_argv = sys.argv
        saved_module = sys.modules.pop("fleet.cli.render_powershell", None)
        sys.argv = ["x", "--spec", str(_ROSTER), "--out", str(tmp_path)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(
                    "fleet.cli.render_powershell", run_name="__main__", alter_sys=False
                )
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules["fleet.cli.render_powershell"] = saved_module
        assert raised.value.code == 0
        assert (tmp_path / "rebuild-restart.ps1").is_file()
