"""Restart-on-failure for the runner services: both halves run for real.

The Windows lines run under Windows PowerShell 5.1 with an ``sc.exe``
function recording its arguments, so no service on this machine changes.
The systemd lines run under a real bash from a script file (as provision.sh
delivers them) with ``systemctl`` defined as a recording function and the
one absolute directory, ``/etc/systemd/system``, rewritten to the test's
own directory: the comparison, the write, the reload and the idempotence
all run as rendered.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from typing import Literal

import pytest

from fleet.contracts.runners import RunnerInstall
from fleet.core import runner_recovery
from fleet.core.dialect_windows import POWERSHELL_INVOCATION
from tests._host_bash import host_bash

_UNIT = "actions.runner.wagner-austin-MCPs.lavender-wsl.service"
_SERVICE = "actions.runner.wagner-austin-MCPs.lavender"


def _install(side: Literal["wsl", "windows"], service: str) -> RunnerInstall:
    """An install on one side.

    Args:
        side: ``"windows"`` or ``"wsl"``.
        service: The service or unit name.

    Returns:
        The install.
    """
    workdir = "C:/actions-runner/_work" if side == "windows" else "/home/gharunner/r/_work"
    return RunnerInstall(
        repo="wagner-austin/MCPs",
        runner_name="lavender",
        side=side,
        service=service,
        workdir=workdir,
        labels=["lavender"],
        python_toolcache=[],
    )


class TestRendered:
    """The refusal that renders without running."""

    def test_an_unscriptable_unit_is_refused_on_both_sides(self) -> None:
        for render in (
            runner_recovery.render_windows_recovery_lines,
            runner_recovery.render_wsl_recovery_lines,
        ):
            with pytest.raises(ValueError) as refused:
                render(_install("wsl", "bad'unit"))
            assert str(refused.value) == (
                'service "bad\'unit" contains "\'", which cannot be embedded in a rendered '
                "script verbatim; rename the item rather than escaping it"
            )


@pytest.mark.skipif(sys.platform != "win32", reason="the lines are PowerShell; run them here")
class TestTheWindowsLinesRunForReal:
    """sc.exe is handed the fleet's restart policy, and a failure throws."""

    def _run(self, tmp_path: pathlib.Path, exit_code: int) -> subprocess.CompletedProcess[str]:
        fakes = (
            "function sc.exe { Write-Host ('SC ' + ($args -join ' ')); "
            f"$global:LASTEXITCODE = {exit_code} }}\n"
        )
        script = tmp_path / "recovery.ps1"
        lines = runner_recovery.render_windows_recovery_lines(_install("windows", _SERVICE))
        script.write_text(fakes + "\n".join(lines) + "\n", encoding="utf-8")
        return subprocess.run(
            [*POWERSHELL_INVOCATION, str(script)], capture_output=True, text=True, check=False
        )

    def test_restarts_forever_and_on_a_non_crash_failure(self, tmp_path: pathlib.Path) -> None:
        ran = self._run(tmp_path, 0)
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout.splitlines() == [
            f"SC failure {_SERVICE} reset= 86400 actions= restart/5000/restart/5000/restart/5000",
            f"SC failureflag {_SERVICE} 1",
        ]

    def test_a_refused_write_throws_with_the_exit_code(self, tmp_path: pathlib.Path) -> None:
        ran = self._run(tmp_path, 1060)
        assert ran.returncode == 1
        assert ran.stdout.splitlines() == [
            f"SC failure {_SERVICE} reset= 86400 actions= restart/5000/restart/5000/restart/5000"
        ]
        assert " ".join(ran.stderr.split()).startswith(f"sc.exe failure {_SERVICE} exited 1060")


class TestTheSystemdLinesRunForReal:
    """The drop-in lands once, byte for byte, and reloads systemd once."""

    def test_the_drop_in_is_written_once_and_reloaded_once(self, tmp_path: pathlib.Path) -> None:
        bash = host_bash()
        lines = runner_recovery.render_wsl_recovery_lines(_install("wsl", _UNIT))
        body = "\n".join(lines).replace("/etc/systemd/system", "units")
        script = 'systemctl() { echo "$*" >> reloads.log; }\n' + body + "\n"
        (tmp_path / "recovery.sh").write_bytes(script.encode())
        outputs = []
        for _ in range(2):
            ran = subprocess.run(
                [bash, "recovery.sh"], cwd=tmp_path, capture_output=True, text=True, check=False
            )
            assert ran.returncode == 0, ran.stderr
            outputs.append(ran.stdout)
        assert outputs == [f"restart policy set for {_UNIT}\n", ""]
        drop_in = tmp_path / "units" / f"{_UNIT}.d" / runner_recovery.SYSTEMD_DROP_IN_NAME
        assert drop_in.read_bytes() == runner_recovery.SYSTEMD_DROP_IN.encode()
        assert (tmp_path / "reloads.log").read_text(encoding="utf-8") == "daemon-reload\n"
