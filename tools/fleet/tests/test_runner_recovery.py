"""Restart-on-failure for the runner services: both halves run for real.

The Windows lines run inside the Windows provision's install loop and are
executed by tests/pester/rendered-provision-windows.Tests.ps1 with a stand-in
sc.exe, so no service on this machine changes. The systemd lines run under
a real bash from a script file (as provision.sh delivers them) with
``systemctl`` defined as a recording function and the one absolute
directory, ``/etc/systemd/system``, rewritten to the test's own directory:
the comparison, the write, the reload and the idempotence all run as
rendered.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.contracts.runners import RunnerInstall
from fleet.core import runner_recovery
from tests._host_bash import host_bash

_UNIT = "actions.runner.wagner-austin-MCPs.lavender-wsl.service"


def _install(service: str) -> RunnerInstall:
    """A wsl-side install.

    Args:
        service: The unit name.

    Returns:
        The install.
    """
    return RunnerInstall(
        repo="wagner-austin/MCPs",
        runner_name="lavender",
        side="wsl",
        service=service,
        workdir="/home/gharunner/r/_work",
        labels=["lavender"],
        python_toolcache=[],
    )


class TestRendered:
    """What renders without running."""

    def test_an_unscriptable_unit_is_refused(self) -> None:
        with pytest.raises(ValueError) as refused:
            runner_recovery.render_wsl_recovery_lines(_install("bad'unit"))
        assert str(refused.value) == (
            'service "bad\'unit" contains "\'", which cannot be embedded in a rendered '
            "script verbatim; rename the item rather than escaping it"
        )

    def test_the_windows_policy_is_written_through_the_sc_parameter(self) -> None:
        """Executed by tests/pester/rendered-provision-windows.Tests.ps1 inside
        the Windows provision's install loop, a refused write included."""
        lines = runner_recovery.render_windows_recovery_lines()
        assert lines[0] == (
            "& $Sc failure $ServiceName reset= 86400 "
            "actions= restart/5000/restart/5000/restart/5000 | Out-Null"
        )
        assert lines[4] == "& $Sc failureflag $ServiceName 1 | Out-Null"


class TestTheSystemdLinesRunForReal:
    """The drop-in lands once, byte for byte, and reloads systemd once."""

    def test_the_drop_in_is_written_once_and_reloaded_once(self, tmp_path: pathlib.Path) -> None:
        bash = host_bash()
        lines = runner_recovery.render_wsl_recovery_lines(_install(_UNIT))
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
