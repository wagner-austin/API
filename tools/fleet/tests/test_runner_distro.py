"""Running a bash payload inside a runner host's distro: the path and the route."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import _test_hooks, runner_distro
from fleet.core.dialect_windows import WRITE_COMMAND
from tests._runner_fixtures import a_base
from tests.conftest import FakeRun, ok


def _host() -> HostRunnerSpec:
    """A minimal host.

    Returns:
        The spec.
    """
    return HostRunnerSpec(
        name="lavender",
        host="lavender",
        wsl_distro="Ubuntu",
        keepalive_task=None,
        wslconfig_min_memory_gb=None,
        scratch_dir="C:/fleet/stage",
        gpu_required=False,
        systemd_timers=[],
        installs=[
            RunnerInstall(
                repo="wagner-austin/API",
                runner_name="lavender-wsl",
                side="wsl",
                service="actions.runner.wagner-austin-API.lavender-wsl.service",
                workdir="/home/gharunner/actions-runner-api-1/_work",
                labels=["lavender-wsl"],
                python_toolcache=[],
            )
        ],
        assets=[],
        base=a_base(),
    )


class TestWindowsToWslPath:
    """The one path translation the distro route needs."""

    def test_a_drive_path_translates(self) -> None:
        assert runner_distro.windows_to_wsl_path("C:/fleet/stage/x.sh") == "/mnt/c/fleet/stage/x.sh"

    @pytest.mark.parametrize("path", ["relative/x", "/posix/x", "C:x", ""])
    def test_anything_else_is_refused(self, path: str) -> None:
        with pytest.raises(ValueError, match="drive path"):
            runner_distro.windows_to_wsl_path(path)


class TestRunDistroScript:
    """The payload is a file, run by path through a one-line driver."""

    def test_the_payload_and_its_driver_are_sent_and_the_driver_runs(self) -> None:
        runner = FakeRun([ok(""), ok(""), ok("ran\n")])
        _test_hooks.run = runner
        output = runner_distro.run_distro_script(
            _host(), "fleet-x", "#!/usr/bin/env bash\necho ran\n", timeout_seconds=900
        )
        assert output == "ran\n"
        assert runner.stdin[0] == b"#!/usr/bin/env bash\necho ran\n"
        assert runner.calls[0][-1] == runner_distro.EXACT_WRITE_COMMAND.format(
            path="C:/fleet/stage/fleet-x.sh"
        )
        assert runner.stdin[1] == (
            b"$ErrorActionPreference = 'Stop'\n$env:WSL_UTF8 = '1'\n"
            b"wsl -d 'Ubuntu' -u root -- bash '/mnt/c/fleet/stage/fleet-x.sh'\n"
            b"exit $LASTEXITCODE"
        )
        assert runner.calls[2][-1] == "C:/fleet/stage/fleet-x-driver.ps1"
        assert runner.timeouts == [120, 120, 900]


#: A payload with everything the dialect's own write would change: a
#: shebang that must stay first, LF line ends, and a non-ASCII character.
_PAYLOAD = "#!/usr/bin/env bash\nset -euo pipefail\necho 'caf\u00e9'\n".encode()


#: The same payload in ASCII, for the one claim a re-encoding cannot blur.
_ASCII_PAYLOAD = b"#!/usr/bin/env bash\nset -euo pipefail\necho ran\n"


def _write_locally(command: str, target: pathlib.Path, payload: bytes) -> bytes:
    """Run a write command the way cmd.exe hands it to PowerShell.

    Args:
        command: The write command, its path already in place.
        target: The file it writes.
        payload: The bytes streamed to it.

    Returns:
        The bytes that landed.
    """
    ran = subprocess.run(command, input=payload, capture_output=True, check=False)
    assert ran.returncode == 0, ran.stderr
    return target.read_bytes()


@pytest.mark.skipif(sys.platform != "win32", reason="the write command is Windows PowerShell")
class TestTheWriteIsByteExact:
    """What lands on the runner host is exactly what bash will read."""

    def test_the_payload_lands_byte_for_byte(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "stage" / "payload.sh"
        command = runner_distro.EXACT_WRITE_COMMAND.format(path=target.as_posix())
        assert _write_locally(command, target, _PAYLOAD) == _PAYLOAD

    def test_the_dialect_write_it_replaces_adds_a_bom_and_crlf(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The defect, reproduced: bash reads the BOM as part of the
        shebang and the CR as part of ``pipefail``, as on lavender. (That
        write also decodes stdin in the console code page, so a non-ASCII
        character arrives changed too; the ASCII payload keeps this claim
        exact.)"""
        target = tmp_path / "stage" / "payload.sh"
        command = WRITE_COMMAND.format(path=target.as_posix())
        written = _write_locally(command, target, _ASCII_PAYLOAD)
        assert written == b"\xef\xbb\xbf" + _ASCII_PAYLOAD.replace(b"\n", b"\r\n")
