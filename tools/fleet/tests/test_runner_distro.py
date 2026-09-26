"""Running a bash payload inside a runner host's distro: the path and the route."""

from __future__ import annotations

import pytest

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import _test_hooks, runner_distro
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
        assert "C:/fleet/stage/fleet-x.sh" in " ".join(runner.calls[0])
        assert runner.stdin[1] == (
            b"$ErrorActionPreference = 'Stop'\n$env:WSL_UTF8 = '1'\n"
            b"wsl -d 'Ubuntu' -u root -- bash '/mnt/c/fleet/stage/fleet-x.sh'\n"
            b"exit $LASTEXITCODE"
        )
        assert "C:/fleet/stage/fleet-x-driver.ps1" in " ".join(runner.calls[2])
        assert runner.timeouts == [120, 120, 900]
