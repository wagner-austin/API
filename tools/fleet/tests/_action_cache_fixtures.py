"""Installs and hosts the action archive cache's suites share (MCPs board task ce2b92e0).

Support module, not a test module: test_runner_action_cache runs the fill,
test_runner_action_cache_units the units and the runner drop-in.
"""

from __future__ import annotations

from typing import Literal

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from tests._runner_fixtures import a_base, a_ci_slice

#: The unit the drop-in cases run, one of lavender's four MCPs runners.
UNIT = "actions.runner.wagner-austin-MCPs.lavender-wsl-4.service"


def an_install(
    service: str,
    side: Literal["wsl", "windows"] = "wsl",
    workdir: str = "/home/gharunner/r/_work",
) -> RunnerInstall:
    """One install.

    Args:
        service: The unit or service name.
        side: ``wsl`` or ``windows``.
        workdir: Its work tree.

    Returns:
        The install.
    """
    return RunnerInstall(
        repo="wagner-austin/MCPs",
        runner_name="lavender-wsl-4",
        side=side,
        service=service,
        workdir=workdir,
        labels=["lavender-wsl"],
        python_toolcache=[],
    )


def a_host(installs: list[RunnerInstall]) -> HostRunnerSpec:
    """lavender with the given installs.

    Args:
        installs: Its installs.

    Returns:
        The roster entry.
    """
    return HostRunnerSpec(
        name="lavender",
        host="lavender",
        wsl_distro="Ubuntu",
        keepalive_task="wsl-keepalive",
        wslconfig_min_memory_gb=26,
        scratch_dir="C:/fleet/stage",
        gpu_required=True,
        systemd_timers=["ci-clean.timer"],
        job_timeout_minutes=360,
        installs=installs,
        assets=[],
        base=a_base(),
        ci_slice=a_ci_slice(),
    )
