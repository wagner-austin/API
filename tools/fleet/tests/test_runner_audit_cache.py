"""The cache rows: which directories a runner host is held to, and how.

The rendered lines are executed by the Pester suite over the committed
``rendered/audit-<host>.ps1`` (``tests/pester/rendered-audit.Tests.ps1``),
against a stand-in ``wsl`` that answers ``du`` as a real distro does; these
tests hold the rows to the roster.
"""

from __future__ import annotations

from typing import Literal

import pytest

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import runner_audit_cache
from tests._runner_fixtures import a_base, a_ci_slice


def _install(side: Literal["wsl", "windows"], workdir: str, name: str) -> RunnerInstall:
    """One install on the given side.

    Args:
        side: ``wsl`` or ``windows``.
        workdir: The install's ``_work`` tree.
        name: The runner name.

    Returns:
        The install.
    """
    return RunnerInstall(
        repo="wagner-austin/MCPs",
        runner_name=name,
        side=side,
        service=f"actions.runner.{name}",
        workdir=workdir,
        labels=[name],
        python_toolcache=[],
    )


def _host(installs: list[RunnerInstall]) -> HostRunnerSpec:
    """A host with the given installs and the fixture base.

    Args:
        installs: The installs.

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
        installs=installs,
        assets=[],
        base=a_base(),
        ci_slice=a_ci_slice(),
    )


class TestCacheRows:
    def test_the_cache_comes_first_then_each_wsl_work_tree_in_roster_order(self) -> None:
        spec = _host(
            [
                _install("wsl", "/home/gharunner/actions-runner/_work", "lavender-wsl"),
                _install("windows", "C:/actions-runner/_work", "lavender-win"),
                _install("wsl", "/home/gharunner/actions-runner-2/_work", "lavender-wsl-2"),
            ]
        )
        assert runner_audit_cache.cache_rows(spec) == [
            runner_audit_cache.CacheRow(
                check_id="cache:/home/gharunner/.cache:ceiling-60gb",
                path="/home/gharunner/.cache",
                ceiling_gb=60,
                reason=runner_audit_cache.CACHE_REASON,
            ),
            runner_audit_cache.CacheRow(
                check_id="cache:/home/gharunner/actions-runner/_work:ceiling-15gb",
                path="/home/gharunner/actions-runner/_work",
                ceiling_gb=15,
                reason=runner_audit_cache.WORK_REASON,
            ),
            runner_audit_cache.CacheRow(
                check_id="cache:/home/gharunner/actions-runner-2/_work:ceiling-15gb",
                path="/home/gharunner/actions-runner-2/_work",
                ceiling_gb=15,
                reason=runner_audit_cache.WORK_REASON,
            ),
        ]

    def test_a_host_with_only_windows_installs_has_the_cache_row_alone(self) -> None:
        spec = _host([_install("windows", "C:/actions-runner/_work", "lavender-win")])
        ids = [row["check_id"] for row in runner_audit_cache.cache_rows(spec)]
        assert ids == ["cache:/home/gharunner/.cache:ceiling-60gb"]


class TestRenderCacheCheckLines:
    def test_each_row_is_measured_by_du_and_held_to_its_own_ceiling(self) -> None:
        spec = _host([_install("wsl", "/home/gharunner/actions-runner/_work", "lavender-wsl")])
        lines = runner_audit_cache.render_cache_check_lines(spec)
        assert len(lines) == 12
        assert lines[0] == (
            "$Probe = Invoke-InDistro $Cmd $Wsl $Distro \"du -s -BG '/home/gharunner/.cache'\""
        )
        assert lines[5].startswith(
            "Write-Check 'cache:/home/gharunner/.cache:ceiling-60gb' ($Probe.Exit -eq 0 -and "
            "$CacheGb -ge 0 -and $CacheGb -le 60)"
        )
        assert lines[6] == (
            "$Probe = Invoke-InDistro $Cmd $Wsl $Distro "
            "\"du -s -BG '/home/gharunner/actions-runner/_work'\""
        )
        assert "$CacheGb -le 15)" in lines[11]

    @pytest.mark.parametrize("bad", ["/home/it's", "/home/$x", "/home/a`b"])
    def test_a_path_that_cannot_be_embedded_is_refused(self, bad: str) -> None:
        spec = _host([_install("wsl", bad, "lavender-wsl")])
        with pytest.raises(ValueError, match="cannot be embedded"):
            runner_audit_cache.render_cache_check_lines(spec)
