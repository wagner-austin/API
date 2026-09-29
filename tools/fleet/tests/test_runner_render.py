"""The rendered provision: both scripts, the manual steps, and the payloads.

The hygiene payload constants are asserted here as executable content --
shellcheck-grade claims a regex can make -- because this module is now the
source of truth for a script that previously existed only on one machine's
disk.
"""

from __future__ import annotations

import pathlib
import subprocess

from fleet.contracts.runners import FileAsset, HostRunnerSpec, RunnerInstall
from fleet.core import (
    runner_install,
    runner_recovery,
    runner_render,
    runner_slice_render,
    runner_windows_provision,
)
from tests._host_bash import host_bash
from tests._runner_fixtures import a_base, a_ci_slice


def _host(
    *,
    keepalive_task: str | None = "wsl-keepalive",
    wslconfig_min_memory_gb: int | None = 26,
    assets: list[FileAsset] | None = None,
) -> HostRunnerSpec:
    """A host spec shaped per test, two installs across two repos.

    Args:
        keepalive_task: The keepalive declaration.
        wslconfig_min_memory_gb: The memory floor declaration.
        assets: The asset declarations, default one of each kind.

    Returns:
        The spec.
    """
    return HostRunnerSpec(
        name="lavender",
        host="lavender",
        wsl_distro="Ubuntu",
        keepalive_task=keepalive_task,
        wslconfig_min_memory_gb=wslconfig_min_memory_gb,
        scratch_dir="C:/fleet/stage",
        gpu_required=True,
        systemd_timers=["ci-clean.timer"],
        installs=[
            RunnerInstall(
                repo="wagner-austin/API",
                runner_name="lavender-wsl",
                side="wsl",
                service="actions.runner.wagner-austin-API.lavender-wsl.service",
                workdir="/home/gharunner/actions-runner-1/_work",
                labels=["lavender-wsl"],
                python_toolcache=[],
            ),
            RunnerInstall(
                repo="wagner-austin/MCPs",
                runner_name="lavender-wsl",
                side="wsl",
                service="actions.runner.wagner-austin-MCPs.lavender-wsl.service",
                workdir="/home/gharunner/actions-runner-2/_work",
                labels=["lavender-wsl", "linux-ci"],
                python_toolcache=[],
            ),
        ],
        assets=[
            FileAsset(
                path="/opt/corvis/rw-game/game-lib.jar",
                sha256="8a" * 32,
                writable=False,
                reason="the provenance jar",
                manual=True,
                provision_command=None,
            ),
            FileAsset(
                path="/opt/llama.cpp/convert_lora_to_gguf.py",
                sha256=None,
                writable=False,
                reason="the GGUF converter",
                manual=False,
                provision_command="git clone --depth 1 https://example.invalid/l /opt/llama.cpp",
            ),
            FileAsset(
                path="/data",
                sha256=None,
                writable=True,
                reason="checkpoint writes",
                manual=False,
                provision_command=None,
            ),
        ]
        if assets is None
        else assets,
        base=a_base(),
        ci_slice=a_ci_slice(),
    )


class TestWindowsScript:
    """provision.ps1 is :mod:`fleet.core.runner_windows_provision`'s, tested there."""

    def test_the_windows_script_is_the_windows_provision_with_no_tokens(self) -> None:
        spec = _host()
        rendered = runner_render.render_provision(spec)
        assert rendered["windows_script"] == (
            runner_windows_provision.render_windows_provision_script(spec, {})
        )
        # A Windows install never leaks into the Linux script, whose
        # environment cannot run it.
        assert "config.cmd" not in rendered["linux_script"]


class TestLinuxScript:
    """provision.sh."""

    def test_the_hygiene_payloads_are_embedded_byte_identically(self) -> None:
        script = runner_render.render_provision(_host())["linux_script"]
        assert runner_render.CI_CLEAN_SCRIPT.rstrip("\n") in script
        assert runner_render.CI_CLEAN_SERVICE.rstrip("\n") in script
        assert runner_render.CI_CLEAN_TIMER.rstrip("\n") in script
        assert "systemctl enable --now ci-clean.timer" in script

    def test_a_writable_asset_becomes_mkdir_and_chown(self) -> None:
        script = runner_render.render_provision(_host())["linux_script"]
        assert "mkdir -p '/data'" in script
        assert "chown gharunner:gharunner '/data'" in script

    def test_a_fetchable_asset_runs_its_own_command_idempotently(self) -> None:
        script = runner_render.render_provision(_host())["linux_script"]
        assert "if [ ! -e '/opt/llama.cpp/convert_lora_to_gguf.py' ]; then" in script
        assert "git clone --depth 1 https://example.invalid/l /opt/llama.cpp" in script

    def test_a_manual_asset_appears_in_no_script(self) -> None:
        rendered = runner_render.render_provision(_host())
        assert "rw-game" not in rendered["linux_script"]
        assert "rw-game" not in rendered["windows_script"]

    def test_each_install_demands_its_repos_token_and_configures_it(self) -> None:
        script = runner_render.render_provision(_host())["linux_script"]
        assert "RUNNER_TOKEN_API" in script
        assert "RUNNER_TOKEN_MCPS" in script
        assert "--url https://github.com/wagner-austin/API" in script
        assert "--labels lavender-wsl,linux-ci" in script
        assert runner_install.RUNNER_VERSION in script

    def test_a_same_named_registration_is_taken_back(self) -> None:
        """A rebuilt host's old runners are still registered, offline, under
        the roster's names (board task 1aa6a021), so config.sh replaces
        rather than refuses; rendered-provision-windows.Tests.ps1 holds
        config.cmd to the same."""
        wsl = RunnerInstall(
            repo="wagner-austin/MCPs",
            runner_name="lavender-wsl",
            side="wsl",
            service="actions.runner.wagner-austin-MCPs.lavender-wsl.service",
            workdir="/home/gharunner/actions-runner/_work",
            labels=["lavender-wsl"],
            python_toolcache=[],
        )
        [configure] = [
            line for line in runner_render.render_wsl_install_lines(wsl) if "./config.sh" in line
        ]
        assert configure.endswith('--name lavender-wsl --labels lavender-wsl --replace"')

    def test_the_local_bin_line_appends_once_when_run_for_real(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The .path lines are executed, twice, by a real bash against a real
        file shaped as config.sh writes it: one colon-joined line. Both
        entries land once and in order, and the second run leaves the file
        unchanged, because the provision is re-run after any roster change.

        The lines run from a script file, as runner_onboard delivers them
        (``bash <file>``), never through ``bash -c``: on a host whose PATH
        resolves bash to System32's WSL launcher, that launcher re-expands
        its argument and turns ``s#$#`` into ``s#0#``, which the 2026-09-26
        review re-run hit and production never can."""
        bash = host_bash()
        install = RunnerInstall(
            repo="wagner-austin/MCPs",
            runner_name="lavender-wsl",
            side="wsl",
            service="actions.runner.wagner-austin-MCPs.lavender-wsl.service",
            workdir="rt/_work",
            labels=["lavender-wsl"],
            python_toolcache=[],
        )
        appends = [
            line for line in runner_render.render_wsl_install_lines(install) if ".path" in line
        ]
        assert len(appends) == len(runner_render.RUNNER_PATH_ENTRIES)
        runner_dir = tmp_path / "rt"
        runner_dir.mkdir()
        (runner_dir / ".path").write_bytes(b"/usr/local/bin:/usr/bin:/bin\n")
        (tmp_path / "append-path.sh").write_bytes(("\n".join(appends) + "\n").encode())
        for _ in range(2):
            ran = subprocess.run(
                [bash, "append-path.sh"],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                check=False,
            )
            assert ran.returncode == 0, ran.stderr
        assert (runner_dir / ".path").read_bytes() == (
            f"/usr/local/bin:/usr/bin:/bin:{runner_render.LOCAL_BIN}:{runner_render.WSL_LIB}\n"
        ).encode()

    def test_the_ci_clean_payload_gates_on_a_running_worker(self) -> None:
        assert "Runner.Worker" in runner_render.CI_CLEAN_SCRIPT
        assert "skipping this pass" in runner_render.CI_CLEAN_SCRIPT

    def test_the_ci_clean_payload_keeps_venvs_of_unknown_provenance(self) -> None:
        assert "unknown provenance: keep" in runner_render.CI_CLEAN_SCRIPT

    def test_the_ci_clean_payload_prunes_docker_before_the_worker_gate(self) -> None:
        # The order IS the fix: a prune behind the gate never runs on a host
        # whose eight runners are never all idle, and that host is lavender.
        script = runner_render.CI_CLEAN_SCRIPT
        gate = script.index("pgrep -f 'Runner.Worker'")
        for command in (
            "docker container prune -f --filter until=24h",
            "docker volume prune -f",
            "docker image prune -f",
        ):
            assert script.index(command) < gate, command

    def test_the_ci_clean_payload_never_prunes_named_volumes_or_tagged_images(
        self,
    ) -> None:
        # --all widens volume prune to named volumes and image prune to every
        # unused tagged image, the pull cache each job would pay back.
        docker_lines = [
            line
            for line in runner_render.CI_CLEAN_SCRIPT.splitlines()
            if line.startswith("docker ")
        ]
        assert len(docker_lines) == 3
        assert [line for line in docker_lines if "--all" in line or " -a" in line] == []

    def test_the_ci_clean_timer_fires_daily(self) -> None:
        assert "OnCalendar=*-*-* 04:00" in runner_render.CI_CLEAN_TIMER
        assert "Persistent=true" in runner_render.CI_CLEAN_TIMER


class TestRerunsOverAHalfBuiltHost:
    """--rebuild re-runs every stage, so a configured install is left alone."""

    def test_a_configured_wsl_runner_skips_config_and_svc_install(self) -> None:
        install = RunnerInstall(
            repo="wagner-austin/MCPs",
            runner_name="lavender-wsl",
            side="wsl",
            service="actions.runner.wagner-austin-MCPs.lavender-wsl.service",
            workdir="/home/gharunner/actions-runner/_work",
            labels=["lavender-wsl"],
            python_toolcache=[],
        )
        lines = runner_render.render_wsl_install_lines(install)
        guard = lines.index("if [ ! -f /home/gharunner/actions-runner/.runner ]; then")
        assert "./config.sh" in lines[guard + 1]
        assert lines[guard + 2] == "fi"
        # The unit exists after svc.sh install and gets its restart policy
        # before it starts, and is moved into the CI slice after it starts,
        # when its live cgroup can be read.
        recovery = runner_recovery.render_wsl_recovery_lines(install)
        placed = runner_slice_render.render_runner_slice_lines(install)
        start = len(lines) - len(placed) - 1
        assert lines[start - 1 - len(recovery)] == (
            "[ -f /home/gharunner/actions-runner/.service ] || "
            "(cd /home/gharunner/actions-runner && ./svc.sh install gharunner)"
        )
        assert lines[start - len(recovery) : start] == recovery
        assert lines[start] == "(cd /home/gharunner/actions-runner && ./svc.sh start)"
        assert lines[start + 1 :] == placed


class TestManualSteps:
    """What no script may do."""

    def test_every_manual_asset_becomes_a_loud_step(self) -> None:
        rendered = runner_render.render_provision(_host())
        assert len(rendered["manual_steps"]) == 1
        step = rendered["manual_steps"][0]
        assert step.startswith("PLACE BY HAND: /opt/corvis/rw-game/game-lib.jar")
        assert "re-run fleet-runners audit" in step

    def test_a_host_with_no_manual_assets_has_no_steps(self) -> None:
        assets = [
            FileAsset(
                path="/data",
                sha256=None,
                writable=True,
                reason="checkpoint writes",
                manual=False,
                provision_command=None,
            )
        ]
        rendered = runner_render.render_provision(_host(assets=assets))
        assert rendered["manual_steps"] == []
