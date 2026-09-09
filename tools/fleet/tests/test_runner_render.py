"""The rendered provision: both scripts, the manual steps, and the payloads.

The hygiene payload constants are asserted here as executable content --
shellcheck-grade claims a regex can make -- because this module is now the
source of truth for a script that previously existed only on one machine's
disk.
"""

from __future__ import annotations

from fleet.contracts.runners import FileAsset, HostRunnerSpec, RunnerInstall
from fleet.core import runner_render


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
            ),
            RunnerInstall(
                repo="wagner-austin/MCPs",
                runner_name="lavender-wsl",
                side="wsl",
                service="actions.runner.wagner-austin-MCPs.lavender-wsl.service",
                workdir="/home/gharunner/actions-runner-2/_work",
                labels=["lavender-wsl", "linux-ci"],
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
    )


class TestWindowsScript:
    """provision.ps1."""

    def test_the_memory_floor_becomes_a_wslconfig(self) -> None:
        rendered = runner_render.render_provision(_host())
        assert "memory=26GB" in rendered["windows_script"]
        assert ".wslconfig" in rendered["windows_script"]

    def test_the_keepalive_task_is_registered_and_started(self) -> None:
        script = runner_render.render_provision(_host())["windows_script"]
        assert "/tn 'wsl-keepalive'" in script
        assert "schtasks /run /tn 'wsl-keepalive'" in script
        assert "sleep infinity" in script

    def test_a_windows_side_install_joins_the_windows_script(self) -> None:
        spec = _host()
        windows_install = RunnerInstall(
            repo="wagner-austin/tree-bot",
            runner_name="lavender",
            side="windows",
            service="actions.runner.wagner-austin-tree-bot.lavender",
            workdir="C:/actions-runner-tree-bot/_work",
            labels=["lavender"],
        )
        spec["installs"].append(windows_install)
        script = runner_render.render_provision(spec)["windows_script"]
        expected_block = "\n".join(runner_render.render_windows_install_lines(windows_install))
        assert expected_block in script
        # And it never leaks into the Linux script, whose environment
        # cannot run it.
        assert "config.cmd" not in runner_render.render_provision(spec)["linux_script"]

    def test_a_host_declaring_neither_says_so_instead_of_vanishing(self) -> None:
        rendered = runner_render.render_provision(
            _host(keepalive_task=None, wslconfig_min_memory_gb=None)
        )
        assert "nothing declared" in rendered["windows_script"]
        assert ".wslconfig" not in rendered["windows_script"]
        assert "schtasks" not in rendered["windows_script"]


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
        assert runner_render.RUNNER_VERSION in script

    def test_the_ci_clean_payload_gates_on_a_running_worker(self) -> None:
        assert "Runner.Worker" in runner_render.CI_CLEAN_SCRIPT
        assert "skipping this pass" in runner_render.CI_CLEAN_SCRIPT

    def test_the_ci_clean_payload_keeps_venvs_of_unknown_provenance(self) -> None:
        assert "unknown provenance: keep" in runner_render.CI_CLEAN_SCRIPT


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
