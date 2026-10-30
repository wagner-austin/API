"""The rendered converge scripts: the base, the provision, and the manual steps.

The base and the lines that act on a host's files run here under a real bash,
each command they ask a stand-in that records what it was asked
(tests/_stand_ins.py). The hygiene payloads are tests/test_ci_clean.py's and
each section's lines their own module's; here they are asserted as composed
into the provision, in order.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

import pytest

from fleet.contracts.runners import FileAsset, HostRunnerSpec, RunnerInstall
from fleet.core import (
    ci_clean,
    runner_action_cache,
    runner_cache_adopt,
    runner_install,
    runner_job_scope_render,
    runner_reaper_render,
    runner_recovery,
    runner_render,
    runner_slice_render,
)
from fleet.core.runner_cpuset_render import render_ci_cpuset_lines
from fleet.core.runner_unit_files import render_unit_file_lines
from tests._host_bash import bash_path
from tests._runner_fixtures import RESERVED_CORES, a_host, an_install
from tests._stand_ins import run_with_stand_ins

#: The base's line once it has done everything.
_READY = "base ready: packages installed, docker reachable as gharunner"


def _recording(tmp_path: pathlib.Path, name: str, then: str = "exit 0") -> str:
    """A stand-in body that appends its command line to ``asked``, then answers.

    Args:
        tmp_path: The case's directory, holding ``asked``.
        name: The command's name, written before its arguments.
        then: What the stand-in does after recording.

    Returns:
        The body.
    """
    return f'echo "{name} $*" >> "{bash_path(tmp_path)}/asked"\n{then}'


def _asked(tmp_path: pathlib.Path) -> list[str]:
    """Every command line the recording stand-ins were asked, in order.

    Args:
        tmp_path: The case's directory.

    Returns:
        The lines, none when nothing was asked.
    """
    asked = tmp_path / "asked"
    return asked.read_text().splitlines() if asked.exists() else []


def _assets(tmp_path: pathlib.Path) -> list[FileAsset]:
    """One asset of each kind, the fetchable one's command recording its run.

    Args:
        tmp_path: The case's directory, which the fetchable and writable
            assets live under, spelled as the host's bash spells it.

    Returns:
        A manual jar, a fetchable converter and a writable directory.
    """
    base = bash_path(tmp_path)
    converter = f"{base}/convert.py"
    return [
        FileAsset(
            path="/opt/corvis/rw-game/game-lib.jar",
            sha256="8a" * 32,
            writable=False,
            reason="the provenance jar",
            manual=True,
            provision_command=None,
        ),
        FileAsset(
            path=converter,
            sha256=None,
            writable=False,
            reason="the GGUF converter",
            manual=False,
            provision_command=f"echo fetched >> '{base}/asked'; touch '{converter}'",
        ),
        FileAsset(
            path=f"{base}/data",
            sha256=None,
            writable=True,
            reason="checkpoint writes",
            manual=False,
            provision_command=None,
        ),
    ]


def _two_repos(tmp_path: pathlib.Path) -> HostRunnerSpec:
    """A host with an install of each of two repositories and every kind of asset.

    Args:
        tmp_path: The case's directory, for the assets.

    Returns:
        The spec.
    """
    return a_host(
        [an_install("wagner-austin/API"), an_install("wagner-austin/MCPs")],
        assets=_assets(tmp_path),
    )


class TestTheBase:
    """base.sh, run for real."""

    def _run(
        self, tmp_path: pathlib.Path, *, reboot: bool = False, **answers: str
    ) -> tuple[int, str, str]:
        """Run the base with every command it asks a recording stand-in.

        Args:
            tmp_path: The case's directory.
            reboot: Whether a package asked for a reboot.
            **answers: Stand-in bodies replacing the recording ones.

        Returns:
            The exit status, standard output and standard error.
        """
        flag = tmp_path / "reboot-required"
        if reboot:
            flag.touch()
        script = runner_render.render_base_script(a_host())
        commands = {
            name: _recording(tmp_path, name)
            for name in ("systemctl", "apt-get", "useradd", "usermod", "runuser")
        }
        commands["id"] = "exit 1"
        ran = run_with_stand_ins(
            tmp_path,
            script.replace(
                runner_render.REBOOT_REQUIRED_PATH, f"{bash_path(tmp_path)}/reboot-required"
            ).splitlines(),
            {**commands, **answers},
            options=(),
        )
        return ran.returncode, ran.stdout, ran.stderr

    def test_a_fresh_node_gets_the_packages_and_the_account_in_docker(
        self, tmp_path: pathlib.Path
    ) -> None:
        assert self._run(tmp_path) == (0, f"{_READY}\n", "")
        assert _asked(tmp_path) == [
            "systemctl is-active --quiet docker",
            "apt-get update -qq",
            "apt-get install -y -qq ffmpeg golang-go",
            "useradd -m -s /bin/bash gharunner",
            "usermod -aG docker gharunner",
            "runuser -u gharunner -- docker ps",
        ]

    def test_an_existing_account_is_not_created_again(self, tmp_path: pathlib.Path) -> None:
        assert self._run(tmp_path, id="exit 0")[0] == 0
        assert not any(line.startswith("useradd") for line in _asked(tmp_path))

    def test_a_package_that_asks_for_a_reboot_is_said(self, tmp_path: pathlib.Path) -> None:
        status, out, _ = self._run(tmp_path, reboot=True)
        assert (status, out) == (0, f"{runner_render.REBOOT_MARKER}\n{_READY}\n")

    def test_a_node_with_no_docker_is_refused_before_anything_is_installed(
        self, tmp_path: pathlib.Path
    ) -> None:
        status, out, err = self._run(tmp_path, systemctl="exit 3")
        assert (status, out) == (1, "")
        assert err == (
            "docker is not active; run the fleet node provision "
            "(MCPs scripts/host/fleet-node/provision.sh) first\n"
        )
        assert _asked(tmp_path) == []

    def test_an_account_that_cannot_reach_docker_fails_the_base(
        self, tmp_path: pathlib.Path
    ) -> None:
        status, out, _ = self._run(tmp_path, runuser="exit 1")
        assert (status, out) == (1, "")

    @pytest.mark.parametrize("package", ["ffmpeg;rm", "FFmpeg", "-y", "go lang", "pkg$"])
    def test_a_package_name_that_is_not_plain_is_refused(self, package: str) -> None:
        host = a_host()
        host["base"]["apt_packages"] = [package]
        with pytest.raises(ValueError, match="not a plain package name"):
            runner_render.render_base_script(host)


def _in_order(lines: Sequence[str], *blocks: Sequence[str]) -> bool:
    """Whether each block appears whole in ``lines``, each after the one before.

    Compared as text, since a renderer's line may carry a whole heredoc.

    Args:
        lines: The script's lines.
        *blocks: The blocks, in the order they must appear.

    Returns:
        True when every block was found in turn, each starting a line.
    """
    text = "\n" + "\n".join(lines) + "\n"
    at = 0
    for block in blocks:
        found = text.find("\n" + "\n".join(block) + "\n", at)
        if found < 0:
            return False
        at = found + len("\n".join(block)) + 1
    return True


class TestTheProvision:
    """provision.sh, composed from each section's renderer."""

    def test_the_tokens_are_exported_first_in_repository_order(
        self, tmp_path: pathlib.Path
    ) -> None:
        tokens = {"wagner-austin/MCPs": "TWO", "wagner-austin/API": "ONE"}
        script = runner_render.render_provision_script(_two_repos(tmp_path), tokens, RESERVED_CORES)
        lines = script.splitlines()
        assert lines[:2] == list(runner_render.SCRIPT_HEADER)
        assert lines[4:6] == ["export RUNNER_TOKEN_API='ONE'", "export RUNNER_TOKEN_MCPS='TWO'"]

    def test_a_token_that_cannot_be_embedded_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="registration token"):
            runner_render.render_provision_script(
                _two_repos(tmp_path), {"wagner-austin/API": "a'b"}, RESERVED_CORES
            )

    def test_a_host_reserving_its_runners_no_cores_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match=r"reserved_cores is 0; a runner host's roster file"):
            runner_render.render_provision(_two_repos(tmp_path), 0)

    def test_every_section_is_its_renderer_s_lines_in_order(self, tmp_path: pathlib.Path) -> None:
        spec = _two_repos(tmp_path)
        lines = runner_render.render_provision_script(spec, {}, RESERVED_CORES).splitlines()
        assets = spec["assets"]
        assert _in_order(
            lines,
            ci_clean.CI_CLEAN_SCRIPT.rstrip("\n").splitlines(),
            render_unit_file_lines(
                runner_render.CI_CLEAN_SERVICE_PATH,
                ci_clean.CI_CLEAN_SERVICE,
                announcement="hygiene unit set: ci-clean.service",
            ),
            render_unit_file_lines(
                runner_render.CI_CLEAN_TIMER_PATH,
                ci_clean.CI_CLEAN_TIMER,
                announcement="hygiene unit set: ci-clean.timer",
            ),
            ["systemctl enable --now ci-clean.timer"],
            runner_render.render_asset_lines(assets[1]),
            runner_render.render_asset_lines(assets[2]),
            runner_slice_render.render_slice_unit_lines(spec["ci_slice"]),
            render_ci_cpuset_lines(RESERVED_CORES),
            runner_reaper_render.render_reaper_lines(spec),
            runner_job_scope_render.render_job_scope_lines(),
            runner_action_cache.render_fill_lines(spec),
            runner_cache_adopt.render_adopt_lines(spec),
            *(runner_render.render_install_lines(install) for install in spec["installs"]),
        )
        assert lines[-1] == runner_render.FIRST_HYGIENE_PASS

    def test_a_manual_asset_appears_in_no_script(self, tmp_path: pathlib.Path) -> None:
        rendered = runner_render.render_provision(_two_repos(tmp_path), RESERVED_CORES)
        assert "rw-game" not in rendered["provision_script"]
        assert "rw-game" not in rendered["base_script"]


class TestTheAssetLines:
    """Each fetchable asset's lines, run for real."""

    def test_a_writable_directory_is_made_and_given_to_the_runners(
        self, tmp_path: pathlib.Path
    ) -> None:
        data = _assets(tmp_path)[2]
        ran = run_with_stand_ins(
            tmp_path,
            runner_render.render_asset_lines(data),
            {"chown": _recording(tmp_path, "chown")},
            options=("-eu",),
        )
        assert ran.returncode == 0, ran.stderr
        assert (tmp_path / "data").is_dir()
        assert _asked(tmp_path) == [f"chown gharunner:gharunner {bash_path(tmp_path)}/data"]

    def test_a_fetchable_asset_is_fetched_once_over_two_runs(self, tmp_path: pathlib.Path) -> None:
        lines = runner_render.render_asset_lines(_assets(tmp_path)[1])
        ran = run_with_stand_ins(tmp_path, [*lines, *lines], {}, options=("-eu",))
        assert ran.returncode == 0, ran.stderr
        assert _asked(tmp_path) == ["fetched"]
        assert (tmp_path / "convert.py").is_file()

    def test_a_path_that_cannot_be_embedded_is_refused(self, tmp_path: pathlib.Path) -> None:
        asset = _assets(tmp_path)[2]
        asset["path"] = "/da'ta"
        with pytest.raises(ValueError, match="asset path"):
            runner_render.render_asset_lines(asset)


class TestTheInstallLines:
    """One runner's lines: configured once, its PATH extended once, then started."""

    def _install_in(self, tmp_path: pathlib.Path) -> RunnerInstall:
        """An install whose directory is the case's ``runner``.

        Args:
            tmp_path: The case's directory.

        Returns:
            The install.
        """
        install = an_install()
        install["workdir"] = f"{bash_path(tmp_path)}/runner/_work"
        (tmp_path / "runner").mkdir()
        return install

    def _registration(self, install: RunnerInstall) -> list[str]:
        """The block that registers an unconfigured runner.

        Args:
            install: The install.

        Returns:
            Its four lines.
        """
        lines = runner_render.render_install_lines(install)
        root = runner_install.install_root_for(install)
        guard = lines.index(f"if [ ! -f '{root}/.runner' ]; then")
        return lines[guard : guard + 4]

    def test_an_unconfigured_runner_is_registered_with_its_token_replacing_its_name(
        self, tmp_path: pathlib.Path
    ) -> None:
        install = self._install_in(tmp_path)
        ran = run_with_stand_ins(
            tmp_path,
            ["export RUNNER_TOKEN_API='TOK'", *self._registration(install)],
            {"runuser": _recording(tmp_path, "runuser")},
            options=("-eu",),
        )
        assert ran.returncode == 0, ran.stderr
        assert _asked(tmp_path) == [
            f"runuser -u gharunner -- bash -c cd '{bash_path(tmp_path)}/runner' && ./config.sh "
            '--unattended --url https://github.com/wagner-austin/API --token "TOK" --name lavender '
            "--labels lavender --replace"
        ]

    def test_an_unconfigured_runner_without_its_token_stops_naming_it(
        self, tmp_path: pathlib.Path
    ) -> None:
        install = self._install_in(tmp_path)
        ran = run_with_stand_ins(
            tmp_path,
            self._registration(install),
            {"runuser": _recording(tmp_path, "runuser")},
            options=("-eu",),
        )
        assert ran.returncode != 0
        assert ran.stderr.rstrip("\n").endswith(
            "RUNNER_TOKEN_API: set RUNNER_TOKEN_API to a fresh registration token for "
            "wagner-austin/API; fleet-runners --rebuild and --onboard mint it with gh"
        )
        assert _asked(tmp_path) == []

    def test_a_configured_runner_needs_no_token(self, tmp_path: pathlib.Path) -> None:
        install = self._install_in(tmp_path)
        (tmp_path / "runner" / ".runner").touch()
        ran = run_with_stand_ins(
            tmp_path,
            self._registration(install),
            {"runuser": _recording(tmp_path, "runuser")},
            options=("-eu",),
        )
        assert ran.returncode == 0, ran.stderr
        assert _asked(tmp_path) == []

    def test_the_local_bin_lands_on_the_runner_path_once_over_two_runs(
        self, tmp_path: pathlib.Path
    ) -> None:
        install = self._install_in(tmp_path)
        [append] = [
            line for line in runner_render.render_install_lines(install) if line.startswith("grep")
        ]
        (tmp_path / "runner" / ".path").write_bytes(b"/usr/local/bin:/usr/bin:/bin\n")
        ran = run_with_stand_ins(tmp_path, [append, append], {}, options=("-eu",))
        assert ran.returncode == 0, ran.stderr
        assert (tmp_path / "runner" / ".path").read_bytes() == (
            f"/usr/local/bin:/usr/bin:/bin:{runner_render.LOCAL_BIN}\n"
        ).encode()

    def test_every_drop_in_is_written_before_the_unit_starts(self) -> None:
        """The unit exists after svc.sh install and gets its restart policy,
        its kill mode, its job hook, the action cache and the listener slice
        before it starts, since a new runner takes a queued job within
        seconds; once it runs, a unit an older provision started is moved
        onto the cache and into the slice."""
        install = an_install()
        root = runner_install.install_root_for(install)
        started = [f"(cd '{root}' && ./svc.sh start)"]
        assert not _in_order(
            runner_render.render_install_lines(install),
            started,
            runner_recovery.render_recovery_lines(install),
        )
        assert _in_order(
            runner_render.render_install_lines(install),
            [f"[ -f '{root}/.service' ] || (cd '{root}' && ./svc.sh install gharunner)"],
            runner_recovery.render_recovery_lines(install),
            runner_reaper_render.render_kill_mode_lines(install),
            runner_job_scope_render.render_job_hook_lines(install),
            runner_cache_adopt.render_cache_drop_in_lines(install),
            runner_slice_render.render_runner_slice_drop_in_lines(install),
            started,
            runner_cache_adopt.render_cache_adopt_now_lines(install),
            runner_slice_render.render_runner_slice_move_lines(install),
        )

    def test_a_directory_that_cannot_be_embedded_is_refused(self) -> None:
        install = an_install()
        install["workdir"] = "/home/it's/_work"
        with pytest.raises(ValueError, match="workdir"):
            runner_render.render_install_lines(install)


class TestTheRender:
    """What ``fleet-runners --render`` writes and prints."""

    def test_the_render_is_both_scripts_with_no_tokens_and_the_manual_steps(
        self, tmp_path: pathlib.Path
    ) -> None:
        spec = _two_repos(tmp_path)
        rendered = runner_render.render_provision(spec, RESERVED_CORES)
        assert rendered == runner_render.RenderedProvision(
            base_script=runner_render.render_base_script(spec),
            provision_script=runner_render.render_provision_script(spec, {}, RESERVED_CORES),
            manual_steps=[
                "PLACE BY HAND: /opt/corvis/rw-game/game-lib.jar -- the provenance jar. This "
                "asset can never be fetched by a script; place it, then re-run fleet-runners "
                "audit."
            ],
        )

    def test_a_host_with_no_manual_assets_has_no_steps(self, tmp_path: pathlib.Path) -> None:
        assert runner_render.manual_steps(a_host(assets=_assets(tmp_path)[1:])) == []
