"""Onboarding, end to end against fakes at the command seam.

Every provisioning byte the onboarder would ship is captured through the
same ``_test_hooks.run`` seam production uses, so these tests assert the
actual scripts -- the token line, the config.cmd invocation, the tool-cache
seeding -- not a summary of them.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import _test_hooks, runner_audit, runner_onboard, runner_render
from tests.conftest import FakeRun, failed, ok


def _host() -> HostRunnerSpec:
    """A lavender-shaped host with one pre-existing install.

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
            )
        ],
        assets=[],
    )


def _clean_transcript(spec: HostRunnerSpec) -> str:
    """One OK line per expected check for the given spec state.

    Args:
        spec: The host, in whatever install state the test has built.

    Returns:
        The transcript.
    """
    return (
        "\n".join(f"CHECK {check['check_id']} OK" for check in runner_audit.expected_checks(spec))
        + "\n"
    )


class TestPlan:
    """plan_onboard's refusals and its derived convention."""

    def test_the_plan_follows_the_fleet_convention(self) -> None:
        plan = runner_onboard.plan_onboard(
            _host(), "wagner-austin/tree-bot", sides=("wsl", "windows")
        )
        wsl, windows = plan["installs"]
        assert wsl["side"] == "wsl"
        assert wsl["runner_name"] == "lavender-wsl"
        assert wsl["workdir"] == "/home/gharunner/actions-runner-tree-bot-1/_work"
        assert wsl["service"] == "actions.runner.wagner-austin-tree-bot.lavender-wsl.service"
        assert windows["side"] == "windows"
        assert windows["runner_name"] == "lavender"
        assert windows["workdir"] == "C:/actions-runner-tree-bot/_work"
        assert windows["service"] == "actions.runner.wagner-austin-tree-bot.lavender"

    def test_one_side_yields_one_install(self) -> None:
        plan = runner_onboard.plan_onboard(_host(), "wagner-austin/x", sides=("wsl",))
        assert [i["side"] for i in plan["installs"]] == ["wsl"]

    def test_a_repo_already_on_the_host_is_refused(self) -> None:
        with pytest.raises(AppError) as fault:
            runner_onboard.plan_onboard(_host(), "wagner-austin/API", sides=("wsl",))
        assert fault.value.code is FleetErrorCode.RUNNER_ALREADY_ONBOARDED

    @pytest.mark.parametrize("repo", ["no-slash", "a/b/c", "owner/"])
    def test_a_malformed_repo_is_refused(self, repo: str) -> None:
        with pytest.raises(ValueError, match="owner/repo"):
            runner_onboard.plan_onboard(_host(), repo, sides=("wsl",))

    def test_empty_sides_are_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            runner_onboard.plan_onboard(_host(), "wagner-austin/x", sides=())

    def test_an_unknown_side_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown side"):
            runner_onboard.plan_onboard(_host(), "wagner-austin/x", sides=("linux",))


class TestMintToken:
    """The gh seam."""

    def test_a_minted_token_is_returned(self) -> None:
        runner = FakeRun([ok("ABCDEF123\n")])
        _test_hooks.run = runner
        assert runner_onboard.mint_registration_token("wagner-austin/x") == "ABCDEF123"
        argv = tuple(runner.calls[0])
        assert argv == (
            "gh",
            "api",
            "-X",
            "POST",
            "repos/wagner-austin/x/actions/runners/registration-token",
            "-q",
            ".token",
        )

    def test_a_gh_failure_is_a_local_auth_condition(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "HTTP 403: no admin rights")])
        with pytest.raises(AppError) as fault:
            runner_onboard.mint_registration_token("wagner-austin/x")
        assert fault.value.code is FleetErrorCode.RUNNER_TOKEN_UNAVAILABLE
        assert "403" in fault.value.message

    def test_an_empty_token_is_refused(self) -> None:
        _test_hooks.run = FakeRun([ok("\n")])
        with pytest.raises(AppError) as fault:
            runner_onboard.mint_registration_token("wagner-austin/x")
        assert fault.value.code is FleetErrorCode.RUNNER_TOKEN_UNAVAILABLE

    def test_a_token_that_could_break_quoting_is_refused(self) -> None:
        _test_hooks.run = FakeRun([ok("AB'; rm -rf /; '\n")])
        with pytest.raises(AppError) as fault:
            runner_onboard.mint_registration_token("wagner-austin/x")
        assert fault.value.code is FleetErrorCode.RUNNER_TOKEN_UNAVAILABLE
        assert "non-alphanumeric" in fault.value.message


class TestOnboard:
    """The whole act, scripts captured byte for byte."""

    def test_both_sides_provision_roster_grows_and_audit_runs(self) -> None:
        spec = _host()
        grown = _host()
        grown["installs"].extend(
            runner_onboard.plan_onboard(
                _host(), "wagner-austin/tree-bot", sides=("wsl", "windows")
            )["installs"]
        )
        transcript = _clean_transcript(grown)
        runner = FakeRun(
            [
                ok("TOKENABC123\n"),  # gh mint
                ok(""),  # send wsl payload
                ok(""),  # send wsl driver
                ok("done"),  # run wsl driver
                ok(""),  # send windows payload
                ok("done"),  # run windows payload
                ok(""),  # audit: send driver
                ok(transcript),  # audit: run driver
            ]
        )
        _test_hooks.run = runner
        plan, findings = runner_onboard.onboard(
            spec,
            "wagner-austin/tree-bot",
            sides=("wsl", "windows"),
            python_versions=("3.11.9",),
        )
        assert [i["repo"] for i in spec["installs"]].count("wagner-austin/tree-bot") == 2
        assert all(finding["ok"] for finding in findings)
        wsl_install, windows_install = plan["installs"]
        # The shipped payloads are asserted byte for byte against the same
        # renderers production composes from -- the claim under test is the
        # composition (shebang, strict mode, token line, then the install),
        # while the renderers' own content has its own tests.
        expected_wsl = "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "export RUNNER_TOKEN_TREE_BOT='TOKENABC123'",
                *runner_render.render_wsl_install_lines(wsl_install),
            ]
        ).encode("utf-8")
        assert runner.stdin[1] == expected_wsl
        expected_windows = "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12",
                "$env:RUNNER_TOKEN_TREE_BOT = 'TOKENABC123'",
                *runner_render.render_windows_install_lines(windows_install),
                *runner_render.render_windows_python_toolcache_lines(windows_install, ("3.11.9",)),
            ]
        ).encode("utf-8")
        assert runner.stdin[4] == expected_windows

    def test_no_python_versions_means_no_toolcache_lines(self) -> None:
        spec = _host()
        grown = _host()
        grown["installs"].extend(
            runner_onboard.plan_onboard(_host(), "wagner-austin/x", sides=("windows",))["installs"]
        )
        runner = FakeRun([ok("TOK1\n"), ok(""), ok("done"), ok(""), ok(_clean_transcript(grown))])
        _test_hooks.run = runner
        plan, _ = runner_onboard.onboard(spec, "wagner-austin/x", sides=("windows",))
        expected = "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12",
                "$env:RUNNER_TOKEN_X = 'TOK1'",
                *runner_render.render_windows_install_lines(plan["installs"][0]),
            ]
        ).encode("utf-8")
        assert runner.stdin[1] == expected

    def test_an_unreachable_audit_names_the_staged_state(self) -> None:
        spec = _host()
        _test_hooks.run = FakeRun(
            [
                ok("TOK1\n"),
                ok(""),
                ok("done"),
                failed(255, "No route to host"),
            ]
        )
        with pytest.raises(AppError) as fault:
            runner_onboard.onboard(spec, "wagner-austin/x", sides=("windows",))
        assert fault.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert "staged" in fault.value.message


class TestWindowsToWslPath:
    """The one path translation onboarding needs."""

    def test_a_drive_path_translates(self) -> None:
        assert (
            runner_onboard._windows_to_wsl_path("C:/fleet/stage/x.sh") == "/mnt/c/fleet/stage/x.sh"
        )

    @pytest.mark.parametrize("path", ["relative/x", "/posix/x", "C:x", ""])
    def test_anything_else_is_refused(self, path: str) -> None:
        with pytest.raises(ValueError, match="drive path"):
            runner_onboard._windows_to_wsl_path(path)
