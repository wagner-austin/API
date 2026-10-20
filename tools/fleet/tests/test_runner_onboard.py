"""Onboarding, end to end against fakes at the command seam.

Every provisioning byte the onboarder would ship is captured through the
same ``_test_hooks.run`` seam production uses, so these tests assert the
actual payload -- the whole host's provision with the new install in it and
the new repository's token at its head -- not a summary of it.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import _test_hooks, remote, runner_audit, runner_onboard, runner_render
from tests._runner_fixtures import a_host, an_install, no_dropped_jobs
from tests.conftest import FakeRun, failed, ok

#: The install onboarding tree-bot onto lavender creates.
_TREE_BOT = RunnerInstall(
    repo="wagner-austin/tree-bot",
    runner_name="lavender",
    service="actions.runner.wagner-austin-tree-bot.lavender.service",
    workdir="/home/gharunner/actions-runner-tree-bot-1/_work",
    labels=["lavender"],
)


def _host() -> HostRunnerSpec:
    """A lavender-shaped host with one pre-existing install, API's.

    Returns:
        The spec.
    """
    return a_host([an_install("wagner-austin/API")])


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
        plan = runner_onboard.plan_onboard(_host(), "wagner-austin/Tree-Bot")
        assert plan == runner_onboard.OnboardPlan(
            repo="wagner-austin/Tree-Bot",
            install=RunnerInstall(
                repo="wagner-austin/Tree-Bot",
                runner_name="lavender",
                service="actions.runner.wagner-austin-Tree-Bot.lavender.service",
                workdir="/home/gharunner/actions-runner-tree-bot-1/_work",
                labels=["lavender"],
            ),
        )

    def test_a_repo_already_on_the_host_is_refused_naming_its_runners(self) -> None:
        spec = a_host(
            [an_install("wagner-austin/API"), an_install("wagner-austin/API", "lavender-2", slot=2)]
        )
        with pytest.raises(AppError) as fault:
            runner_onboard.plan_onboard(spec, "wagner-austin/API")
        assert fault.value.code is FleetErrorCode.RUNNER_ALREADY_ONBOARDED
        assert fault.value.message.startswith(
            "wagner-austin/API already has 2 install(s) on lavender (lavender, lavender-2); "
        )

    @pytest.mark.parametrize("repo", ["no-slash", "a/b/c", "owner/", "/name"])
    def test_a_malformed_repo_is_refused(self, repo: str) -> None:
        with pytest.raises(ValueError, match="owner/repo"):
            runner_onboard.plan_onboard(_host(), repo)


class TestOnboard:
    """The whole act, the payload captured byte for byte."""

    def test_the_host_s_provision_runs_with_the_new_install_and_the_audit_covers_it(
        self,
    ) -> None:
        spec = _host()
        grown = a_host([an_install("wagner-austin/API"), _TREE_BOT])
        runner = FakeRun(
            [
                ok("TOKENABC123\n"),  # gh mint
                ok(""),  # send the provision
                ok("done\n"),  # run it
                ok(""),  # audit: send
                ok(_clean_transcript(grown)),  # audit: run
                *no_dropped_jobs(grown),  # audit: the dropped-job row's reads
            ]
        )
        _test_hooks.run = runner
        plan, findings = runner_onboard.onboard(spec, "wagner-austin/tree-bot")
        assert plan["install"] == _TREE_BOT
        assert spec == grown
        assert [finding["ok"] for finding in findings] == [True] * (
            len(runner_audit.expected_checks(grown)) + 1
        )
        assert runner.calls[0][4] == (
            "repos/wagner-austin/tree-bot/actions/runners/registration-token"
        )
        path = "/home/corvis/fleet/runner-payloads/fleet-onboard-tree-bot.sh"
        assert runner.stdin[1] == runner_render.render_provision_script(
            grown, {"wagner-austin/tree-bot": "TOKENABC123"}
        ).encode("utf-8")
        assert tuple(runner.calls[2][-4:]) == (*remote.ROOT_BASH, path)
        assert runner.timeouts[2] == runner_render.STAGE_TIMEOUT_SECONDS

    def test_a_provision_that_fails_raises_with_its_words(self) -> None:
        _test_hooks.run = FakeRun(
            [ok("TOK1\n"), ok(""), failed(1, "Http response code: NotFound from 'POST'")]
        )
        with pytest.raises(AppError) as fault:
            runner_onboard.onboard(_host(), "wagner-austin/x")
        assert fault.value.code is FleetErrorCode.DISPATCH_FAILED
        assert "Http response code: NotFound" in fault.value.message

    def test_an_unreachable_audit_names_the_staged_state(self) -> None:
        _test_hooks.run = FakeRun(
            [ok("TOK1\n"), ok(""), ok("done\n"), failed(255, "No route to host")]
        )
        with pytest.raises(AppError) as fault:
            runner_onboard.onboard(_host(), "wagner-austin/x")
        assert fault.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert fault.value.message.startswith("post-onboard audit could not reach lavender: ")
        assert "its roster entry is staged" in fault.value.message
