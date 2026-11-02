"""Onboarding: the plan the fleet convention derives, and the roster entry it grows.

Onboarding touches no network (MCPs board task 9aba868d): the host's NixOS
generation lays the new runner once the roster is rendered and switched to,
and ``fleet-runners --register`` registers it, so these hold the plan and
the mutation alone.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import runner_onboard
from tests._runner_fixtures import a_host, an_install

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
    """The act: the install appended to the host's entry, or nothing changed."""

    def test_the_install_is_appended_to_the_host_s_entry(self) -> None:
        spec = _host()
        plan = runner_onboard.onboard(spec, "wagner-austin/tree-bot")
        assert plan["install"] == _TREE_BOT
        assert spec == a_host([an_install("wagner-austin/API"), _TREE_BOT])

    def test_a_refused_onboard_leaves_the_entry_as_it_was(self) -> None:
        spec = _host()
        with pytest.raises(AppError) as fault:
            runner_onboard.onboard(spec, "wagner-austin/API")
        assert fault.value.code is FleetErrorCode.RUNNER_ALREADY_ONBOARDED
        assert spec == _host()
