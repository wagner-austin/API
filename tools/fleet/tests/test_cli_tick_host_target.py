"""The host-target lane of a fleet tick (MCPs board task 7cd6036a).

The MCPs hub runner's ``osrs-desktop-update`` row runs this lane every hour
on the machine holding the control duty, naming the project whose host
target it runs; the tick hands that project to rolled's ``fleet-host-target``
and acts as ``fleet-host-target-<that machine>``, the identity the run's
lease names. Every lane is named by its word here, as the hub row names it.
The tick suite's fixtures are written again at the size these cases need.
"""

from __future__ import annotations

import os
import pathlib
import sys
from collections.abc import Generator
from typing import Final

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.cli import host_target as host_target_cli
from fleet.cli import rolled as rolled_cli
from fleet.cli import tick
from fleet.core import _test_hooks
from tests._registry import SHIPPED_REGISTRY
from tests.conftest import FakeRun, failed, ok

#: 2026-09-29T00:40:00Z, the pinned clock of the tick here.
STARTED: Final = 1_790_642_400

#: The project the hourly row names: loki's OSRS desktops' update.
PROJECT: Final = "osrs-lab/manual-desktop"

#: The host-target runner's session on lacrymal: the version-5 UUID of
#: ``fleet-agent/host-target/lacrymal`` in the URL namespace, computed with
#: Python's uuid module outside the code under test and written down.
LACRYMAL_SESSION: Final = "d920d33e-d1bd-5182-93ad-56819ca9b16e"

#: The same for ``fleet-agent/host-target/diphtheria``.
DIPHTHERIA_SESSION: Final = "cabea6d2-b807-5ab5-b1aa-254703c14db5"

#: The netboot lane's flags, which a host-target tick refuses.
NETBOOT: Final = {
    "--netboot-release": "/r/MCPs",
    "--nodejs": "/usr/bin/node",
    "--served-root": "/srv/corvis-netboot",
    "--env-file": "/w/.env",
}


def _lacrymal() -> str:
    """The control duty's holder, as the hostname hook reports it.

    Returns:
        ``lacrymal``.
    """
    return "lacrymal"


def _diphtheria() -> str:
    """Another machine, as the hostname hook reports it.

    Returns:
        ``diphtheria``.
    """
    return "diphtheria"


@pytest.fixture(name="on_lacrymal", autouse=True)
def _on_lacrymal() -> None:
    """Run every tick here as lacrymal unless a case says otherwise."""
    _test_hooks.hostname = _lacrymal


def _launch(root: pathlib.Path, label: str, session: str) -> tuple[str, ...]:
    """What a host-target plan hands the launcher.

    Args:
        root: The API checkout.
        label: The runner's label.
        session: The runner's session.

    Returns:
        The launcher's flags, then the host target's identity and project.
    """
    return (
        "--repo-root",
        str(root),
        "--agent",
        "fleet-host-target",
        "--",
        "--agent",
        label,
        "--session",
        session,
        "--project",
        PROJECT,
    )


class TestTheHostTargetLane:
    def test_it_runs_the_projects_target_as_the_control_machines_runner(self) -> None:
        root = pathlib.Path("/home/corvis/PROJECTS/API")

        plan = tick.plan_tick(root, tick.require_lane("host-target"), None, {"--project": PROJECT})

        assert plan == tick.TickPlan(
            arguments=_launch(root, "fleet-host-target-lacrymal", LACRYMAL_SESSION),
            stem="fleet-host-target-lacrymal",
            header="host-target osrs-lab/manual-desktop",
        )

    def test_another_machine_holding_the_duty_acts_as_its_own_runner(self) -> None:
        _test_hooks.hostname = _diphtheria
        root = pathlib.Path("/api")

        plan = tick.plan_tick(root, tick.require_lane("host-target"), None, {"--project": PROJECT})

        assert plan["arguments"] == _launch(
            root, "fleet-host-target-diphtheria", DIPHTHERIA_SESSION
        )
        assert plan["stem"] == "fleet-host-target-diphtheria"

    def test_its_word_is_its_lane_and_a_launch_runs_the_host_target_module(self) -> None:
        lane = tick.require_lane("host-target")

        assert lane.value == "host-target"
        assert rolled_cli.require_module("fleet-host-target") == "fleet.cli.host_target"

    def test_without_a_project_it_is_refused(self) -> None:
        with pytest.raises(ValueError) as refused:
            tick.plan_tick(pathlib.Path("/api"), tick.require_lane("host-target"), None, {})

        assert str(refused.value) == "FLEET_TICK_USAGE: the host-target lane needs --project"

    def test_naming_a_node_is_refused_since_the_registry_names_it(self) -> None:
        with pytest.raises(ValueError) as refused:
            tick.plan_tick(
                pathlib.Path("/api"),
                tick.require_lane("host-target"),
                "loki",
                {"--project": PROJECT},
            )

        assert str(refused.value) == (
            "FLEET_TICK_USAGE: the host-target lane takes no --node; the node a host target "
            "runs on is its project's registry entry's"
        )

    def test_a_netboot_flag_is_refused(self) -> None:
        with pytest.raises(ValueError) as refused:
            tick.plan_tick(
                pathlib.Path("/api"),
                tick.require_lane("host-target"),
                None,
                {"--project": PROJECT, "--nodejs": "/usr/bin/node"},
            )

        assert str(refused.value) == (
            "FLEET_TICK_USAGE: the host-target lane takes no --nodejs; only a netboot tick arms"
        )

    @pytest.mark.parametrize(
        ("word", "node", "places"),
        [
            ("hub", None, {}),
            ("node", "sedona", {}),
            ("sessions", "ziggurat", {}),
            ("netboot", None, NETBOOT),
            ("research", None, {}),
        ],
    )
    def test_every_other_lane_refuses_a_project(
        self, word: str, node: str | None, places: dict[str, str]
    ) -> None:
        with pytest.raises(ValueError) as refused:
            tick.plan_tick(
                pathlib.Path("/api"),
                tick.require_lane(word),
                node,
                {**places, "--project": PROJECT},
            )

        assert str(refused.value) == (
            f"FLEET_TICK_USAGE: the {word} lane takes no --project; only a host-target tick "
            "runs a project's target"
        )


class TestTheHostTargetReadsWhatTheTickHands:
    def test_every_flag_the_tick_hands_is_one_the_host_target_parses(self) -> None:
        """The launcher's split and the real verb's parser, over the shipped registry.

        osrs-lab/recorder declares no host target, so the verb refuses it by
        name once it has parsed every flag and loaded the registry, before
        anything is fetched; a flag it did not read would be refused first.
        """
        plan = tick.plan_tick(
            pathlib.Path("/api"),
            tick.require_lane("host-target"),
            None,
            {"--project": "osrs-lab/recorder"},
        )
        launcher, passthrough = rolled_cli.split_command_line(plan["arguments"])
        nothing = FakeRun([])
        _test_hooks.run = nothing

        with pytest.raises(AppError) as raised:
            host_target_cli.main([*passthrough, "--config", str(SHIPPED_REGISTRY)])

        assert launcher == {"--repo-root": "/api", "--agent": "fleet-host-target"}
        assert raised.value.code is FleetErrorCode.HOST_TARGET_UNDECLARED
        assert "project 'osrs-lab/recorder' declares no host_target" in raised.value.message
        assert nothing.calls == []


class TestOneHostTargetTick:
    @pytest.fixture(name="clock")
    def _clock(self) -> Generator[list[int], None, None]:
        """A clock answering the start, then 12 s later."""
        readings = [STARTED, STARTED + 12]

        def now() -> int:
            return readings.pop(0)

        _test_hooks.now = now
        yield readings

    @pytest.fixture(name="api_root")
    def _api_root(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """An API checkout holding only the credentials file a tick reads."""
        root = tmp_path / "PROJECTS" / "API"
        env_file = root / pathlib.PurePosixPath(tick.ENV_FILE)
        env_file.parent.mkdir(parents=True)
        env_file.write_text("FLEET_MCP_API_KEY=fleet-key\n", encoding="utf-8")
        return root

    def _argv(self, root: pathlib.Path, logs: pathlib.Path) -> list[str]:
        return [
            "--api-root",
            str(root),
            "--log-directory",
            str(logs),
            "--lane",
            "host-target",
            "--project",
            PROJECT,
        ]

    def test_a_tick_runs_the_host_target_with_the_credentials_and_logs_under_its_label(
        self, api_root: pathlib.Path, tmp_path: pathlib.Path, clock: list[int]
    ) -> None:
        logs = tmp_path / "logs"
        runner = FakeRun([ok("host target update of osrs-lab/manual-desktop ... passed\n")])
        _test_hooks.run = runner

        assert tick.main(self._argv(api_root, logs)) == 0

        assert runner.calls == [
            (
                sys.executable,
                "-m",
                "fleet.cli.rolled",
                *_launch(api_root, "fleet-host-target-lacrymal", LACRYMAL_SESSION),
            )
        ]
        assert runner.set_env == [(("FLEET_MCP_API_KEY", "fleet-key"),)]
        assert runner.timeouts == [39 * 60 + 30]
        log = logs / "fleet-host-target-lacrymal-2026-09-29.log"
        assert log.read_text(encoding="utf-8").splitlines() == [
            "TICK START 2026-09-29T00:40:00+00:00 host-target osrs-lab/manual-desktop "
            f"task-pid {os.getpid()}",
            "host target update of osrs-lab/manual-desktop ... passed",
            "TICK EXIT 0 2026-09-29T00:40:12+00:00",
        ]
        assert clock == []

    def test_a_failed_target_is_the_ticks_exit_status(
        self, api_root: pathlib.Path, tmp_path: pathlib.Path, clock: list[int]
    ) -> None:
        logs = tmp_path / "logs"
        _test_hooks.run = FakeRun([failed(2, "make: *** [Makefile:40: update] Error 1\n")])

        assert tick.main(self._argv(api_root, logs)) == 2

        log = logs / "fleet-host-target-lacrymal-2026-09-29.log"
        assert log.read_text(encoding="utf-8").splitlines()[1:] == [
            "make: *** [Makefile:40: update] Error 1",
            "TICK EXIT 2 2026-09-29T00:40:12+00:00",
        ]
        assert clock == []
