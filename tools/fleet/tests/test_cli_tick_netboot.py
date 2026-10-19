"""The netboot lane of a fleet tick (MCPs board task 83102d9d).

The MCPs hub runner's ``fleet-netboot`` row runs this lane on the machine
holding the netboot duty, with the installed netboot release, the node
binary, the served root and the workspace env file; the tick hands those to
the agent and acts as ``fleet-netboot-<that machine>``. Split from
``test_cli_tick.py`` at the file-size ceiling; the clock and credentials
fixtures are that suite's, written again here at the size they need.
"""

from __future__ import annotations

import os
import pathlib
import sys
from collections.abc import Generator
from typing import Final

import pytest

from fleet.cli import tick
from fleet.contracts.dispatch import DispatchLane
from fleet.core import _test_hooks
from tests.conftest import FakeRun, ok

#: 2026-09-29T00:40:00Z, the pinned clock of the tick here.
STARTED: Final = 1_790_642_400

#: The installed netboot release, as diphtheria's install record names it.
RELEASE: Final = "/home/corvis/.local/share/corvis/netboot-releases/7fcc96d6/MCPs"

#: The row's four netboot values, as the hub runner fills them on diphtheria.
NETBOOT: Final = {
    "--netboot-release": RELEASE,
    "--nodejs": "/usr/bin/node",
    "--served-root": "/srv/corvis-netboot",
    "--env-file": "/home/corvis/PROJECTS/MCPs/.env",
}

#: Those values as the command line carries them, in the agent's flag order.
NETBOOT_ARGS: Final = tuple(token for pair in NETBOOT.items() for token in pair)

#: The netboot runner's session on diphtheria: the version-5 UUID of
#: ``fleet-agent/netboot/diphtheria``, computed once and written down.
DIPHTHERIA_SESSION: Final = "c0c7f738-1eac-5ed4-bde8-8cb72e418245"


def _diphtheria() -> str:
    """The netboot duty's holder, as the hostname hook reports it.

    Returns:
        ``diphtheria``.
    """
    return "diphtheria"


@pytest.fixture(name="on_diphtheria", autouse=True)
def _on_diphtheria() -> None:
    """Run every tick here as diphtheria."""
    _test_hooks.hostname = _diphtheria


def _agent_tail(root: pathlib.Path) -> tuple[str, ...]:
    """What every netboot plan hands the launcher before its lane's own flags.

    Args:
        root: The API checkout.

    Returns:
        The launcher's flags and the agent's identity, repo and lane.
    """
    return (
        "--repo-root",
        str(root),
        "--agent",
        "fleet-agent",
        "--",
        "--agent",
        "fleet-netboot-diphtheria",
        "--session",
        DIPHTHERIA_SESSION,
        "--repo-root",
        str(root),
        "--lane",
        "netboot",
    )


class TestTheNetbootLanes:
    def test_the_netboot_lane_runs_the_agent_as_this_machines_netboot_runner(self) -> None:
        root = pathlib.Path("/home/corvis/PROJECTS/API")

        assert tick.plan_tick(root, tick.Lane.NETBOOT, None, NETBOOT) == tick.TickPlan(
            arguments=(*_agent_tail(root), *NETBOOT_ARGS),
            stem="fleet-netboot-diphtheria",
            header="netboot",
        )

    def test_the_netboot_announce_lane_checks_the_runner_in_into_its_own_log(self) -> None:
        root = pathlib.Path("/home/corvis/PROJECTS/API")

        assert tick.plan_tick(root, tick.Lane.NETBOOT_ANNOUNCE, None, {}) == tick.TickPlan(
            arguments=(*_agent_tail(root), "--announce"),
            stem="fleet-netboot-diphtheria",
            header="netboot-announce",
        )

    def test_each_machine_has_its_own_netboot_runner_identity(self) -> None:
        assert tick.runner_identity(DispatchLane.NETBOOT, "diphtheria") == (
            "fleet-netboot-diphtheria",
            DIPHTHERIA_SESSION,
        )
        assert tick.runner_identity(DispatchLane.NETBOOT, "lacrymal") == (
            "fleet-netboot-lacrymal",
            "fabb438a-f8ae-56da-8155-b9dd5ac979f8",
        )

    def test_the_netboot_lane_names_each_flag_it_lacks(self) -> None:
        partial = {"--netboot-release": RELEASE, "--served-root": "/srv/corvis-netboot"}

        with pytest.raises(ValueError) as refused:
            tick.plan_tick(pathlib.Path("/api"), tick.Lane.NETBOOT, None, partial)

        assert str(refused.value) == (
            "FLEET_TICK_USAGE: the netboot lane needs --nodejs, --env-file"
        )

    def test_the_netboot_lane_naming_a_node_is_refused(self) -> None:
        with pytest.raises(ValueError) as refused:
            tick.plan_tick(pathlib.Path("/api"), tick.Lane.NETBOOT, "lavender", NETBOOT)

        assert str(refused.value) == "FLEET_TICK_USAGE: the netboot lane takes no --node"

    @pytest.mark.parametrize(
        ("lane", "node"),
        [
            (tick.Lane.NETBOOT_ANNOUNCE, None),
            (tick.Lane.HUB, None),
            (tick.Lane.SESSIONS, "ziggurat"),
            (tick.Lane.NODE, "sedona"),
        ],
    )
    def test_every_other_lane_refuses_the_netboot_flags(
        self, lane: tick.Lane, node: str | None
    ) -> None:
        with pytest.raises(ValueError) as refused:
            tick.plan_tick(pathlib.Path("/api"), lane, node, {"--nodejs": "/usr/bin/node"})

        assert str(refused.value) == (
            f"FLEET_TICK_USAGE: the {lane.value} lane takes no --nodejs; only the netboot lane arms"
        )


class TestOneNetbootTick:
    @pytest.fixture(name="clock")
    def _clock(self) -> Generator[list[int], None, None]:
        """A clock answering the start, then 12 s later."""
        readings = [STARTED, STARTED + 12]

        def now() -> int:
            return readings.pop(0)

        _test_hooks.now = now
        yield readings

    def test_a_tick_hands_the_rows_values_to_the_agent_and_logs_under_its_label(
        self, tmp_path: pathlib.Path, clock: list[int]
    ) -> None:
        root = tmp_path / "PROJECTS" / "API"
        env_file = root / pathlib.PurePosixPath(tick.ENV_FILE)
        env_file.parent.mkdir(parents=True)
        env_file.write_text("$env:FLEET_MCP_API_KEY = 'fleet-key'\n", encoding="utf-8")
        logs = tmp_path / "logs"
        runner = FakeRun([ok("netboot lane empty\n")])
        _test_hooks.run = runner
        argv = ["--api-root", str(root), "--log-directory", str(logs), "--lane", "netboot"]

        assert tick.main([*argv, *NETBOOT_ARGS]) == 0

        assert runner.calls == [
            (sys.executable, "-m", "fleet.cli.rolled", *_agent_tail(root), *NETBOOT_ARGS)
        ]
        # The tick hands the agent the machine's queue credentials and
        # nothing else: the tailnet key stays in the env file the row names,
        # for the arm alone to read.
        assert runner.set_env == [(("FLEET_MCP_API_KEY", "fleet-key"),)]
        log = logs / "fleet-netboot-diphtheria-2026-09-29.log"
        assert log.read_text(encoding="utf-8").splitlines() == [
            f"TICK START 2026-09-29T00:40:00+00:00 netboot task-pid {os.getpid()}",
            "netboot lane empty",
            "TICK EXIT 0 2026-09-29T00:40:12+00:00",
        ]
        assert clock == []
