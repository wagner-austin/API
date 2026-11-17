"""The inspect lane of a fleet tick (MCPs board task 3e9c99af).

The MCPs hub runner's ``fleet-inspect`` row runs this lane on the machine
holding the control duty, with the installed fleet-audit release, the node
binary and the workspace whose registry names the roster; the tick hands
those to the agent and acts as ``fleet-inspect-<that machine>``.
"""

from __future__ import annotations

import pathlib
from typing import Final

import pytest

from fleet.cli import tick
from fleet.contracts.dispatch import DispatchLane
from fleet.core import _test_hooks

#: The installed fleet-audit release, as an install record names one.
RELEASE: Final = "/home/corvis/.local/share/corvis/releases/fleet-audit/5d0e6b1c/MCPs"

#: The row's three inspect values, as the hub runner fills them on lacrymal.
INSPECT: Final = {
    "--inspect-release": RELEASE,
    "--nodejs": "/usr/bin/node",
    "--workspace": "/home/corvis/PROJECTS/MCPs",
}

#: Those values as the command line carries them, in the agent's flag order.
INSPECT_ARGS: Final = tuple(token for pair in INSPECT.items() for token in pair)

#: The inspect runner's session on lacrymal: the version-5 UUID of
#: ``fleet-agent/inspect/lacrymal``, computed once and written down.
LACRYMAL_SESSION: Final = "890cfa06-9b85-563a-987c-330b6882d150"


def _lacrymal() -> str:
    """The control duty's holder, as the hostname hook reports it.

    Returns:
        ``lacrymal``.
    """
    return "lacrymal"


@pytest.fixture(name="on_lacrymal", autouse=True)
def _on_lacrymal() -> None:
    """Run every tick here as lacrymal."""
    _test_hooks.hostname = _lacrymal


def _agent_tail(root: pathlib.Path) -> tuple[str, ...]:
    """What every inspect plan hands the launcher before its lane's own flags.

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
        "fleet-inspect-lacrymal",
        "--session",
        LACRYMAL_SESSION,
        "--repo-root",
        str(root),
        "--lane",
        "inspect",
    )


class TestTheInspectLanes:
    def test_the_inspect_lane_runs_the_agent_as_this_machine_s_inspect_runner(self) -> None:
        root = pathlib.Path("/home/corvis/PROJECTS/API")

        assert tick.plan_tick(root, tick.Lane.INSPECT, None, INSPECT) == tick.TickPlan(
            arguments=(*_agent_tail(root), *INSPECT_ARGS),
            stem="fleet-inspect-lacrymal",
            header="inspect",
        )

    def test_the_inspect_announce_lane_checks_the_runner_in_into_its_own_log(self) -> None:
        root = pathlib.Path("/home/corvis/PROJECTS/API")

        assert tick.plan_tick(root, tick.Lane.INSPECT_ANNOUNCE, None, {}) == tick.TickPlan(
            arguments=(*_agent_tail(root), "--announce"),
            stem="fleet-inspect-lacrymal",
            header="inspect-announce",
        )

    def test_the_runner_s_identity_is_derived_from_the_lane_and_the_machine(self) -> None:
        assert tick.runner_identity(DispatchLane.INSPECT, "lacrymal") == (
            "fleet-inspect-lacrymal",
            LACRYMAL_SESSION,
        )

    def test_the_inspect_lane_names_each_flag_it_lacks(self) -> None:
        with pytest.raises(ValueError) as refused:
            tick.plan_tick(
                pathlib.Path("/api"), tick.Lane.INSPECT, None, {"--inspect-release": RELEASE}
            )

        assert str(refused.value) == (
            "FLEET_TICK_USAGE: the inspect lane needs --nodejs, --workspace"
        )

    def test_the_inspect_lane_naming_a_node_is_refused(self) -> None:
        with pytest.raises(ValueError) as refused:
            tick.plan_tick(pathlib.Path("/api"), tick.Lane.INSPECT, "pendragon", INSPECT)

        assert str(refused.value) == "FLEET_TICK_USAGE: the inspect lane takes no --node"

    def test_the_inspect_lane_refuses_a_netboot_flag(self) -> None:
        given = {**INSPECT, "--served-root": "/srv/corvis-netboot"}

        with pytest.raises(ValueError) as refused:
            tick.plan_tick(pathlib.Path("/api"), tick.Lane.INSPECT, None, given)

        assert str(refused.value) == (
            "FLEET_TICK_USAGE: the inspect lane takes no --served-root; only a netboot tick arms"
        )

    @pytest.mark.parametrize(
        ("lane", "node"),
        [
            (tick.Lane.HUB, None),
            (tick.Lane.NETBOOT, None),
            (tick.Lane.INSPECT_ANNOUNCE, None),
            (tick.Lane.NODE, "sedona"),
        ],
    )
    def test_every_other_lane_refuses_an_inspect_only_flag(
        self, lane: tick.Lane, node: str | None
    ) -> None:
        with pytest.raises(ValueError) as refused:
            tick.plan_tick(pathlib.Path("/api"), lane, node, {"--workspace": "/w"})

        assert str(refused.value) == (
            f"FLEET_TICK_USAGE: the {lane.value} lane takes no --workspace; "
            "only an inspect tick inspects"
        )
