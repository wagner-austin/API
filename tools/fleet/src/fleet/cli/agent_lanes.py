"""Where each placed lane of ``fleet-agent`` runs its commands from.

Two lanes run a command from an installed sealed release on the machine
holding their duty, and name that place on their command line: the netboot
lane, which arms and retires network installs (MCPs board task 83102d9d),
and the inspect lane, which reads one roster node with one catalog probe
(MCPs board task 3e9c99af). Their flags, the refusals of those flags on any
other lane, and the reading of each place are this module's; split out of
:mod:`fleet.cli.agent` when the inspect lane took that file past the
600-line ceiling, so :mod:`fleet.cli.tick`, which hands a lane exactly its
own flags, reads the same table.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Final

from platform_core import cli_args

from fleet.contracts.dispatch import DispatchLane
from fleet.core import fleet_inspect, netboot_arm

NETBOOT_RELEASE_FLAG: Final = "--netboot-release"
NODEJS_FLAG: Final = "--nodejs"
SERVED_ROOT_FLAG: Final = "--served-root"
ENV_FILE_FLAG: Final = "--env-file"
INSPECT_RELEASE_FLAG: Final = "--inspect-release"
WORKSPACE_FLAG: Final = "--workspace"

#: The flags a netboot tick places its arms with, which every other lane
#: refuses rather than ignores.
NETBOOT_FLAGS: Final = (NETBOOT_RELEASE_FLAG, NODEJS_FLAG, SERVED_ROOT_FLAG, ENV_FILE_FLAG)

#: The flags an inspect tick places its inspections with: the installed
#: fleet-audit release, the node binary and the MCPs workspace whose registry
#: names the roster.
INSPECT_FLAGS: Final = (INSPECT_RELEASE_FLAG, NODEJS_FLAG, WORKSPACE_FLAG)

#: Each lane that names where its commands run from: its flags, which it
#: requires; why every other lane refuses those of them it does not read
#: itself; and why it refuses ``--node``, since the machine its job names is
#: the job's target, never where it runs, so it claims every job of the lane.
LANE_PLACES: Final[Mapping[DispatchLane, tuple[tuple[str, ...], str, str]]] = {
    DispatchLane.NETBOOT: (
        NETBOOT_FLAGS,
        "only a netboot tick arms",
        "it claims every arm, and the machine an arm names is its target",
    ),
    DispatchLane.INSPECT: (
        INSPECT_FLAGS,
        "only an inspect tick inspects",
        "it claims every inspection, and the machine an inspection names is its target",
    ),
}

#: Every flag a placed lane reads, each once.
PLACE_FLAGS: Final = tuple(
    dict.fromkeys(flag for flags, _, _ in LANE_PLACES.values() for flag in flags)
)


def netboot_place(parsed: Mapping[str, str], registry: pathlib.Path) -> netboot_arm.NetbootPlace:
    """Read where a netboot tick's arms run from and write to.

    Args:
        parsed: The parsed flags.
        registry: The registry directory this tick loaded.

    Returns:
        The installed release, the node binary, the served root, the
        workspace env file and the registry.

    Raises:
        ValueError: When any of :data:`NETBOOT_FLAGS` is absent.
    """
    return netboot_arm.NetbootPlace(
        release=pathlib.Path(cli_args.require_flag(parsed, NETBOOT_RELEASE_FLAG)),
        nodejs=cli_args.require_flag(parsed, NODEJS_FLAG),
        served_root=pathlib.Path(cli_args.require_flag(parsed, SERVED_ROOT_FLAG)),
        env_file=pathlib.Path(cli_args.require_flag(parsed, ENV_FILE_FLAG)),
        registry=registry,
    )


def inspect_place(parsed: Mapping[str, str]) -> fleet_inspect.InspectPlace:
    """Read where an inspect tick's inspections run from.

    Args:
        parsed: The parsed flags.

    Returns:
        The installed fleet-audit release, the node binary and the MCPs
        workspace.

    Raises:
        ValueError: When any of :data:`INSPECT_FLAGS` is absent.
    """
    return fleet_inspect.InspectPlace(
        release=pathlib.Path(cli_args.require_flag(parsed, INSPECT_RELEASE_FLAG)),
        nodejs=cli_args.require_flag(parsed, NODEJS_FLAG),
        workspace=pathlib.Path(cli_args.require_flag(parsed, WORKSPACE_FLAG)),
    )


__all__ = [
    "ENV_FILE_FLAG",
    "INSPECT_FLAGS",
    "INSPECT_RELEASE_FLAG",
    "LANE_PLACES",
    "NETBOOT_FLAGS",
    "NETBOOT_RELEASE_FLAG",
    "NODEJS_FLAG",
    "PLACE_FLAGS",
    "SERVED_ROOT_FLAG",
    "WORKSPACE_FLAG",
    "inspect_place",
    "netboot_place",
]
