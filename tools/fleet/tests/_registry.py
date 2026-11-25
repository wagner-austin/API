"""Registry directories for tests: the shipped one, and ones written from a document.

The registry is a directory (:mod:`fleet.core.workspace_files`, MCPs board task
4a78c916), so a test that used to write one ``fleet.json`` writes the
directory instead, split exactly as the shipped one is, and a test that read
the shipped file reads the shipped directory through the one reader.
"""

from __future__ import annotations

import pathlib
import shutil
from typing import Final

from platform_core.json_utils import JSONObject, JSONValue, dump_json_str

from fleet.contracts.node import LiveLoad, NodeConfig, NodeState
from fleet.contracts.workspace import FleetWorkspace, decode_fleet_workspace
from fleet.core.workspace_files import (
    JSON_SUFFIX,
    MACHINES_DIRECTORY,
    MACHINES_KEY,
    PROJECTS_DIRECTORY,
    PROJECTS_KEY,
    SETTINGS_FILE,
    read_workspace_document,
)
from tests._egress_fixtures import REACHED

#: The registry this commit ships, beside the tests.
SHIPPED_REGISTRY: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parents[1] / "registry"


def shipped_document() -> JSONObject:
    """The shipped registry, assembled.

    Returns:
        The workspace document every runner of this commit reads.
    """
    return read_workspace_document(SHIPPED_REGISTRY)


def shipped_workspace() -> FleetWorkspace:
    """The shipped registry, decoded, as the admission cases weigh it.

    Returns:
        The decoded workspace.
    """
    return decode_fleet_workspace(shipped_document())


def idle_state(node: NodeConfig, *, free_ram_gb: float = 24.0) -> NodeState:
    """A node with nothing of the fleet's on it.

    The shipped-registry cases (admission, platform tags and the cores each
    project's worker is charged) weigh the registry against this one state;
    MCPs board task 5cd7f739 lifted it here from the two files that each
    carried a copy.

    Args:
        node: The node's declaration.
        free_ram_gb: Its free memory; 24 GB, memory to spare, unless a case
            measures a node that has less.

    Returns:
        Its state with that memory free, no live run, no reservation and no
        pressure.
    """
    return NodeState(
        host=node["host"],
        free_ram_gb=free_ram_gb,
        free_disk_gb=100.0,
        live=LiveLoad(runs=0, workers=0, cores=0, ram_gb=0.0),
        ci_slice=None,
        reserved_cores=0,
        action_cache=None,
        cpu_pressure_pct=0.0,
        io_pressure_pct=0.0,
        check_cpu_quota=None,
        on_battery=False,
        egress=REACHED,
        gpus=(),
    )


def _write(path: pathlib.Path, value: JSONValue) -> None:
    """Write one registry file.

    Args:
        path: The file.
        value: Its JSON value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_json_str(value) + "\n", encoding="utf-8")


def write_registry(directory: pathlib.Path, document: JSONObject) -> pathlib.Path:
    """Write a workspace document out as a registry directory.

    A registry already there is replaced whole, so a fixture that rewrites
    the shared registry with fewer projects leaves none of the old ones
    behind, as rewriting the one file did.

    Args:
        directory: Where the registry goes; created, or replaced.
        document: A workspace document, split as the shipped registry is:
            each machine to its own file, each project to its own file, and
            the rest to ``settings.json``.

    Returns:
        The directory, ready to pass as ``--config``.

    Raises:
        TypeError: When ``document``'s machines or projects are not an object.
    """
    if directory.exists():
        shutil.rmtree(directory)
    split = ((MACHINES_KEY, MACHINES_DIRECTORY), (PROJECTS_KEY, PROJECTS_DIRECTORY))
    for key, subdirectory in split:
        named = document[key]
        if not isinstance(named, dict):
            raise TypeError(f"{key} must be an object, got {type(named).__name__}")
        for name, entry in named.items():
            _write(directory / subdirectory / f"{name}{JSON_SUFFIX}", entry)
    _write(
        directory / SETTINGS_FILE,
        {k: v for k, v in document.items() if k not in (MACHINES_KEY, PROJECTS_KEY)},
    )
    return directory


__all__ = [
    "SHIPPED_REGISTRY",
    "idle_state",
    "shipped_document",
    "shipped_workspace",
    "write_registry",
]
