"""Reading the workspace document every command starts from.

Shared by all five entry points, which is the point: they cannot disagree
about where the nodes are or which ledger records them, because they read the
same file through the same function.

The document is located relative to the process and the paths inside it
relative to the document, so a workspace can live beside the records it
describes and be used from any working directory. That resolution happens here
rather than in the contract, because a contract that resolved paths could not
be decoded from a string in a test without inventing a directory for it.
"""

from __future__ import annotations

import pathlib

from platform_core import cli_args
from platform_core.json_utils import load_json_str

from fleet.contracts.workspace import FleetWorkspace, decode_fleet_workspace
from fleet.core import _test_hooks

CONFIG_FLAG = "--config"
"""The one flag every command shares."""


class LoadedWorkspace:
    """A decoded workspace and the three paths it points at.

    A small class rather than a TypedDict because its three path properties
    are DERIVED -- each resolves a declared string against the document's own
    directory -- and a TypedDict of already-resolved paths would either need a
    second constructor function or would let a caller build one whose paths
    disagree with its workspace.

    Attributes:
        workspace: The validated document.
        directory: The directory the document was read from, which its
            relative paths resolve against.
    """

    workspace: FleetWorkspace
    directory: pathlib.Path

    def __init__(self, workspace: FleetWorkspace, directory: pathlib.Path) -> None:
        """Hold a decoded workspace and where it came from.

        Args:
            workspace: The validated document.
            directory: The directory it was read from.
        """
        self.workspace = workspace
        self.directory = directory

    def _resolve(self, declared: str) -> pathlib.Path:
        """Resolve one declared path against the document's directory.

        Args:
            declared: The path as the document spells it.

        Returns:
            An absolute path. An already-absolute declaration is returned
            unchanged, which is what ``/`` on a Path does.
        """
        return self.directory / declared

    @property
    def ledger(self) -> pathlib.Path:
        """Absolute path to the append-only dispatch record.

        Returns:
            The resolved path.
        """
        return self._resolve(self.workspace["ledger"])

    @property
    def feed(self) -> pathlib.Path:
        """Absolute path to the append-only event stream.

        Returns:
            The resolved path.
        """
        return self._resolve(self.workspace["feed"])

    @property
    def leases(self) -> pathlib.Path:
        """Absolute path to the live lease file.

        Returns:
            The resolved path.
        """
        return self._resolve(self.workspace["leases"])


def load_workspace(parsed: dict[str, str]) -> LoadedWorkspace:
    """Read and validate the workspace named on the command line.

    Args:
        parsed: Flags already read from the command line.

    Returns:
        The validated workspace and its resolved record paths.

    Raises:
        ValueError: If ``--config`` was not given.
        JSONTypeError: If the document is not a valid workspace.
        OSError: If the document cannot be read. Propagated rather than
            translated: the message names the path and the reason, which is
            the whole diagnostic.
    """
    path = pathlib.Path(cli_args.require_flag(parsed, CONFIG_FLAG)).resolve()
    value = load_json_str(_test_hooks.read_text(path))
    return LoadedWorkspace(decode_fleet_workspace(value), path.parent)


__all__ = ["CONFIG_FLAG", "LoadedWorkspace", "load_workspace"]
