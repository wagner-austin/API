"""Reading the workspace document every command starts from.

Shared by all seven entry points, which is the point: they cannot disagree
about where the cluster is or which ledger records it, because they all read
the same file through the same function.

The document is located relative to the process, and the paths inside it
relative to the document. That is what lets a workspace live next to the runs
it describes and be used from any working directory.
"""

from __future__ import annotations

import pathlib

from platform_core import cli_args
from platform_core.json_utils import load_json_str

from hpc3.contracts.workspace import Workspace, decode_workspace
from hpc3.core import _test_hooks as core_hooks

CONFIG_FLAG = "--config"
"""The one flag every command shares."""


def load_workspace(parsed: dict[str, str]) -> Workspace:
    """Read and validate the workspace named on the command line.

    Args:
        parsed: Flags already read from the command line.

    Returns:
        The validated workspace, with its ledger path resolved against the
        document's own directory.

    Raises:
        ValueError: If ``--config`` was not given, or the document declares a
            root that is not an absolute POSIX path.
        JSONTypeError: If the document is not a valid workspace.
        AppError: If a project declares a GPU this cluster does not carry.
    """
    path = pathlib.Path(cli_args.require_flag(parsed, CONFIG_FLAG)).resolve()
    raw = core_hooks.read_bytes(path).decode("utf-8")
    return decode_workspace(load_json_str(raw), config_dir=path.parent)


__all__ = ["CONFIG_FLAG", "load_workspace"]
