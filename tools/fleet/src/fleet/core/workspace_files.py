"""The fleet's registry directory, read back into the one workspace document.

The registry was one file, ``tools/fleet/fleet.json``, until it reached 2252
lines on 2026-10-11 (MCPs board task 4a78c916; the operator, that day: "we
usually wan tro ensure a modualr, clear separationor cocnerna ,no monithic
files. 400 - 600 lines"). It is now a directory of three parts, each owned by
a different reader:

* ``nodes.json``, the ``nodes`` and ``not_dispatchable`` objects: the only part
  MCPs ``fleet-mcp``'s ``fleet-enroll`` writes and its roster and standard
  checks read.
* ``settings.json``, every other fleet-wide setting (the record paths, the
  serve and poll seconds, the node-local resources).
* ``projects/<key>.json``, ONE FILE PER PROJECT, whose key is its path under
  ``projects/`` without the suffix: ``projects/MCPs/packages/db.json`` is
  ``MCPs/packages/db`` and ``projects/MCPs.json`` is ``MCPs``. A key is a
  path, so no two files can declare one project, and a file holds one
  project, so none grows as projects register; MCPs' declarations alone were
  about 1300 lines of the old file.

:func:`read_workspace_document` is the one reader: it assembles the three
into the document :func:`fleet.contracts.workspace.decode_fleet_workspace`
has always decoded, so nothing past it knows the registry was split.
"""

from __future__ import annotations

import pathlib
from typing import Final

from platform_core.json_utils import JSONObject, JSONTypeError, load_json_str

from fleet.core import _test_hooks

#: The file holding the nodes and the rows excluded from dispatch.
NODES_FILE: Final = "nodes.json"

#: The two objects :data:`NODES_FILE` holds, and nothing else.
NODES_KEYS: Final = frozenset({"nodes", "not_dispatchable"})

#: The file holding every other fleet-wide setting.
SETTINGS_FILE: Final = "settings.json"

#: The directory of per-project files, and the suffix each carries.
PROJECTS_DIRECTORY: Final = "projects"
PROJECT_SUFFIX: Final = ".json"

#: The key the assembled document carries the projects under.
PROJECTS_KEY: Final = "projects"


def _read_object(path: pathlib.Path) -> JSONObject:
    """Read one registry file, which must hold a JSON object.

    Args:
        path: The file.

    Returns:
        Its object.

    Raises:
        JSONTypeError: When it holds anything but an object.
        OSError: When it cannot be read.
    """
    value = load_json_str(_test_hooks.read_text(path))
    if not isinstance(value, dict):
        raise JSONTypeError(f"{path} must hold a JSON object, got {type(value).__name__}")
    return value


def project_key(projects: pathlib.Path, path: pathlib.Path) -> str:
    """The project a per-project file declares, read from where it sits.

    Args:
        projects: The registry's ``projects`` directory.
        path: One file under it.

    Returns:
        Its path under ``projects`` without :data:`PROJECT_SUFFIX`, with
        forward slashes on every platform.
    """
    return path.relative_to(projects).as_posix().removesuffix(PROJECT_SUFFIX)


def read_workspace_document(directory: pathlib.Path) -> JSONObject:
    """Assemble the registry directory into the one workspace document.

    Args:
        directory: The registry directory, ``tools/fleet/registry``.

    Returns:
        The nodes and settings objects merged, with every per-project file
        under :data:`PROJECTS_KEY` by its key, in key order.

    Raises:
        JSONTypeError: When a file holds no object, :data:`NODES_FILE` holds
            anything but :data:`NODES_KEYS`, :data:`SETTINGS_FILE` holds a
            key that belongs to the nodes or the projects, or no project file
            exists.
        OSError: When a file cannot be read.
    """
    nodes = _read_object(directory / NODES_FILE)
    if set(nodes) != NODES_KEYS:
        raise JSONTypeError(
            f"{directory / NODES_FILE} must hold exactly {', '.join(sorted(NODES_KEYS))}, "
            f"got {', '.join(sorted(nodes))}"
        )
    settings = _read_object(directory / SETTINGS_FILE)
    misplaced = sorted(set(settings) & (NODES_KEYS | {PROJECTS_KEY}))
    if misplaced:
        raise JSONTypeError(
            f"{directory / SETTINGS_FILE} holds {', '.join(misplaced)}, which belong in "
            f"{NODES_FILE} or under {PROJECTS_DIRECTORY}/"
        )
    projects_root = directory / PROJECTS_DIRECTORY
    paths = sorted(projects_root.rglob(f"*{PROJECT_SUFFIX}"))
    if not paths:
        raise JSONTypeError(f"{projects_root} declares no project")
    projects: JSONObject = {project_key(projects_root, path): _read_object(path) for path in paths}
    return {**nodes, **settings, PROJECTS_KEY: dict(sorted(projects.items()))}


__all__ = [
    "NODES_FILE",
    "NODES_KEYS",
    "PROJECTS_DIRECTORY",
    "PROJECTS_KEY",
    "PROJECT_SUFFIX",
    "SETTINGS_FILE",
    "project_key",
    "read_workspace_document",
]
