"""The fleet's registry directory, read back into the one workspace document.

The registry was one file, ``tools/fleet/fleet.json``, until it reached 2252
lines on 2026-10-11 (MCPs board task 4a78c916; the operator, that day: "we
usually wan tro ensure a modualr, clear separationor cocnerna ,no monithic
files. 400 - 600 lines"). It is now a directory of three parts:

* ``machines/<name>.json``, ONE FILE PER MACHINE of the fleet, whose name is
  its stem: the fleet's one roster (:mod:`fleet.contracts.machine`, MCPs
  board task e447de72), which MCPs ``fleet-mcp``'s ``fleet-enroll`` writes a
  file of and every fleet reader in either repository reads.
* ``settings.json``, every other fleet-wide setting (the record paths, the
  serve and poll seconds, the node-local resources, and under ``identity``
  the fleet's ssh key, Cloudflare zone and account and UCI VPN profile
  digest, which MCPs ``fleet-mcp`` reads and this package does not).
* ``projects/<key>.json``, ONE FILE PER PROJECT, whose key is its path under
  ``projects/`` without the suffix: ``projects/MCPs/packages/db.json`` is
  ``MCPs/packages/db`` and ``projects/MCPs.json`` is ``MCPs``. A key is a
  path, so no two files can declare one project, and a file holds one
  project, so none grows as projects register; MCPs' declarations alone were
  about 1300 lines of the old file.

:func:`read_workspace_document` is the one reader: it assembles the three
into the document :func:`fleet.contracts.workspace.decode_fleet_workspace`
decodes, so nothing past it knows the registry is split.
"""

from __future__ import annotations

import pathlib
from typing import Final

from platform_core.json_utils import JSONObject, JSONTypeError, load_json_str

from fleet.core import _test_hooks

#: The directory of per-machine files, and the key the document carries them under.
MACHINES_DIRECTORY: Final = "machines"
MACHINES_KEY: Final = "machines"

#: The file holding every other fleet-wide setting.
SETTINGS_FILE: Final = "settings.json"

#: The directory of per-project files, and the key the document carries them under.
PROJECTS_DIRECTORY: Final = "projects"
PROJECTS_KEY: Final = "projects"

#: The suffix every machine and project file carries.
JSON_SUFFIX: Final = ".json"


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
        Its path under ``projects`` without :data:`JSON_SUFFIX`, with
        forward slashes on every platform.
    """
    return path.relative_to(projects).as_posix().removesuffix(JSON_SUFFIX)


def _read_named(paths: list[pathlib.Path], root: pathlib.Path) -> JSONObject:
    """Read a directory's files into one object keyed by each file's key.

    Args:
        paths: The files, sorted.
        root: The directory they sit under, whose relative path is the key.

    Returns:
        Each file's object under :func:`project_key` of it, in key order.

    Raises:
        JSONTypeError: When no file exists, or one holds no object.
        OSError: When a file cannot be read.
    """
    if not paths:
        raise JSONTypeError(f"{root} declares nothing")
    named: JSONObject = {project_key(root, path): _read_object(path) for path in paths}
    return dict(sorted(named.items()))


def read_workspace_document(directory: pathlib.Path) -> JSONObject:
    """Assemble the registry directory into the one workspace document.

    Args:
        directory: The registry directory, ``tools/fleet/registry``.

    Returns:
        The settings object, with every machine file under
        :data:`MACHINES_KEY` by its name and every project file under
        :data:`PROJECTS_KEY` by its key, each in key order.

    Raises:
        JSONTypeError: When a file holds no object, :data:`SETTINGS_FILE`
            holds a key that belongs to the machines or the projects, or no
            machine or project file exists.
        OSError: When a file cannot be read.
    """
    settings = _read_object(directory / SETTINGS_FILE)
    misplaced = sorted(set(settings) & {MACHINES_KEY, PROJECTS_KEY})
    if misplaced:
        raise JSONTypeError(
            f"{directory / SETTINGS_FILE} holds {', '.join(misplaced)}, which belong under "
            f"{MACHINES_DIRECTORY}/ or {PROJECTS_DIRECTORY}/"
        )
    machines_root = directory / MACHINES_DIRECTORY
    projects_root = directory / PROJECTS_DIRECTORY
    return {
        **settings,
        MACHINES_KEY: _read_named(sorted(machines_root.glob(f"*{JSON_SUFFIX}")), machines_root),
        PROJECTS_KEY: _read_named(sorted(projects_root.rglob(f"*{JSON_SUFFIX}")), projects_root),
    }


__all__ = [
    "JSON_SUFFIX",
    "MACHINES_DIRECTORY",
    "MACHINES_KEY",
    "PROJECTS_DIRECTORY",
    "PROJECTS_KEY",
    "SETTINGS_FILE",
    "project_key",
    "read_workspace_document",
]
