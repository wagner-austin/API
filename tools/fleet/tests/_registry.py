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

from fleet.core.workspace_files import (
    NODES_FILE,
    NODES_KEYS,
    PROJECT_SUFFIX,
    PROJECTS_DIRECTORY,
    PROJECTS_KEY,
    SETTINGS_FILE,
    read_workspace_document,
)

#: The registry this commit ships, beside the tests.
SHIPPED_REGISTRY: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parents[1] / "registry"


def shipped_document() -> JSONObject:
    """The shipped registry, assembled.

    Returns:
        The workspace document every runner of this commit reads.
    """
    return read_workspace_document(SHIPPED_REGISTRY)


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
        document: A workspace document, split as the shipped registry is: its
            node keys to ``nodes.json``, each project to its own file, and the
            rest to ``settings.json``.

    Returns:
        The directory, ready to pass as ``--config``.

    Raises:
        TypeError: When ``document``'s projects are not an object.
    """
    projects = document[PROJECTS_KEY]
    if not isinstance(projects, dict):
        raise TypeError(f"projects must be an object, got {type(projects).__name__}")
    if directory.exists():
        shutil.rmtree(directory)
    _write(directory / NODES_FILE, {k: v for k, v in document.items() if k in NODES_KEYS})
    _write(
        directory / SETTINGS_FILE,
        {k: v for k, v in document.items() if k not in NODES_KEYS and k != PROJECTS_KEY},
    )
    for key, entry in projects.items():
        _write(directory / PROJECTS_DIRECTORY / f"{key}{PROJECT_SUFFIX}", entry)
    return directory


__all__ = ["SHIPPED_REGISTRY", "shipped_document", "write_registry"]
