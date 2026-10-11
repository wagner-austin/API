"""The registry directory is read back into the one workspace document (MCPs board task 4a78c916).

Real directories under ``tmp_path``, read through the production reader: the
shipped registry assembles to a workspace that decodes, each project's key is
its path, and every malformed layout is refused by name before a decoder sees
it.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError, require_dict

from fleet.contracts.workspace import decode_fleet_workspace
from fleet.core.workspace_files import project_key, read_workspace_document
from tests._registry import SHIPPED_REGISTRY, shipped_document, write_registry

#: Three projects whose keys exercise the path rule: one at a repository's
#: root beside a directory of the same name, and two nested under it.
PROJECTS: JSONObject = {
    "MCPs/packages/db": {"marker": "db"},
    "MCPs": {"marker": "root"},
    "MCPs/fleet-mcp": {"marker": "fleet-mcp"},
}


def _document() -> JSONObject:
    """A registry document carrying :data:`PROJECTS` and one setting.

    Returns:
        A fresh document.
    """
    return {
        "nodes": {"sedona": {"marker": "node"}},
        "not_dispatchable": {},
        "ledger": "runs/ledger.jsonl",
        "projects": dict(PROJECTS),
    }


def test_the_shipped_registry_assembles_into_a_workspace_that_decodes() -> None:
    document = shipped_document()

    workspace = decode_fleet_workspace(document)

    projects = require_dict(document, "projects")
    assert (
        len(workspace["projects"])
        == len(projects)
        == len(list((SHIPPED_REGISTRY / "projects").rglob("*.json")))
    )
    assert "MCPs/packages/wiki-check" in workspace["projects"]
    assert set(document) - {"projects"} == {
        "nodes",
        "not_dispatchable",
        "node_local_resources",
        "data_paths",
        "node_serve_seconds",
        "node_poll_seconds",
        "ledger",
        "feed",
        "leases",
    }


def test_each_project_is_keyed_by_its_path_in_key_order(tmp_path: pathlib.Path) -> None:
    registry = write_registry(tmp_path / "registry", _document())

    document = read_workspace_document(registry)

    assert document == {**_document(), "projects": dict(sorted(PROJECTS.items()))}
    keys = list(require_dict(document, "projects"))
    assert keys == ["MCPs", "MCPs/fleet-mcp", "MCPs/packages/db"]


def test_a_key_is_the_path_under_projects_with_forward_slashes(tmp_path: pathlib.Path) -> None:
    projects = tmp_path / "projects"
    assert project_key(projects, projects / "MCPs" / "packages" / "db.json") == "MCPs/packages/db"
    assert project_key(projects, projects / "slime.json") == "slime"


def test_a_file_holding_no_object_is_refused_naming_it(tmp_path: pathlib.Path) -> None:
    registry = write_registry(tmp_path / "registry", _document())
    (registry / "projects" / "MCPs.json").write_text("[1, 2]\n", encoding="utf-8")

    with pytest.raises(JSONTypeError, match=r"MCPs\.json must hold a JSON object, got list$"):
        read_workspace_document(registry)


def test_a_nodes_file_holding_anything_but_its_two_objects_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    registry = write_registry(tmp_path / "registry", _document())
    (registry / "nodes.json").write_text('{"nodes": {}}\n', encoding="utf-8")

    with pytest.raises(
        JSONTypeError, match=r"nodes\.json must hold exactly nodes, not_dispatchable, got nodes$"
    ):
        read_workspace_document(registry)


def test_a_settings_file_holding_nodes_or_projects_is_refused(tmp_path: pathlib.Path) -> None:
    registry = write_registry(tmp_path / "registry", _document())
    (registry / "settings.json").write_text(
        '{"ledger": "runs/ledger.jsonl", "projects": {}, "nodes": {}}\n', encoding="utf-8"
    )

    with pytest.raises(
        JSONTypeError,
        match=r"settings\.json holds nodes, projects, which belong in nodes\.json or under "
        r"projects/$",
    ):
        read_workspace_document(registry)


def test_a_registry_with_no_project_file_is_refused(tmp_path: pathlib.Path) -> None:
    registry = write_registry(tmp_path / "registry", {**_document(), "projects": {}})

    with pytest.raises(JSONTypeError, match=r"projects declares no project$"):
        read_workspace_document(registry)
