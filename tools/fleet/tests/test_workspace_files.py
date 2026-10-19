"""The registry directory is read back into the one workspace document (MCPs board task 4a78c916).

Real directories under ``tmp_path``, read through the production reader: the
shipped registry assembles to a workspace that decodes, each project's key is
its path, and every malformed layout is refused by name before a decoder sees
it.
"""

from __future__ import annotations

import pathlib
from typing import Final

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

#: The most lines one registry file may hold, the operator's file-size rule
#: ("no monithic files. 400 - 600 lines", 2026-10-11) that split the
#: 2252-line fleet.json into this directory.
MAX_FILE_LINES: Final = 600


def _document() -> JSONObject:
    """A registry document carrying :data:`PROJECTS` and one setting.

    Returns:
        A fresh document.
    """
    return {
        "machines": {"sedona": {"marker": "machine"}},
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
    assert sorted(workspace["machines"]) == sorted(
        path.stem for path in (SHIPPED_REGISTRY / "machines").glob("*.json")
    )
    assert set(document) - {"projects"} == {
        "machines",
        "identity",
        "node_local_resources",
        "data_paths",
        "node_serve_seconds",
        "node_poll_seconds",
        "ledger",
        "feed",
        "leases",
    }


def test_every_shipped_registry_file_is_at_most_six_hundred_lines() -> None:
    lines = {
        path.relative_to(SHIPPED_REGISTRY).as_posix(): len(
            path.read_text(encoding="utf-8").splitlines()
        )
        for path in sorted(SHIPPED_REGISTRY.rglob("*"))
        if path.is_file()
    }

    named = [*(SHIPPED_REGISTRY / "machines").glob("*.json")]
    named += (SHIPPED_REGISTRY / "projects").rglob("*.json")
    assert {"nodes.json", "settings.json", "machines/sedona.json"} < set(lines)
    assert len(lines) == 2 + len(named)
    assert {name: count for name, count in lines.items() if count > MAX_FILE_LINES} == {}


def test_the_single_file_registry_it_replaced_is_gone() -> None:
    beside = {path.name for path in SHIPPED_REGISTRY.parent.iterdir()}

    assert "registry" in beside
    assert "fleet.json" not in beside


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


def test_each_machine_is_keyed_by_its_file_s_stem(tmp_path: pathlib.Path) -> None:
    machines: JSONObject = {"sedona": {"marker": "s"}, "lavender-wsl": {"marker": "w"}}
    registry = write_registry(tmp_path / "registry", {**_document(), "machines": machines})

    assert list(require_dict(read_workspace_document(registry), "machines")) == [
        "lavender-wsl",
        "sedona",
    ]


def test_a_settings_file_holding_machines_or_projects_is_refused(tmp_path: pathlib.Path) -> None:
    registry = write_registry(tmp_path / "registry", _document())
    (registry / "settings.json").write_text(
        '{"ledger": "runs/ledger.jsonl", "projects": {}, "machines": {}}\n', encoding="utf-8"
    )

    with pytest.raises(
        JSONTypeError,
        match=r"settings\.json holds machines, projects, which belong under machines/ or "
        r"projects/$",
    ):
        read_workspace_document(registry)


@pytest.mark.parametrize("key", ["machines", "projects"])
def test_a_registry_with_no_file_under_a_directory_is_refused(
    tmp_path: pathlib.Path, key: str
) -> None:
    registry = write_registry(tmp_path / "registry", {**_document(), key: {}})

    with pytest.raises(JSONTypeError, match=rf"registry[/\\]{key} declares nothing$"):
        read_workspace_document(registry)
