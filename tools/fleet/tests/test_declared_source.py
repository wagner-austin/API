"""Source coordinates must survive the rolled read and publication, task c7575f45."""

from __future__ import annotations

import pathlib

from platform_core.json_utils import JSONObject, dump_json_str, load_json_str, narrow_json_to_dict

from fleet.core import _test_hooks, declared, queue, rolled
from tests._git_fixtures import committed_repository
from tests._queue_fakes import FakeQueue, queue_env
from tests.test_declared import machine_file, project_file


def test_publishes_source_coordinates_from_the_commit_instead_of_keys_or_working_files(
    tmp_path: pathlib.Path,
) -> None:
    """Hand-declared sources differ from their lane keys and later dirty files."""
    cases: tuple[tuple[str, str | None, str, JSONObject | None], ...] = (
        ("MCPs/packages/db", None, "", None),
        (
            "MCPs/packages/session-audit-linux",
            "https://github.com/wagner-austin/MCPs.git",
            "packages/session-audit",
            {"repository": "MCPs", "directory": "packages/session-audit"},
        ),
        (
            "root-lane",
            "ssh://git@github.com/wagner-austin/slime.git",
            "",
            {"repository": "slime", "directory": ""},
        ),
        (
            "tools/fleet",
            "git@github.com:wagner-austin/API.git",
            "tools/fleet",
            {"repository": "API", "directory": "tools/fleet"},
        ),
    )
    files = {"tools/fleet/registry/machines/sedona.json": machine_file()}
    expected: list[JSONObject] = []
    for key, remote, directory, source in cases:
        body = narrow_json_to_dict(load_json_str(project_file([])))
        body["source"] = (
            None
            if remote is None
            else {"remote": remote, "path": directory, "install": [], "companions": []}
        )
        files[f"tools/fleet/registry/projects/{key}.json"] = dump_json_str(body)
        expected.append(
            {"project": key, "requiredTags": [], "dispatchNodes": ["sedona"], "source": source}
        )
    repo = tmp_path / "registry"
    commit = committed_repository(repo, files, rolled.ROLLED_REF)
    for relative in files:
        if "/projects/" in relative:
            (repo / relative).write_text(project_file([]), encoding="utf-8")
    read = declared.read_declared(repo)
    assert not isinstance(read, str)
    endpoint = FakeQueue(
        [dump_json_str({"published": 4, "removed": 0, "refused": 0, "retired": []})]
    )
    _test_hooks.env = queue_env()
    _test_hooks.http_post = endpoint

    result = declared.publish_registry(queue.load_credentials(), read, identity={})

    assert result == {"published": 4, "removed": 0, "refused": 0, "retired": ()}
    assert endpoint.tools == ["dispatch_registry_publish"]
    assert endpoint.arguments == [
        {
            "projects": expected,
            "runners": ["fleet-node-sedona"],
            "commit": commit,
            "checkedAt": read["checked_at"],
        }
    ]
