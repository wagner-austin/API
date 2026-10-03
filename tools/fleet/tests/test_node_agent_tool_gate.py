"""A missing tool refuses only the projects that need it (MCPs board task 939ec5c7, A3).

pendragon claimed nothing on every tick of 2026-10-02 until 02:48Z, logging
``NODE_TOOL_MISSING: ... ffmpeg -- grandma-api's check converts real audio
files through ffmpeg``, while tools/fleet, libs/platform_core and
tools/maketools waited in the queue with no tag between them and that node.
This replays that tick: pendragon and those four projects exactly as the real
``fleet.json`` declares them, a toolchain answer with every build tool and no
ffmpeg, and room for a run. One pass of the runner must ask the queue for
every project but grandma-api, with tags that keep grandma-api's job away
from it, and must say which tool it lacks without closing the node.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import dump_json_str, load_json_str

from fleet.cli import node_agent
from fleet.contracts.node import encode_node_config
from fleet.contracts.project import encode_project_config
from fleet.contracts.workspace import decode_fleet_workspace
from fleet.core import _test_hooks
from tests._node_agent_fixtures import _credentials_in_env, sourced_document
from tests._queue_fakes import FakeQueue
from tests._toolchain_fixtures import LAVENDER_2026_09_23
from tests.conftest import DEMO_PROJECT, FakeRun, ok

__all__ = ["_credentials_in_env"]

#: The workspace this package ships, read for pendragon and the projects
#: that waited for it.
REAL_WORKSPACE = pathlib.Path(__file__).resolve().parents[1] / "fleet.json"

#: The project that needs ffmpeg, and the tag-free ones that queued behind it,
#: in the name order a claim lists the projects it fits.
GRANDMA = "services/grandma-api"
WAITING = ("libs/platform_core", "tools/fleet", "tools/maketools")

#: pendragon with room for a run: 9.0 GB free of its 11.7 against its 2.5 GB
#: owner reservation.
PENDRAGON_ROOM = "free_ram_gb=9.0\nfree_disk_gb=400.0\n"


@pytest.fixture(name="pendragon_config")
def _pendragon_config(config_path: pathlib.Path) -> pathlib.Path:
    """The shared workspace with pendragon as its node and the real projects beside the demo.

    Args:
        config_path: The shared workspace document, clock pinned.

    Returns:
        The same path, rewritten.
    """
    real = decode_fleet_workspace(load_json_str(REAL_WORKSPACE.read_text("utf-8")))
    document = sourced_document((("npm", "ci"),))
    document["nodes"] = {"pendragon": encode_node_config(real["nodes"]["pendragon"])}
    projects = document["projects"]
    assert isinstance(projects, dict)
    for name in (GRANDMA, *WAITING):
        project = encode_project_config(real["projects"][name])
        project["source"] = None
        projects[name] = project
    config_path.write_text(dump_json_str(document), encoding="utf-8")
    return config_path


def test_pendragon_without_ffmpeg_asks_the_queue_for_every_project_but_grandma_api(
    pendragon_config: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    runner = FakeRun([ok(""), ok(PENDRAGON_ROOM), ok(""), ok(LAVENDER_2026_09_23)])
    _test_hooks.run = runner
    endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
    _test_hooks.http_post = endpoint

    argv = ["--config", str(pendragon_config), node_agent.NODE_FLAG, "pendragon"]
    with caplog.at_level("INFO"):
        assert node_agent.main(argv) == 0

    assert endpoint.tools == ["dispatch_list", "dispatch_claim"]
    claim = endpoint.arguments[1]
    assert claim["tags"] == ["windows"]
    assert claim["projects"] == [DEMO_PROJECT, *WAITING]
    messages = [record.getMessage() for record in caplog.records]
    assert not any("NODE_TOOL_MISSING" in message for message in messages)
    assert any(
        message.startswith("pendragon claims without the tag of every tool it lacks: ffmpeg -- ")
        for message in messages
    )


def test_the_same_pendragon_with_ffmpeg_found_also_asks_for_grandma_api(
    pendragon_config: pathlib.Path,
) -> None:
    """The control: the only difference is the ffmpeg line, so it is what
    kept grandma-api out of the pass above."""
    found = LAVENDER_2026_09_23 + "ffmpeg=yes=ffmpeg version 8.0\n"
    _test_hooks.run = FakeRun([ok(""), ok(PENDRAGON_ROOM), ok(""), ok(found)])
    endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
    _test_hooks.http_post = endpoint

    argv = ["--config", str(pendragon_config), node_agent.NODE_FLAG, "pendragon"]
    assert node_agent.main(argv) == 0

    claim = endpoint.arguments[1]
    assert claim["tags"] == ["ffmpeg", "windows"]
    assert claim["projects"] == [
        DEMO_PROJECT,
        "libs/platform_core",
        GRANDMA,
        "tools/fleet",
        "tools/maketools",
    ]
