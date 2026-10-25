"""A missing tool refuses only the projects that need it (MCPs board task 939ec5c7, A3).

pendragon claimed nothing on every tick of 2026-10-02 until 02:48Z, logging
``NODE_TOOL_MISSING: ... ffmpeg -- grandma-api's check converts real audio
files through ffmpeg``, while tools/fleet, libs/platform_core and
tools/maketools waited in the queue with no tag between them and that node.
This replays that tick: pendragon and those four projects exactly as the real
registry declares them, a toolchain answer with every build tool and no
ffmpeg, and room for a run. One pass of the runner must ask the queue for
every project but grandma-api, with tags that keep grandma-api's job away
from it, and must say which tool it lacks without closing the node.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONObject, dump_json_str

from fleet.cli import node_agent
from fleet.contracts.project import encode_project_config
from fleet.contracts.workspace import decode_fleet_workspace
from fleet.core import _test_hooks
from tests._node_agent_fixtures import NPM_CI, _credentials_in_env, sourced_document
from tests._queue_fakes import FakeQueue
from tests._registry import shipped_document, write_registry
from tests._roster_fixtures import dispatch_document, machine_document
from tests._toolchain_fixtures import DIPHTHERIA_2026_09_23
from tests.conftest import CALM, DEMO_PROJECT, FakeRun, ok

__all__ = ["_credentials_in_env"]

#: The project that needs ffmpeg, and the tag-free ones that queued behind it,
#: in the name order a claim lists the projects it fits.
GRANDMA = "services/grandma-api"
WAITING = ("libs/platform_core", "tools/fleet", "tools/maketools")

#: pendragon with room for a run: 9.0 GB free of its 11.7 against its 1.5 GB
#: owner reservation.
PENDRAGON_ROOM = f"free_ram_gb=9.0\nfree_disk_gb=400.0\nreserved_cores=0\n{CALM}"


def pendragon_2026_10_02() -> JSONObject:
    """pendragon's capacity as the replayed tick read it.

    Written out rather than read from the shipped roster, because a netboot
    reimage rewrites that file (MCPs board task 9fa23d42): on 2026-10-21 the
    Windows worker it was then became a Linux install, and a replay of the
    tick's capacity must not change with it. The replay runs it as the one
    platform the fleet renders since board task e705f08c.

    Returns:
        The machine's object: a worker with 16 logical cores and 11.7 GB, its
        owner's 2 cores and 1.5 GB reserved.
    """
    dispatch = dispatch_document()
    dispatch["ram_gb"] = 11.7
    dispatch["budget"] = {
        "reserved_cores": 2,
        "reserved_ram_gb": 1.5,
        "worker_ram_gb": 1.1,
        "max_disk_gb": 40.0,
        "checks_at_once": None,
        "pinned_checks_only": False,
    }
    return machine_document(role="worker", dispatch=dispatch)


@pytest.fixture(name="pendragon_config")
def _pendragon_config(config_path: pathlib.Path) -> pathlib.Path:
    """The shared workspace with pendragon as its node and the real projects beside the demo.

    Args:
        config_path: The shared workspace document, clock pinned.

    Returns:
        The same path, rewritten.
    """
    real = decode_fleet_workspace(shipped_document())
    document = sourced_document((NPM_CI,))
    document["machines"] = {"pendragon": pendragon_2026_10_02()}
    projects = document["projects"]
    assert isinstance(projects, dict)
    for name in (GRANDMA, *WAITING):
        project = encode_project_config(real["projects"][name])
        project["source"] = None
        projects[name] = project
    return write_registry(config_path, document)


def test_pendragon_without_ffmpeg_asks_the_queue_for_every_project_but_grandma_api(
    pendragon_config: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    runner = FakeRun([ok(""), ok(PENDRAGON_ROOM), ok(""), ok(DIPHTHERIA_2026_09_23)])
    _test_hooks.run = runner
    endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
    _test_hooks.http_post = endpoint

    argv = ["--config", str(pendragon_config), node_agent.NODE_FLAG, "pendragon"]
    with caplog.at_level("INFO"):
        assert node_agent.main(argv) == 0

    assert endpoint.tools == ["dispatch_list", "dispatch_claim"]
    claim = endpoint.arguments[1]
    assert claim["tags"] == ["linux"]
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
    found = DIPHTHERIA_2026_09_23 + "ffmpeg=yes=ffmpeg version 6.1.1-3ubuntu5\n"
    _test_hooks.run = FakeRun([ok(""), ok(PENDRAGON_ROOM), ok(""), ok(found)])
    endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
    _test_hooks.http_post = endpoint

    argv = ["--config", str(pendragon_config), node_agent.NODE_FLAG, "pendragon"]
    assert node_agent.main(argv) == 0

    claim = endpoint.arguments[1]
    assert claim["tags"] == ["ffmpeg", "linux"]
    assert claim["projects"] == [
        DEMO_PROJECT,
        "libs/platform_core",
        GRANDMA,
        "tools/fleet",
        "tools/maketools",
    ]
