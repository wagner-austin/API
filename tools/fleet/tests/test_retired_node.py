"""A node retired from dispatch but kept declared, with its reason (MCPs 7e467416).

lavender's Windows lane was retired on 2026-09-29: its WSL VM, which runs the
lavender-wsl node and the GitHub runners, holds the memory the Windows lane
measures, so the Windows runner found no room while jobs waited. lavender
keeps its declared capacity because lavender-wsl's ``guest`` names it and the
host report reads that declaration, and it stays enabled because the audit
and the session observer still visit it; its machine file holds it out of
dispatch with ``dispatch.held`` (MCPs board task e447de72). So two things
must hold together here, and each case drives one through the real code:
the decoder keeps the held node declared but not dispatching, with its
reason, and a runner whose scheduled task outlived the retirement claims
nothing and asks the node nothing.
"""

from __future__ import annotations

import pathlib
from typing import Final

import pytest
from platform_core.json_utils import JSONObject, dump_json_str

from fleet.cli import node_agent
from fleet.contracts.workspace import decode_fleet_workspace
from fleet.core import _test_hooks
from tests._node_agent_fixtures import _credentials_in_env, node_argv
from tests._queue_fakes import FakeQueue, tick_body
from tests._registry import write_registry
from tests._roster_fixtures import dispatch_of
from tests.conftest import FakeRun, workspace_document

__all__ = ["_credentials_in_env"]

#: The reason registry/machines/lavender.json gives, shortened.
REASON: Final[str] = "its WSL VM holds the memory the Windows lane measures"


def _retired() -> JSONObject:
    """The shared one-node workspace with lavender retired and explained.

    Returns:
        The document.
    """
    document = workspace_document()
    dispatch_of(document, "lavender")["held"] = REASON
    return document


def test_the_decoder_keeps_both_the_declaration_and_the_reason() -> None:
    decoded = decode_fleet_workspace(_retired())

    assert (decoded["nodes"]["lavender"]["enabled"], decoded["not_dispatchable"]) == (
        False,
        {"lavender": REASON},
    )


def test_a_stale_runner_for_the_retired_node_claims_nothing(
    config_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The hub runner drops a retired node's job at its next start, but a
    tick can run first from a rolled checkout; it collects (the held
    listing) and then stops before the probe, so the node is never asked and
    the queue never claimed from."""
    write_registry(config_path, _retired())
    runner = FakeRun([])
    _test_hooks.run = runner
    endpoint = FakeQueue([dump_json_str({"jobs": []})])
    _test_hooks.http_post = endpoint

    with caplog.at_level("INFO"):
        assert node_agent.main(node_argv(config_path)) == 0

    assert (runner.calls, endpoint.tools) == ([], ["dispatch_list"])
    verdict = "is disabled in registry/machines/lavender.json; claiming nothing"
    assert f"lavender {verdict}" in caplog.text
    assert [tick_body(tick) for tick in endpoint.ticks] == [
        {
            "node": "lavender",
            "elevated": False,
            "tags": [],
            "fits": [],
            "claiming": False,
            "verdict": verdict,
        }
    ]
