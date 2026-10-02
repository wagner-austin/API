"""A node retired from dispatch but kept declared, with its reason (MCPs 7e467416).

lavender's Windows lane was retired on 2026-09-29: its WSL VM, which runs the
lavender-wsl node and the GitHub runners, holds the memory the Windows lane
measures, so the Windows runner found no room while jobs waited. lavender
stays under ``nodes`` because lavender-wsl names it as its ``wsl_host`` and
the host report reads that declaration, and it stays enabled in the MCPs
identity registry because the audit and the session observer still visit
it. So three things must hold together, and each case here drives one
through the real code: the decoder accepts a disabled node that
``not_dispatchable`` explains, the registry reconciliation reads that as a
decision rather than unused capacity, and a runner whose scheduled task
outlived the retirement claims nothing and asks the node nothing.
"""

from __future__ import annotations

import pathlib
from typing import Final

import pytest
from platform_core.json_utils import JSONObject, dump_json_str

from fleet.cli import _config, node_agent, nodes
from fleet.contracts.workspace import decode_fleet_workspace
from fleet.core import _test_hooks
from tests._node_agent_fixtures import _credentials_in_env, node_argv
from tests._queue_fakes import FakeQueue, tick_body
from tests.conftest import FakeRun, workspace_document

__all__ = ["_credentials_in_env"]

#: The reason fleet.json gives, shortened.
REASON: Final[str] = "its WSL VM holds the memory the Windows lane measures"


def _retired() -> JSONObject:
    """The shared one-node workspace with lavender retired and explained.

    Returns:
        The document.
    """
    document = workspace_document()
    declared = document["nodes"]
    assert isinstance(declared, dict)
    lavender = declared["lavender"]
    assert isinstance(lavender, dict)
    lavender["enabled"] = False
    document["not_dispatchable"] = {"lavender": REASON}
    return document


def test_the_decoder_keeps_both_the_declaration_and_the_reason() -> None:
    decoded = decode_fleet_workspace(_retired())

    assert (decoded["nodes"]["lavender"]["enabled"], decoded["not_dispatchable"]) == (
        False,
        {"lavender": REASON},
    )


def test_the_reconciliation_reads_the_retirement_as_agreement(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The registry keeps lavender enabled; fleet-nodes still exits 0 and
    says how much it compared, without an ssh call to anything."""
    config = tmp_path / "fleet.json"
    config.write_text(dump_json_str(_retired()), encoding="utf-8")
    registry_path = tmp_path / "fleet-nodes.json"
    entry: JSONObject = {
        "name": "lavender",
        "role": "worker",
        "user": "Test",
        "enabled": True,
        "platform": "windows",
        "gpu": None,
    }
    registry_path.write_text(dump_json_str({"nodes": [entry]}), encoding="utf-8")
    runner = FakeRun([])
    _test_hooks.run = runner
    argv = [_config.CONFIG_FLAG, str(config), nodes.REGISTRY_FLAG, str(registry_path)]

    with caplog.at_level("INFO"):
        status = nodes.main([*argv, nodes.PROBE_FLAG, nodes.PROBE_NEVER])

    assert (status, runner.calls) == (0, [])
    assert "1 node(s) agree with" in caplog.text
    assert "REGISTRY DRIFT" not in caplog.text


def test_a_stale_runner_for_the_retired_node_claims_nothing(
    config_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """register-node-agents removes the task, but a tick can run first from
    a rolled checkout; it collects (the held listing) and then stops before
    the probe, so the node is never asked and the queue never claimed from."""
    config_path.write_text(dump_json_str(_retired()), encoding="utf-8")
    runner = FakeRun([])
    _test_hooks.run = runner
    endpoint = FakeQueue([dump_json_str({"jobs": []})])
    _test_hooks.http_post = endpoint

    with caplog.at_level("INFO"):
        assert node_agent.main(node_argv(config_path)) == 0

    assert (runner.calls, endpoint.tools) == ([], ["dispatch_list"])
    assert "lavender is disabled in fleet.json; claiming nothing" in caplog.text
    assert [tick_body(tick) for tick in endpoint.ticks] == [
        {
            "node": "lavender",
            "elevated": False,
            "tags": [],
            "fits": [],
            "claiming": False,
            "verdict": "is disabled in fleet.json; claiming nothing",
        }
    ]
