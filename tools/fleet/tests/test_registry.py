"""Reconciling the dispatch workspace against the fleet's identity registry.

THE DRIFT THIS EXISTS FOR IS REAL AND DATED. On 2026-09-05
``fleet-mcp/fleet-nodes.json`` marked loki disabled for a trip; this workspace
had no ``enabled`` field to learn it into, so every auto-select dispatch paid a
ten-second ssh timeout rediscovering it, and one was refused outright.

The registry fixtures below are TRIMMED COPIES OF THE REAL DOCUMENT, not
invented shapes -- same key names, same node names, same booleans as the live
file carried when this was written. A fixture shaped by the same assumption as
the decoder agrees with the decoder, which is how the two defects earlier that
day both survived a green suite.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONObject, dump_json_str

from fleet.contracts.workspace import FleetWorkspace, decode_fleet_workspace
from fleet.core import _test_hooks, registry
from tests.conftest import workspace_document


def _registry_document(**enabled: bool) -> str:
    """Render an identity registry carrying the given nodes.

    Shaped like the real one: an object with a ``nodes`` ARRAY whose entries
    carry ``name`` and ``enabled`` among fields this package ignores.

    Args:
        **enabled: Node name to whether the registry says it is expected to
            answer.

    Returns:
        The document's text.
    """
    document: JSONObject = {
        "sshKeyFingerprint": "SHA256:ignored",
        "nodes": [
            {
                "name": name,
                "role": "worker",
                "user": "austi",
                "tailnetIp": "100.0.0.1",
                "enabled": value,
                "tunnel": None,
                "notes": "carried by the real registry; ignored here",
            }
            for name, value in enabled.items()
        ],
    }
    return dump_json_str(document)


def _workspace(*, excluding: dict[str, str] | None = None, **enabled: bool) -> FleetWorkspace:
    """Build a dispatch workspace declaring the given nodes.

    Args:
        excluding: Machines it deliberately will not dispatch to, and why.
        **enabled: Node name to whether this workspace will dispatch to it.

    Returns:
        The decoded workspace.
    """
    document = workspace_document()
    document["not_dispatchable"] = dict(excluding or {})
    nodes = document["nodes"]
    if not isinstance(nodes, dict):
        raise AssertionError("the fixture workspace must declare nodes")
    template = nodes["lavender"]
    if not isinstance(template, dict):
        raise AssertionError("the fixture workspace must declare lavender")
    rebuilt: JSONObject = {}
    for name, value in enabled.items():
        node = dict(template)
        node["host"] = name
        node["enabled"] = value
        rebuilt[name] = node
    document["nodes"] = rebuilt
    return decode_fleet_workspace(document)


class TestTheDriftThatHappened:
    def test_dispatching_to_a_node_the_registry_says_is_off_is_drift(self) -> None:
        """loki, 2026-09-05, exactly. This is the case that cost a refused
        dispatch and a day of ssh timeouts."""
        drift = registry.compare(
            _workspace(lavender=True, loki=True),
            registry.decode_registry_nodes(_registry_document(lavender=True, loki=False)),
        )

        assert drift["enabled_here_disabled_there"] == ("loki",)
        assert registry.has_drifted(drift) is True

    def test_the_line_says_which_way_it_runs_and_what_it_costs(self) -> None:
        """The two directions call for opposite edits. A reader who gets them
        backwards changes the wrong file."""
        drift = registry.compare(
            _workspace(loki=True),
            registry.decode_registry_nodes(_registry_document(loki=False)),
        )

        lines = registry.describe(drift, registry_path="/x/fleet-nodes.json")

        assert len(lines) == 1
        assert "this workspace dispatches to it" in lines[0]
        assert "/x/fleet-nodes.json says it is disabled" in lines[0]
        assert "ssh timeout" in lines[0]

    def test_skipping_a_node_the_registry_says_is_live_is_also_drift(self) -> None:
        """The quiet direction, and worse in its way: capacity that exists and
        is not being used, with nothing to notice it."""
        drift = registry.compare(
            _workspace(sedona=False),
            registry.decode_registry_nodes(_registry_document(sedona=True)),
        )

        assert drift["disabled_here_enabled_there"] == ("sedona",)
        lines = registry.describe(drift, registry_path="/x/r.json")
        assert "Capacity that exists and is not being used" in lines[0]

    def test_a_node_the_registry_never_heard_of_is_drift(self) -> None:
        drift = registry.compare(
            _workspace(lavender=True, mystery=True),
            registry.decode_registry_nodes(_registry_document(lavender=True)),
        )

        assert drift["missing_from_registry"] == ("mystery",)
        lines = registry.describe(drift, registry_path="/x/r.json")
        assert "has never heard of it" in lines[0]

    def test_agreement_is_silent(self) -> None:
        drift = registry.compare(
            _workspace(lavender=True, loki=False),
            registry.decode_registry_nodes(_registry_document(lavender=True, loki=False)),
        )

        assert registry.has_drifted(drift) is False
        assert registry.describe(drift, registry_path="/x/r.json") == ()

    def test_a_registry_node_that_is_off_and_unmentioned_is_not_drift(self) -> None:
        """The registry holds every machine on the tailnet, including two
        boxes offline since August. Nothing dispatches to those and none of
        them is capacity, so calling their absence a disagreement would make
        the check cry wolf on its first run."""
        drift = registry.compare(
            _workspace(lavender=True),
            registry.decode_registry_nodes(
                _registry_document(lavender=True, emerald=False, pendragon=False)
            ),
        )

        assert registry.has_drifted(drift) is False

    def test_a_live_registry_node_this_workspace_never_mentions_is_drift(self) -> None:
        """austinpc, exactly: enabled, 24 logical cores, and invisible to the
        scheduler for weeks. SILENCE IS NOT A DECISION -- absence here cannot
        be told apart from an oversight, and this is the only drift direction
        that hides capacity rather than wasting it."""
        drift = registry.compare(
            _workspace(lavender=True),
            registry.decode_registry_nodes(_registry_document(lavender=True, austinpc=True)),
        )

        assert drift["enabled_there_absent_here"] == ("austinpc",)
        lines = registry.describe(drift, registry_path="/x/r.json")
        assert "says it is enabled and this workspace says nothing at all" in lines[0]
        assert "not_dispatchable" in lines[0]

    def test_naming_it_not_dispatchable_settles_it(self) -> None:
        """The exclusion is not a suppression: it is the workspace answering
        the question the check asked, in writing, where the next reader sees
        the reason rather than an absence."""
        drift = registry.compare(
            _workspace(
                lavender=True,
                excluding={"austinpc": "the hub the dispatcher runs on; it shares the .venv"},
            ),
            registry.decode_registry_nodes(_registry_document(lavender=True, austinpc=True)),
        )

        assert registry.has_drifted(drift) is False


class TestAnUnreadableRegistryIsRefused:
    def test_a_document_that_is_not_an_object(self) -> None:
        with pytest.raises(AppError) as excinfo:
            registry.decode_registry_nodes("[1, 2]")

        assert excinfo.value.code is FleetErrorCode.NODE_REGISTRY_UNREADABLE
        assert "not an object" in excinfo.value.message

    def test_a_nodes_member_that_is_not_an_array(self) -> None:
        with pytest.raises(AppError) as excinfo:
            registry.decode_registry_nodes('{"nodes": {"loki": true}}')

        assert "'nodes' is dict, not an array" in excinfo.value.message

    def test_a_node_that_is_not_an_object(self) -> None:
        with pytest.raises(AppError) as excinfo:
            registry.decode_registry_nodes('{"nodes": ["loki"]}')

        assert "a node is str, not an object" in excinfo.value.message

    def test_a_node_missing_enabled(self) -> None:
        """The one field this reconciler exists to read. A registry that
        stopped carrying it must fail loudly rather than compare nothing."""
        with pytest.raises(Exception) as excinfo:
            registry.decode_registry_nodes('{"nodes": [{"name": "loki"}]}')

        assert "enabled" in str(excinfo.value)


class TestReconcileReadsTheFile:
    def test_it_reads_the_named_path_and_reports(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "fleet-nodes.json"
        path.write_text(_registry_document(loki=False), encoding="utf-8")

        lines = registry.reconcile(_workspace(loki=True), registry_path=str(path))

        assert len(lines) == 1
        assert "loki" in lines[0]

    def test_a_path_that_holds_nothing_raises_rather_than_reporting_agreement(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A reconciliation that could not run has established NOTHING, and
        must never be mistaken for one that found agreement. This is the
        whole reason the path is passed rather than searched for."""
        _test_hooks.read_text = _test_hooks._default_read_text

        with pytest.raises(OSError):
            registry.reconcile(_workspace(loki=True), registry_path=str(tmp_path / "absent.json"))
