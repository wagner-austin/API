"""What a WSL node's host reports when the node does not answer (MCPs board task 45a4f22b).

The listing fixtures are wsl.exe's real bytes: UTF-16LE with no byte-order
mark, as ``ssh lavender wsl.exe -l -v | od -c`` showed on 2026-09-29, decoded
the way the command runner decodes every stdout.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError, dump_json_str

from fleet.cli import node_agent
from fleet.contracts.budget import NodeBudget
from fleet.contracts.node import (
    NodeConfig,
    NodePlatform,
    decode_node_config,
    encode_node_config,
)
from fleet.contracts.workspace import decode_fleet_workspace
from fleet.core import _test_hooks, dialect, host_report, names
from tests._node_agent_fixtures import _credentials_in_env, _sourced_config, sourced_document
from tests._queue_fakes import FakeQueue
from tests.conftest import FakeRun, failed, ok, workspace_document

__all__ = ["_credentials_in_env", "_sourced_config"]


def _wsl_bytes(text: str) -> str:
    """wsl.exe's output as the command runner hands it over.

    Args:
        text: What wsl.exe means to print.

    Returns:
        Its UTF-16LE bytes decoded as UTF-8, a NUL after every character.
    """
    return text.encode("utf-16-le").decode("utf-8", errors="replace")


#: lavender's listing at 11:48Z: one distro, the default, running.
LAVENDER_LIST = _wsl_bytes(
    "  NAME      STATE           VERSION\r\n* Ubuntu    Running         2\r\n"
)

#: What the Windows capacity probe prints for lavender at 0.3 GB free.
LAVENDER_STARVED = "free_ram_gb=0.3\nfree_disk_gb=612.4\n"


def _host() -> NodeConfig:
    """lavender, the Windows host.

    Returns:
        The node.
    """
    return NodeConfig(
        host="lavender",
        platform=NodePlatform.WINDOWS,
        stage_root="C:/fleet/stage",
        logical_cores=16,
        ram_gb=31.7,
        gpu=None,
        enabled=True,
        test_database=False,
        rust=None,
        cxx=None,
        docker=None,
        elevated=False,
        wsl_host=None,
        budget=NodeBudget(
            reserved_cores=4,
            reserved_ram_gb=6.0,
            worker_ram_gb=1.1,
            max_concurrent_runs=1,
            max_disk_gb=40.0,
        ),
    )


def _wsl_node() -> NodeConfig:
    """lavender-wsl, a linux node inside lavender's WSL.

    Returns:
        The node.
    """
    return NodeConfig(
        host="lavender-wsl",
        platform=NodePlatform.LINUX,
        stage_root="/home/corvis/fleet/stage",
        logical_cores=16,
        ram_gb=25.4,
        gpu=None,
        enabled=True,
        test_database=True,
        rust=None,
        cxx=None,
        docker=None,
        elevated=False,
        wsl_host="lavender",
        budget=NodeBudget(
            reserved_cores=8,
            reserved_ram_gb=12.0,
            worker_ram_gb=1.1,
            max_concurrent_runs=1,
            max_disk_gb=40.0,
        ),
    )


def _probe_send_context() -> str:
    """What remote reports it was doing when the capacity probe's send fails.

    Returns:
        ``sending <the Windows probe's path under lavender's stage root>``.
    """
    spoken = dialect.for_platform(NodePlatform.WINDOWS)
    return f"sending {spoken.script_path('C:/fleet/stage', names.CAPACITY_PROBE_STEM)}"


class TestListing:
    """parse_wsl_list over wsl.exe's real encoding."""

    def test_lavender_s_listing_reads_as_one_running_distro(self) -> None:
        assert host_report.parse_wsl_list(LAVENDER_LIST) == [
            host_report.Distro(name="Ubuntu", state="Running")
        ]

    def test_every_distro_is_read_whether_or_not_it_is_the_default(self) -> None:
        listing = _wsl_bytes(
            "  NAME            STATE    VERSION\r\n"
            "* Ubuntu          Stopped  2\r\n"
            "  docker-desktop  Running  2\r\n"
        )
        assert host_report.parse_wsl_list(listing) == [
            host_report.Distro(name="Ubuntu", state="Stopped"),
            host_report.Distro(name="docker-desktop", state="Running"),
        ]

    def test_a_listing_with_only_its_header_has_no_rows(self) -> None:
        assert host_report.parse_wsl_list(_wsl_bytes("  NAME  STATE  VERSION\r\n\r\n")) == []


def _with_nodes(document: JSONObject, extra: JSONObject) -> JSONObject:
    """A workspace document with nodes added beside its own.

    Args:
        document: The document to extend, modified in place.
        extra: The added nodes, keyed by name.

    Returns:
        The same document.
    """
    nodes = document["nodes"]
    assert isinstance(nodes, dict)
    nodes.update(extra)
    return document


def _describe(tmp_path: pathlib.Path) -> str:
    """describe_wsl_host for lavender as the shared workspace declares it (32 GB).

    Args:
        tmp_path: Where the case's empty ledger resolves.

    Returns:
        The report.
    """
    workspace = decode_fleet_workspace(workspace_document())
    return host_report.describe_wsl_host(workspace, tmp_path / "ledger.jsonl", "lavender")


class TestDescribe:
    """describe_wsl_host over the fake ssh runner."""

    def test_a_starved_host_says_how_much_it_has_and_what_wsl_is_doing(
        self, tmp_path: pathlib.Path
    ) -> None:
        runner = FakeRun([ok(""), ok(LAVENDER_STARVED), ok(LAVENDER_LIST)])
        _test_hooks.run = runner

        assert _describe(tmp_path) == (
            "its host lavender answers with 0.3 GB of 32.0 GB RAM and 612.4 GB disk free; "
            "wsl: Ubuntu Running"
        )
        assert runner.calls[-1][-3:] == host_report.WSL_LIST_ARGV

    def test_a_host_that_does_not_answer_either_says_so(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun([failed(255, "ssh: connect to host lavender: timed out")])

        assert _describe(tmp_path) == (
            "its host lavender did not answer either: ssh to lavender failed while "
            f"{_probe_send_context()}: ssh: connect to host lavender: timed out"
        )

    def test_a_listing_that_fails_is_named_beside_the_memory(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun(
            [ok(""), ok(LAVENDER_STARVED), failed(1, "wsl.exe is not recognized")]
        )

        assert _describe(tmp_path) == (
            "its host lavender answers with 0.3 GB of 32.0 GB RAM and 612.4 GB disk free; "
            "wsl: wsl.exe -l -v failed: running `wsl.exe -l -v` on lavender exited 1: "
            "wsl.exe is not recognized"
        )

    def test_a_host_with_no_distro_says_none_is_listed(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun(
            [ok(""), ok(LAVENDER_STARVED), ok(_wsl_bytes("  NAME  STATE  VERSION\r\n"))]
        )

        assert _describe(tmp_path) == (
            "its host lavender answers with 0.3 GB of 32.0 GB RAM and 612.4 GB disk free; "
            "wsl: no distro listed"
        )


class TestDeclaration:
    """wsl_host's decode, on the node and across the workspace."""

    def test_the_declaration_round_trips(self) -> None:
        encoded = encode_node_config(_wsl_node())
        assert encoded["wsl_host"] == "lavender"
        assert decode_node_config(encoded) == _wsl_node()

    def test_an_absent_declaration_is_refused(self) -> None:
        encoded = encode_node_config(_wsl_node())
        del encoded["wsl_host"]
        with pytest.raises(JSONTypeError, match="must declare 'wsl_host'"):
            decode_node_config(encoded)

    @pytest.mark.parametrize("value", ["", 7])
    def test_a_value_that_is_not_a_name_is_refused(self, value: str | int) -> None:
        encoded = {**encode_node_config(_wsl_node()), "wsl_host": value}
        with pytest.raises(JSONTypeError, match="non-empty string or null"):
            decode_node_config(encoded)

    def test_a_windows_node_may_not_name_a_host(self) -> None:
        encoded = {**encode_node_config(_host()), "wsl_host": "sedona"}
        with pytest.raises(JSONTypeError, match="only a linux node runs inside"):
            decode_node_config(encoded)

    def test_the_workspace_accepts_a_host_that_is_one_of_its_windows_nodes(self) -> None:
        document = _with_nodes(
            workspace_document(), {"lavender-wsl": encode_node_config(_wsl_node())}
        )
        assert decode_fleet_workspace(document)["nodes"]["lavender-wsl"]["wsl_host"] == "lavender"

    def test_the_workspace_refuses_a_host_it_does_not_declare(self) -> None:
        document = _with_nodes(
            workspace_document(),
            {"lavender-wsl": {**encode_node_config(_wsl_node()), "wsl_host": "pendragon"}},
        )
        with pytest.raises(JSONTypeError, match="'pendragon', which is not a node"):
            decode_fleet_workspace(document)

    def test_the_workspace_refuses_a_host_that_is_not_windows(self) -> None:
        document = _with_nodes(
            workspace_document(),
            {
                "lavender-wsl": encode_node_config(_wsl_node()),
                "other-wsl": {**encode_node_config(_wsl_node()), "wsl_host": "lavender-wsl"},
            },
        )
        with pytest.raises(JSONTypeError, match="a linux node; only a windows node runs WSL"):
            decode_fleet_workspace(document)


class TestTheTick:
    """A whole node tick for a WSL node that does not answer."""

    def test_the_tick_logs_what_the_host_sees(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        document = _with_nodes(
            sourced_document((("npm", "ci"),)), {"lavender-wsl": encode_node_config(_wsl_node())}
        )
        sourced_config.write_text(dump_json_str(document), encoding="utf-8")
        _test_hooks.run = FakeRun(
            [
                failed(255, "Connection timed out during banner exchange"),
                ok(""),
                ok(LAVENDER_STARVED),
                ok(LAVENDER_LIST),
            ]
        )
        endpoint = FakeQueue([dump_json_str({"jobs": []})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            argv = ["--config", str(sourced_config), node_agent.NODE_FLAG, "lavender-wsl"]
            assert node_agent.main(argv) == 0

        messages = [record.getMessage() for record in caplog.records]
        reports = [m for m in messages if m.startswith("lavender-wsl did not answer, and ")]
        assert reports == [
            "lavender-wsl did not answer, and its host lavender answers with 0.3 GB of "
            "32.0 GB RAM and 612.4 GB disk free; wsl: Ubuntu Running"
        ]
        assert endpoint.tools == ["dispatch_list"]
