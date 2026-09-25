"""One tick of the hub runner, against a faked queue and a faked fleet.

The queue speaks the real JSON-RPC-over-SSE shape and the node speaks through
the same command hook every other dispatch test uses, so what is exercised is
the real engine with only the two boundaries this machine cannot cross in a
test faked.

WHAT THIS RUNNER IS NOW. Since MCPs board task fd5cabfa the hub runner claims
the queue's HUB lane only: the rebuild and the session verbs, each run on the
hub and closed in the tick that claimed it (``test_agent_rebuild.py`` and
``test_agent_restart.py`` carry those). The make targets are the node lane,
one :mod:`fleet.cli.node_agent` per enabled node (``test_node_agent.py``).
What is asserted here is the shape of the hub tick itself: that it claims
from the hub lane with no tags, collects nothing because it starts nothing,
refuses to run without the queue's credentials, and observes the fleet's
sessions after the queue work.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from board_watch import _test_hooks as board_watch_hooks
from board_watch import config as board_config
from platform_core.error_codes_tooling import BoardWatchErrorCode
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONTypeError, dump_json_str
from platform_core.mcp_testing import DECLARED_TASKBOARD_URL

from fleet.cli import agent
from fleet.core import _test_hooks
from tests._queue_fakes import FakeEnv, FakeQueue, queue_env
from tests.conftest import FakeRun, agent_argv, failed, ok

#: The board's variables for the observe pass, with the url overridden so
#: no test reads the MCPs checkout's endpoint declaration.
BOARD_ENV = {
    board_config.API_KEY_VARIABLE: "board-key",
    board_config.TENANT_ID_VARIABLE: "tenant",
    board_config.URL_VARIABLE: DECLARED_TASKBOARD_URL,
}


@pytest.fixture(name="credentials_in_env", autouse=True)
def _credentials_in_env() -> None:
    """Give every test the two variables the agent refuses to run without."""
    _test_hooks.env = queue_env()


class TestAnEmptyHubLane:
    def test_a_tick_with_nothing_to_do_succeeds(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """The outcome of most ticks, and it must be exit 0 -- a scheduling
        loop that stopped on an empty lane would stop immediately."""
        endpoint = FakeQueue([dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        assert agent.main(agent_argv(config_path, repo)) == 0
        assert endpoint.tools == ["dispatch_claim"]

    def test_the_claim_is_the_hub_lane_with_no_tags_and_nothing_is_collected(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """The partition that keeps a revive out of a queue of checks (board
        task fd5cabfa, A5): this runner asks for hub verbs only, and it has
        no held_by pass because every hub verb closes in its own tick."""
        endpoint = FakeQueue([dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        agent.main(agent_argv(config_path, repo))

        assert endpoint.tools == ["dispatch_claim"]
        assert endpoint.arguments[0]["lane"] == "hub"
        assert endpoint.arguments[0]["tags"] == []
        assert endpoint.arguments[0]["leaseSeconds"] == agent.CLAIM_LEASE_SECONDS
        assert "node" not in endpoint.arguments[0]


class TestCredentialsAndEntryPoint:
    def test_a_missing_credential_stops_the_tick_before_any_call(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        _test_hooks.env = FakeEnv({})
        endpoint = FakeQueue([])
        _test_hooks.http_post = endpoint

        with pytest.raises(AppError) as raised:
            agent.main(agent_argv(config_path, repo))

        assert raised.value.code is FleetErrorCode.QUEUE_CREDENTIALS_MISSING
        assert endpoint.tools == []

    def test_a_malformed_workspace_stops_the_tick_before_any_call(
        self, tmp_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """The hub lane dispatches nothing to a node, and the workspace is
        still read: a runner whose fleet.json does not decode has no business
        claiming, and the node runners beside it read the same file."""
        broken = tmp_path / "fleet.json"
        broken.write_text(dump_json_str({"nodes": {}}), encoding="utf-8")
        endpoint = FakeQueue([])
        _test_hooks.http_post = endpoint

        with pytest.raises(JSONTypeError, match="workspace declares no nodes"):
            agent.main(agent_argv(broken, repo))

        assert endpoint.tools == []

    def test_a_node_scoped_agent_passes_its_node_to_the_claim(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        endpoint = FakeQueue([dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        agent.main([*agent_argv(config_path, repo), agent.NODE_FLAG, "austinpc"])

        assert endpoint.arguments[0]["node"] == "austinpc"

    def test_the_console_entry_point_exits_zero(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        _test_hooks.http_post = FakeQueue([dump_json_str({"claimed": None})])
        saved = sys.argv
        sys.argv = ["fleet-agent", *agent_argv(config_path, repo)]
        try:
            with pytest.raises(SystemExit) as raised:
                agent.entrypoint()
        finally:
            sys.argv = saved

        assert raised.value.code == 0

    def test_running_as_a_module_actually_runs(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """The half that silently goes missing without an `if __name__` block:
        `python -m` would import the module, run nothing, and exit 0 -- which
        reads as a tick that found an empty queue, on a runner that never
        asked."""
        _test_hooks.http_post = FakeQueue([dump_json_str({"claimed": None})])
        saved_argv = sys.argv
        saved_module = sys.modules.pop("fleet.cli.agent", None)
        sys.argv = ["x", *agent_argv(config_path, repo)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module("fleet.cli.agent", run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules["fleet.cli.agent"] = saved_module

        assert raised.value.code == 0


def _registry(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write an identity registry with one worker and the hub, as the real file
    is shaped.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The registry's path.
    """
    path = tmp_path / "fleet-nodes.json"
    path.write_text(
        dump_json_str(
            {
                "nodes": [
                    {
                        "name": "austinpc",
                        "role": "hub",
                        "user": "Test",
                        "enabled": True,
                        "platform": "windows",
                        "gpu": {
                            "probe": "nvidia-smi",
                            "measured": "2026-09-20",
                            "adapter": "NVIDIA GeForce RTX 3090 Ti",
                            "vramMib": 24564,
                            "driver": "591.86",
                        },
                    },
                    {
                        "name": "serendipity",
                        "role": "worker",
                        "user": "austi",
                        "enabled": True,
                        "platform": "windows",
                        "gpu": None,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


class TestObservingSessions:
    """The second pass: the fleet's Claude Code sessions into the board's ledger."""

    def test_a_registry_makes_the_tick_walk_the_workers_after_the_queue_work(
        self,
        config_path: pathlib.Path,
        repo: pathlib.Path,
        tmp_path: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        board_watch_hooks.env = FakeEnv(BOARD_ENV)
        _test_hooks.run = FakeRun([failed(255, "ssh: serendipity is asleep")])
        endpoint = FakeQueue([dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert (
                agent.main(
                    [*agent_argv(config_path, repo), agent.REGISTRY_FLAG, str(_registry(tmp_path))]
                )
                == 0
            )

        # The queue pass ran, the hub was skipped, the one worker was asked
        # and its silence was logged as an outcome rather than raised.
        assert endpoint.tools == ["dispatch_claim"]
        lines = [
            record.getMessage() for record in caplog.records if "serendipity" in record.getMessage()
        ]
        assert lines == [
            "serendipity: not observed -- ssh to serendipity failed while sending "
            "C:/Users/austi/.fleet/observe-sessions.ps1: ssh: serendipity is asleep"
        ]

    def test_a_worker_that_answers_is_recorded_on_the_board(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        board_watch_hooks.env = FakeEnv(BOARD_ENV)
        _test_hooks.run = FakeRun(
            [
                ok(""),
                ok(dump_json_str({"platform": "win32", "hostname": "serendipity", "records": []})),
            ]
        )
        endpoint = FakeQueue(
            [
                dump_json_str({"claimed": None}),
                "observed 0 session(s) for win32:serendipity (0 new row(s)) at t",
            ]
        )
        _test_hooks.http_post = endpoint

        agent.main([*agent_argv(config_path, repo), agent.REGISTRY_FLAG, str(_registry(tmp_path))])

        assert endpoint.tools == ["dispatch_claim", "task_session_observe"]
        assert endpoint.arguments[1]["machine"] == "win32:serendipity"
        assert endpoint.arguments[1]["agent"] == "fleet-runner-austinpc"

    def test_without_a_registry_the_tick_says_so_and_touches_no_node(
        self, config_path: pathlib.Path, repo: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _test_hooks.run = FakeRun([])
        endpoint = FakeQueue([dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            agent.main(agent_argv(config_path, repo))

        assert endpoint.tools == ["dispatch_claim"]
        messages = [record.getMessage() for record in caplog.records]
        assert "session observation skipped: no --registry given" in messages
        assert "hub lane empty" in messages

    def test_the_board_key_is_required_only_once_the_queue_work_is_done(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """The two secrets are different containers' keys. A tick refused
        for the board's before the queue pass ran would stall dispatch on a
        credential dispatch does not use."""
        board_watch_hooks.env = FakeEnv({})
        endpoint = FakeQueue([dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        with pytest.raises(AppError) as raised:
            agent.main(
                [*agent_argv(config_path, repo), agent.REGISTRY_FLAG, str(_registry(tmp_path))]
            )

        assert raised.value.code is BoardWatchErrorCode.API_KEY_MISSING
        assert endpoint.tools == ["dispatch_claim"]
