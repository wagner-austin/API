"""The hub pump's record of what each wake bridge ran, end to end (MCPs board task 9f1367d5).

Real git checkouts under ``tmp_path`` and the queue as :class:`FakeQueue`,
which speaks the live endpoint's framing, so what is asserted is the call
this module sends for a checkout git actually read.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import dump_json_str

from fleet.cli import publisher_ticks
from fleet.cli.publisher_ticks import RanPublisher, parse_ran
from fleet.core import _test_hooks
from tests._git_fixtures import committed_repository, git
from tests._queue_fakes import FakeQueue, queue_env


def _checkout(root: pathlib.Path, name: str) -> tuple[pathlib.Path, str]:
    """Build a one-commit checkout of ``name`` with a bridge directory in it.

    Args:
        root: Where to build it.
        name: The repository its origin names.

    Returns:
        The bridge's directory inside it, and the checkout's HEAD.
    """
    repo = root / name
    commit = committed_repository(
        repo, {"tools/bridge/run.py": "print('tick')\n"}, "refs/heads/main"
    )
    git(repo, "checkout", "--quiet", "main")
    git(repo, "remote", "add", "origin", f"https://github.com/wagner-austin/{name}.git")
    return repo / "tools" / "bridge", commit


def _counted(ticks: int) -> str:
    """``dispatch_publisher_tick``'s answer, trimmed to what this module reads.

    Args:
        ticks: The stored state's count.

    Returns:
        The tool's text.
    """
    return dump_json_str({"tick": {"ticks": ticks}})


def _austinpc() -> str:
    """The hub's hostname, as the hook reports it.

    Returns:
        ``austinpc``.
    """
    return "austinpc"


class TestParseRan:
    def test_it_reads_a_bridge_s_name_exit_and_directory(self) -> None:
        assert parse_ran("hpc-wake=0=C:/API/tools/hpc-wake") == RanPublisher(
            publisher="hpc-wake", exit_code=0, directory=pathlib.Path("C:/API/tools/hpc-wake")
        )

    def test_a_signal_death_reads_negative(self) -> None:
        assert parse_ran("ci-wake=-9=/srv/API/tools/ci-wake")["exit_code"] == -9

    def test_a_directory_holding_the_separator_reads_whole(self) -> None:
        assert parse_ran("ci-wake=1=/srv/a=b")["directory"] == pathlib.Path("/srv/a=b")

    @pytest.mark.parametrize(
        "token",
        [
            "hpc-wake",
            "hpc-wake=0",
            "=0=/srv",
            "hpc-wake=0=",
            "hpc-wake=zero=/srv",
            "hpc-wake=-=/srv",
        ],
    )
    def test_a_malformed_token_is_refused_by_name(self, token: str) -> None:
        with pytest.raises(ValueError) as raised:
            parse_ran(token)

        assert str(raised.value).startswith(f"FLEET_PUBLISHER_TICKS_USAGE: {token!r} is not ")


class TestMain:
    def test_it_records_each_bridge_s_checkout_under_the_hub_runner(
        self, tmp_path: pathlib.Path
    ) -> None:
        hpc_wake, api_head = _checkout(tmp_path, "API")
        lock_wake, mcps_head = _checkout(tmp_path, "MCPs")
        (lock_wake / "run.py").write_text("print('edited')\n", encoding="utf-8")
        endpoint = FakeQueue([_counted(4), _counted(1)])
        _test_hooks.http_post = endpoint
        _test_hooks.env = queue_env()
        _test_hooks.hostname = _austinpc

        code = publisher_ticks.main([f"hpc-wake=0={hpc_wake}", f"lock-wake=3={lock_wake}"])

        assert code == 0
        assert endpoint.tools == ["dispatch_publisher_tick", "dispatch_publisher_tick"]
        identity = {
            "agent": "fleet-hub-austinpc",
            "sessionId": "6f531d60-5c82-5b23-b2a3-c523f13c9490",
            "cwd": str(pathlib.Path.cwd()),
        }
        assert endpoint.arguments == [
            {
                "host": "austinpc",
                "publisher": "hpc-wake",
                "repo": "API",
                "headSha": api_head,
                "dirty": False,
                "exitCode": 0,
                **identity,
            },
            {
                "host": "austinpc",
                "publisher": "lock-wake",
                "repo": "MCPs",
                "headSha": mcps_head,
                "dirty": True,
                "exitCode": 3,
                **identity,
            },
        ]

    def test_no_tokens_is_refused_before_anything_is_sent(self) -> None:
        endpoint = FakeQueue([])
        _test_hooks.http_post = endpoint

        with pytest.raises(ValueError, match="FLEET_PUBLISHER_TICKS_USAGE: name at least one"):
            publisher_ticks.main([])

        assert endpoint.tools == []

    def test_a_malformed_token_refuses_every_tick_before_anything_is_sent(
        self, tmp_path: pathlib.Path
    ) -> None:
        hpc_wake, _ = _checkout(tmp_path, "API")
        endpoint = FakeQueue([])
        _test_hooks.http_post = endpoint
        _test_hooks.env = queue_env()

        with pytest.raises(ValueError, match="FLEET_PUBLISHER_TICKS_USAGE"):
            publisher_ticks.main([f"hpc-wake=0={hpc_wake}", "ci-wake"])

        assert endpoint.tools == []

    def test_a_directory_outside_any_checkout_is_refused(self, tmp_path: pathlib.Path) -> None:
        endpoint = FakeQueue([])
        _test_hooks.http_post = endpoint
        _test_hooks.env = queue_env()
        _test_hooks.hostname = _austinpc

        with pytest.raises(AppError) as raised:
            publisher_ticks.main([f"hpc-wake=0={tmp_path}"])

        assert raised.value.code is ErrorCode.CONFIG_ERROR
        assert endpoint.tools == []


class TestEntryPoint:
    def test_running_as_a_module_records_and_exits_zero(self, tmp_path: pathlib.Path) -> None:
        hpc_wake, _ = _checkout(tmp_path, "API")
        endpoint = FakeQueue([_counted(1)])
        _test_hooks.http_post = endpoint
        _test_hooks.env = queue_env()
        _test_hooks.hostname = _austinpc
        saved_argv = sys.argv
        saved_module = sys.modules.pop("fleet.cli.publisher_ticks")
        sys.argv = ["publisher_ticks", f"hpc-wake=0={hpc_wake}"]
        try:
            with pytest.raises(SystemExit) as exited:
                runpy.run_module("fleet.cli.publisher_ticks", run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            sys.modules["fleet.cli.publisher_ticks"] = saved_module

        assert exited.value.code == 0
        assert endpoint.tools == ["dispatch_publisher_tick"]
