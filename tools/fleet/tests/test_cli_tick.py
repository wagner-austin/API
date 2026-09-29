"""One scheduled fleet tick, run as a native process (MCPs board task 94ac1c4f).

The credentials file and the log directory are real files under tmp_path;
only the launcher's process and the clock are the hooks' fakes, because a
real launcher would extract a rolled tree and run an agent against the queue.
"""

from __future__ import annotations

import os
import pathlib
import runpy
import sys
from collections.abc import Generator

import pytest
from platform_core.errors import AppError, ErrorCode

from fleet.cli import rolled as rolled_cli
from fleet.cli import tick
from fleet.core import _test_hooks
from tests.conftest import FakeRun, failed, ok, timed_out

#: 2026-09-29T00:40:00Z, the pinned clock of every tick here.
STARTED = 1_790_642_400

CREDENTIALS = (
    "# hpc-wake credentials. UNTRACKED.\n"
    "$env:TASKBOARD_MCP_API_KEY = 'board-key'\n"
    "$env:FLEET_MCP_API_KEY = 'fleet-key'\n"
)


@pytest.fixture(name="api_root")
def _api_root(tmp_path: pathlib.Path) -> pathlib.Path:
    """An API checkout holding only the credentials file a tick reads."""
    root = tmp_path / "PROJECTS" / "API"
    env_file = root / pathlib.PurePosixPath(tick.ENV_FILE)
    env_file.parent.mkdir(parents=True)
    env_file.write_text(CREDENTIALS, encoding="utf-8")
    return root


@pytest.fixture(name="clock")
def _clock() -> Generator[list[int], None, None]:
    """A clock answering the start, then 12 s later, for every tick."""
    readings = [STARTED, STARTED + 12]

    def now() -> int:
        return readings.pop(0)

    _test_hooks.now = now
    yield readings


def _argv(api_root: pathlib.Path, logs: pathlib.Path, *lane: str) -> list[str]:
    return ["--api-root", str(api_root), "--log-directory", str(logs), "--lane", *lane]


def _log_lines(logs: pathlib.Path, stem: str) -> list[str]:
    return (logs / f"{stem}-2026-09-29.log").read_text(encoding="utf-8").splitlines()


class TestTheLanes:
    def test_the_hub_lane_runs_the_fleet_agent_as_the_hub_runner(self) -> None:
        root = pathlib.Path("C:/Users/Test/PROJECTS/API")
        mcps = pathlib.Path("C:/Users/Test/PROJECTS/MCPs")
        assert tick.plan_tick(root, "hub", None) == tick.TickPlan(
            arguments=(
                "--repo-root",
                str(root),
                "--agent",
                "fleet-agent",
                "--",
                "--agent",
                "fleet-runner-austinpc",
                "--session",
                "a850f688-f98d-415c-a244-e993226ca2fc",
                "--repo-root",
                str(root),
                "--mcps-root",
                str(mcps),
                "--registry",
                str(mcps / "fleet-mcp" / "fleet-nodes.json"),
            ),
            stem="fleet-agent",
            header="hub",
        )

    def test_a_node_lane_runs_the_node_agent_for_its_node(self) -> None:
        root = pathlib.Path("C:/api")
        assert tick.plan_tick(root, "node", "sedona") == tick.TickPlan(
            arguments=(
                "--repo-root",
                str(root),
                "--agent",
                "fleet-node-agent",
                "--",
                "--node",
                "sedona",
            ),
            stem="fleet-node-sedona",
            header="node sedona",
        )

    def test_the_announce_lane_checks_in_and_shares_its_nodes_log(self) -> None:
        plan = tick.plan_tick(pathlib.Path("C:/api"), "announce", "sedona")
        assert plan["arguments"][-3:] == ("--node", "sedona", "--announce")
        assert plan["stem"] == "fleet-node-sedona"
        assert plan["header"] == "announce sedona"

    def test_the_hub_lane_naming_a_node_is_refused(self) -> None:
        with pytest.raises(ValueError) as refused:
            tick.plan_tick(pathlib.Path("C:/api"), "hub", "sedona")
        assert str(refused.value) == "FLEET_TICK_USAGE: the hub lane takes no --node"

    @pytest.mark.parametrize("lane", ["node", "announce"])
    def test_a_node_lane_naming_no_node_is_refused(self, lane: tick.Lane) -> None:
        with pytest.raises(ValueError) as refused:
            tick.plan_tick(pathlib.Path("C:/api"), lane, None)
        assert str(refused.value) == f"FLEET_TICK_USAGE: the {lane} lane needs --node"

    def test_each_lane_name_is_accepted_and_nothing_else(self) -> None:
        assert [tick.require_lane(name) for name in ("hub", "node", "announce")] == [
            "hub",
            "node",
            "announce",
        ]
        with pytest.raises(ValueError) as refused:
            tick.require_lane("agent")
        assert str(refused.value) == (
            "FLEET_TICK_USAGE: --lane is one of ['hub', 'node', 'announce'], not 'agent'"
        )

    def test_the_ticks_deadline_is_half_a_minute_past_the_agents(self) -> None:
        assert tick.TICK_WALL_SECONDS == rolled_cli.AGENT_WALL_SECONDS + 30 == 39 * 60 + 30


class TestOneTick:
    def test_a_tick_runs_the_launcher_with_the_credentials_and_logs_its_record(
        self, api_root: pathlib.Path, tmp_path: pathlib.Path, clock: list[int]
    ) -> None:
        logs = tmp_path / "logs"
        runner = FakeRun([ok("claimed nothing\n")])
        _test_hooks.run = runner

        status = tick.main(_argv(api_root, logs, "node", "--node", "sedona"))

        assert status == 0
        assert runner.calls == [
            (
                sys.executable,
                "-m",
                "fleet.cli.rolled",
                "--repo-root",
                str(api_root),
                "--agent",
                "fleet-node-agent",
                "--",
                "--node",
                "sedona",
            )
        ]
        assert runner.set_env == [
            (("TASKBOARD_MCP_API_KEY", "board-key"), ("FLEET_MCP_API_KEY", "fleet-key"))
        ]
        assert runner.timeouts == [tick.TICK_WALL_SECONDS]
        assert _log_lines(logs, "fleet-node-sedona") == [
            f"TICK START 2026-09-29T00:40:00+00:00 node sedona task-pid {os.getpid()}",
            "claimed nothing",
            "TICK EXIT 0 2026-09-29T00:40:12+00:00",
        ]
        assert clock == []

    def test_the_agents_status_and_both_streams_are_the_ticks(
        self, api_root: pathlib.Path, tmp_path: pathlib.Path, clock: list[int]
    ) -> None:
        logs = tmp_path / "logs"
        _test_hooks.run = FakeRun(
            [
                _test_hooks.CommandResult(
                    returncode=3, stdout="out line\n", stderr="err one\nerr two\n", timed_out=False
                )
            ]
        )

        assert tick.main(_argv(api_root, logs, "hub")) == 3

        assert _log_lines(logs, "fleet-agent") == [
            f"TICK START 2026-09-29T00:40:00+00:00 hub task-pid {os.getpid()}",
            "out line",
            "err one",
            "err two",
            "TICK EXIT 3 2026-09-29T00:40:12+00:00",
        ]

    def test_a_launcher_ended_at_its_deadline_is_logged_as_such(
        self, api_root: pathlib.Path, tmp_path: pathlib.Path, clock: list[int]
    ) -> None:
        logs = tmp_path / "logs"
        _test_hooks.run = FakeRun([timed_out(tick.TICK_WALL_SECONDS)])

        status = tick.main(_argv(api_root, logs, "announce", "--node", "lavender"))

        assert status == _test_hooks.TIMED_OUT_RETURNCODE
        lines = _log_lines(logs, "fleet-node-lavender")
        assert lines[0].startswith("TICK START 2026-09-29T00:40:00+00:00 announce lavender ")
        assert lines[1] == f"timed out after {tick.TICK_WALL_SECONDS} s"
        assert lines[2] == f"TICK EXIT {_test_hooks.TIMED_OUT_RETURNCODE} 2026-09-29T00:40:12+00:00"

    def test_a_second_tick_the_same_day_appends_to_the_same_log(
        self, api_root: pathlib.Path, tmp_path: pathlib.Path, clock: list[int]
    ) -> None:
        logs = tmp_path / "logs"
        _test_hooks.run = FakeRun([ok("first\n"), failed(1, "second\n")])
        tick.main(_argv(api_root, logs, "hub"))
        clock.extend([STARTED + 180, STARTED + 181])

        assert tick.main(_argv(api_root, logs, "hub")) == 1

        lines = _log_lines(logs, "fleet-agent")
        assert [line.split(" ")[:2] for line in lines if line.startswith("TICK")] == [
            ["TICK", "START"],
            ["TICK", "EXIT"],
            ["TICK", "START"],
            ["TICK", "EXIT"],
        ]
        assert lines[-2:] == ["second", "TICK EXIT 1 2026-09-29T00:43:01+00:00"]

    def test_a_credentials_file_the_shared_parser_refuses_runs_nothing(
        self, api_root: pathlib.Path, tmp_path: pathlib.Path, clock: list[int]
    ) -> None:
        env_file = api_root / pathlib.PurePosixPath(tick.ENV_FILE)
        env_file.write_text(CREDENTIALS + '$env:CORVIS_TENANT_ID = "double"\n', encoding="utf-8")
        runner = FakeRun([])
        _test_hooks.run = runner

        with pytest.raises(AppError) as refused:
            tick.main(_argv(api_root, tmp_path / "logs", "hub"))

        assert refused.value.code is ErrorCode.CONFIG_ERROR
        assert refused.value.message.startswith(f"line 4 of {env_file} ")
        assert runner.calls == []
        assert not (tmp_path / "logs").exists()


class TestRetention:
    def test_only_this_lanes_logs_past_the_retention_are_removed(
        self, tmp_path: pathlib.Path
    ) -> None:
        now = STARTED
        day = 86_400
        cases = {
            "fleet-node-sedona-2026-09-10.log": now - 19 * day,
            "fleet-node-sedona-2026-09-14.log": now - tick.RETENTION_DAYS * day - 1,
            "fleet-node-sedona-2026-09-15.log": now - tick.RETENTION_DAYS * day + 1,
            "fleet-node-sedona-2026-09-28.log": now - day,
            "fleet-node-lavender-2026-09-10.log": now - 19 * day,
            "fleet-agent-2026-09-10.log": now - 19 * day,
        }
        for name, written in cases.items():
            path = tmp_path / name
            path.write_text("x", encoding="utf-8")
            os.utime(path, (written, written))

        assert tick.remove_stale_logs(tmp_path, "fleet-node-sedona", now_unix=now) == 2

        assert sorted(path.name for path in tmp_path.iterdir()) == [
            "fleet-agent-2026-09-10.log",
            "fleet-node-lavender-2026-09-10.log",
            "fleet-node-sedona-2026-09-15.log",
            "fleet-node-sedona-2026-09-28.log",
        ]

    def test_a_tick_sweeps_its_lanes_old_logs_before_it_runs(
        self, api_root: pathlib.Path, tmp_path: pathlib.Path, clock: list[int]
    ) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        old = logs / "fleet-agent-2026-09-01.log"
        old.write_text("old\n", encoding="utf-8")
        written = STARTED - 20 * 86_400
        os.utime(old, (written, written))
        _test_hooks.run = FakeRun([ok("")])

        tick.main(_argv(api_root, logs, "hub"))

        assert sorted(path.name for path in logs.iterdir()) == ["fleet-agent-2026-09-29.log"]


class TestEntryPoint:
    def test_running_as_a_module_runs_a_tick_and_exits_with_its_status(
        self, api_root: pathlib.Path, tmp_path: pathlib.Path, clock: list[int]
    ) -> None:
        logs = tmp_path / "logs"
        runner = FakeRun([failed(7, "refused\n")])
        _test_hooks.run = runner
        saved_argv = sys.argv
        saved_module = sys.modules.pop("fleet.cli.tick")
        sys.argv = ["tick", *_argv(api_root, logs, "hub")]
        try:
            with pytest.raises(SystemExit) as exited:
                runpy.run_module("fleet.cli.tick", run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            sys.modules["fleet.cli.tick"] = saved_module

        assert exited.value.code == 7
        assert len(runner.calls) == 1
        assert _log_lines(logs, "fleet-agent")[-1] == "TICK EXIT 7 2026-09-29T00:40:12+00:00"
