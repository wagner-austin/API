"""The rebuild lane at the tick level: a claimed ``build-bases`` job runs
HERE, synchronously, as the submitter (MCPs board task 3c9033ff).

Split from ``test_agent.py`` at the file-size ceiling, by role: that module
holds the node-dispatch outcomes, this one holds the hub-local verb. Same
fixtures, same fakes -- the queue speaks the real wire shape and the make
invocation goes through the same command hook every dispatch test uses.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import dump_json_str, narrow_json_to_str

from fleet.cli import agent
from fleet.core import _test_hooks, rebuild
from tests._queue_fakes import DEFAULT_JOB_ID, FakeQueue, queue_env, queue_job
from tests.conftest import FakeRun, agent_argv


@pytest.fixture(name="credentials_in_env", autouse=True)
def _credentials_in_env() -> None:
    """Give every test the two variables the agent refuses to run without."""
    _test_hooks.env = queue_env()


def rebuild_argv(config_path: pathlib.Path, repo: pathlib.Path, mcps: pathlib.Path) -> list[str]:
    """The tick's arguments with the MCPs checkout the rebuild arm needs.

    Args:
        config_path: The workspace document.
        repo: The monorepo root on this machine.
        mcps: The MCPs checkout the runner may rebuild in.

    Returns:
        The argument list.
    """
    return [*agent_argv(config_path, repo), agent.MCPS_ROOT_FLAG, str(mcps)]


class TestRebuildLane:
    def test_a_claimed_rebuild_runs_make_as_the_submitter_and_closes_passed(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        mcps = tmp_path / "mcps-checkout"
        mcps.mkdir()
        runner = FakeRun(
            [
                _test_hooks.CommandResult(
                    returncode=0, stdout="Base images rebuilt.\n", stderr="", timed_out=False
                )
            ]
        )
        _test_hooks.run = runner
        endpoint = FakeQueue(
            [
                dump_json_str(
                    {
                        "claimed": queue_job(
                            status="claimed",
                            command="build-bases",
                            project="MCPs",
                            requestedNode="austinpc",
                            submittedBy="opus-phone-0911",
                        )
                    }
                ),
                dump_json_str({"job": queue_job(status="running", node="austinpc")}),
                dump_json_str({"job": queue_job(status="passed", node="austinpc")}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(rebuild_argv(config_path, repo, mcps)) == 0

        assert endpoint.tools == ["dispatch_claim", "dispatch_report", "dispatch_report"]
        assert runner.calls == [
            (
                "make",
                "-C",
                str(mcps),
                "build-bases",
                "BOARD_AGENT_LABEL=opus-phone-0911",
            )
        ]
        # A bake that wedges under the fleet lock ends at the lane's own
        # deadline rather than holding the tick to the scheduler's ceiling.
        assert runner.timeouts == [rebuild.BUILD_BASES_TIMEOUT_SECONDS] == [1800]
        started = endpoint.arguments[1]
        assert started["action"] == "start"
        assert started["node"] == "austinpc"
        assert started["runId"] == f"bases-{DEFAULT_JOB_ID}"
        closed = endpoint.arguments[2]
        assert closed["action"] == "close"
        assert closed["status"] == "passed"
        assert closed["exitCode"] == 0
        assert "make build-bases exited 0" in narrow_json_to_str(closed["detail"])

    def test_a_failing_bake_closes_failed_with_its_exit_code(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        mcps = tmp_path / "mcps-checkout"
        mcps.mkdir()
        _test_hooks.run = FakeRun(
            [
                _test_hooks.CommandResult(
                    returncode=2,
                    stdout="",
                    stderr="make: *** [Makefile:73: build-bases] Error 2",
                    timed_out=False,
                )
            ]
        )
        endpoint = FakeQueue(
            [
                dump_json_str(
                    {"claimed": queue_job(status="claimed", command="build-bases", project="MCPs")}
                ),
                dump_json_str({"job": queue_job(status="running", node="austinpc")}),
                dump_json_str({"job": queue_job(status="failed", node="austinpc")}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(rebuild_argv(config_path, repo, mcps)) == 0

        closed = endpoint.arguments[2]
        assert closed["status"] == "failed"
        assert closed["exitCode"] == 2
        assert "Error 2" in narrow_json_to_str(closed["detail"])

    def test_without_the_mcps_root_flag_the_job_is_refused_and_nothing_runs(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        runner = FakeRun([])
        _test_hooks.run = runner
        endpoint = FakeQueue(
            [
                dump_json_str(
                    {"claimed": queue_job(status="claimed", command="build-bases", project="MCPs")}
                ),
                dump_json_str({"job": queue_job(status="refused")}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(agent_argv(config_path, repo)) == 0

        assert runner.calls == []
        closed = endpoint.arguments[1]
        assert closed["status"] == "refused"
        assert "exitCode" not in closed
        assert "REBUILD_ROOT_MISSING" in narrow_json_to_str(closed["detail"])

    def test_a_lawless_submitter_label_is_refused_before_the_make(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        mcps = tmp_path / "mcps-checkout"
        mcps.mkdir()
        runner = FakeRun([])
        _test_hooks.run = runner
        endpoint = FakeQueue(
            [
                dump_json_str(
                    {
                        "claimed": queue_job(
                            status="claimed",
                            command="build-bases",
                            project="MCPs",
                            submittedBy="Opus Fleet 0911",
                        )
                    }
                ),
                dump_json_str({"job": queue_job(status="refused")}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(rebuild_argv(config_path, repo, mcps)) == 0

        assert runner.calls == []
        closed = endpoint.arguments[1]
        assert "REBUILD_LABEL_INVALID" in narrow_json_to_str(closed["detail"])
