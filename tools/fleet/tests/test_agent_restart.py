"""The restart lane at the tick level: a claimed ``restart-session`` job
runs HERE, synchronously, and reports session-audit's own verdict (MCPs
mig 507, board task ccec3417).

Split by role like ``test_agent_rebuild.py``: same fixtures, same fakes --
the queue speaks the real wire shape and the session-audit invocation goes
through the same command hook every dispatch test uses.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str, narrow_json_to_str

from fleet.cli import agent
from fleet.core import _test_hooks, queue, restart
from tests._queue_fakes import DEFAULT_JOB_ID, FakeEnv, FakeQueue, queue_job
from tests.conftest import FakeRun, agent_argv
from tests.test_agent_rebuild import rebuild_argv as hub_argv

TARGET = "934d9975-0d65-4e68-83de-b74f8c4df0c4"


@pytest.fixture(name="credentials_in_env", autouse=True)
def _credentials_in_env() -> None:
    """Give every test the two variables the agent refuses to run without."""
    _test_hooks.env = FakeEnv(
        {queue.API_KEY_VARIABLE: "test-key", queue.TENANT_ID_VARIABLE: "tenant"}
    )


def restart_row(**overrides: JSONValue) -> JSONObject:
    """A claimed restart-session row in the real wire shape.

    Args:
        **overrides: Fields to vary.

    Returns:
        The wire object.
    """
    fields: dict[str, JSONValue] = {
        "status": "claimed",
        "command": "restart-session",
        "project": "MCPs",
        "requestedNode": "austinpc",
        "sessionTarget": TARGET,
        "submittedBy": "fable-dm-versionsplit-0912",
    }
    fields.update(overrides)
    return queue_job(**fields)


class TestRestartLane:
    def test_a_claimed_restart_runs_session_audit_and_closes_with_its_verdict(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        mcps = tmp_path / "mcps-checkout"
        mcps.mkdir()
        runner = FakeRun(
            [
                _test_hooks.CommandResult(
                    returncode=0,
                    stdout=(
                        "ROLLOVER APPLIED - 1 restart(s) attempted: "
                        "1 restarted, 0 skipped, 0 failed\n"
                        "  RESTARTED  mcps-99 (opus-board-triage-0912) pane %16 "
                        "-- now pid 41324 as mcps-d4 on 2.1.270\n"
                    ),
                    stderr="",
                    timed_out=False,
                )
            ]
        )
        _test_hooks.run = runner
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": restart_row()}),
                dump_json_str({"job": restart_row(status="running", node="austinpc")}),
                dump_json_str({"job": restart_row(status="passed", node="austinpc")}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(hub_argv(config_path, repo, mcps)) == 0

        assert endpoint.tools == [
            "dispatch_list",
            "dispatch_claim",
            "dispatch_report",
            "dispatch_report",
        ]
        assert runner.calls == [restart.restart_argv(mcps, TARGET)]
        # The child must not inherit this agent's own poetry venv, or the
        # session-audit script resolves inside the wrong environment.
        assert runner.unset_env == [("VIRTUAL_ENV",)]
        # And it carries the lane's deadline, so a pane or hop that stops
        # answering closes the job failed instead of holding the tick.
        assert runner.timeouts == [restart.SESSION_JOB_TIMEOUT_SECONDS] == [600]
        started = endpoint.arguments[2]
        assert started["action"] == "start"
        assert started["node"] == "austinpc"
        assert started["runId"] == f"restart-{DEFAULT_JOB_ID}"
        closed = endpoint.arguments[3]
        assert closed["action"] == "close"
        assert closed["status"] == "passed"
        assert closed["exitCode"] == 0
        detail = narrow_json_to_str(closed["detail"])
        assert detail.startswith("session-audit rollover exited 0:")
        # The session-audit outcome line reaches the queue VERBATIM: it is
        # what the submitter reads, and it names the new pid.
        assert "RESTARTED  mcps-99" in detail
        assert "now pid 41324" in detail

    def test_a_claimed_revive_runs_session_audits_revive_mode_with_the_submitter(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """The second session verb (MCPs mig 525, board task 1fe89973) rides the
        same job path: one invocation, session-audit's own REVIVE line back."""
        mcps = tmp_path / "mcps-checkout"
        mcps.mkdir()
        runner = FakeRun(
            [
                _test_hooks.CommandResult(
                    returncode=0,
                    stdout=(
                        f"REVIVE - REVIVED session {TARGET}: now pid 15280 as mcps-b3 "
                        "on 2.1.270 in main:6; brief typed\n"
                        "  departure  last ledger event: end (other) at 2026-09-16 10:31Z\n"
                    ),
                    stderr="",
                    timed_out=False,
                )
            ]
        )
        _test_hooks.run = runner
        revive_row = restart_row(command="revive-session")
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": revive_row}),
                dump_json_str(
                    {
                        "job": restart_row(
                            command="revive-session", status="running", node="austinpc"
                        )
                    }
                ),
                dump_json_str(
                    {"job": restart_row(command="revive-session", status="passed", node="austinpc")}
                ),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(hub_argv(config_path, repo, mcps)) == 0

        assert runner.calls == [restart.revive_argv(mcps, TARGET, "fable-dm-versionsplit-0912")]
        assert runner.unset_env == [("VIRTUAL_ENV",)]
        started = endpoint.arguments[2]
        assert started["runId"] == f"revive-{DEFAULT_JOB_ID}"
        closed = endpoint.arguments[3]
        assert closed["status"] == "passed"
        detail = narrow_json_to_str(closed["detail"])
        assert detail.startswith("session-audit revive exited 0:")
        assert "REVIVE - REVIVED" in detail
        assert "now pid 15280" in detail

    def test_a_revive_whose_submitter_is_not_a_label_is_refused_before_it_runs(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        mcps = tmp_path / "mcps-checkout"
        mcps.mkdir()
        runner = FakeRun([])
        _test_hooks.run = runner
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str(
                    {"claimed": restart_row(command="revive-session", submittedBy="Not A Label")}
                ),
                dump_json_str({"job": restart_row(command="revive-session", status="refused")}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(hub_argv(config_path, repo, mcps)) == 0

        assert runner.calls == []
        closed = endpoint.arguments[2]
        assert closed["status"] == "refused"
        assert "SESSION_REQUESTER_INVALID" in narrow_json_to_str(closed["detail"])

    @pytest.mark.parametrize(
        ("command", "hard", "outcome"),
        [
            ("kill-session", False, "ENDED session {target} (graceful): /exit typed into pane %16"),
            ("kill-session-hard", True, "ENDED session {target} (hard): taskkill ended pid 26576"),
        ],
    )
    def test_a_claimed_kill_runs_session_audits_kill_mode_in_the_rows_own_mode(
        self,
        config_path: pathlib.Path,
        repo: pathlib.Path,
        tmp_path: pathlib.Path,
        command: str,
        hard: bool,
        outcome: str,
    ) -> None:
        """The two kill verbs (MCPs mig 526, board task 660964d9) ride the same
        job path; the hard flag comes from the row's verb and nowhere else."""
        mcps = tmp_path / "mcps-checkout"
        mcps.mkdir()
        line = f"KILL - {outcome.format(target=TARGET)}\n"
        runner = FakeRun(
            [_test_hooks.CommandResult(returncode=0, stdout=line, stderr="", timed_out=False)]
        )
        _test_hooks.run = runner
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": restart_row(command=command)}),
                dump_json_str(
                    {"job": restart_row(command=command, status="running", node="austinpc")}
                ),
                dump_json_str(
                    {"job": restart_row(command=command, status="passed", node="austinpc")}
                ),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(hub_argv(config_path, repo, mcps)) == 0

        assert runner.calls == [
            restart.kill_argv(mcps, TARGET, "fable-dm-versionsplit-0912", hard=hard)
        ]
        assert runner.unset_env == [("VIRTUAL_ENV",)]
        assert endpoint.arguments[2]["runId"] == f"kill-{DEFAULT_JOB_ID}"
        closed = endpoint.arguments[3]
        assert closed["status"] == "passed"
        detail = narrow_json_to_str(closed["detail"])
        assert detail == f"session-audit kill exited 0: {line.strip()}"

    def test_a_kill_session_audit_did_not_carry_out_closes_failed(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        mcps = tmp_path / "mcps-checkout"
        mcps.mkdir()
        refused = f"KILL - REFUSED session {TARGET} (graceful): nothing typed: busy right now\n"
        _test_hooks.run = FakeRun(
            [_test_hooks.CommandResult(returncode=1, stdout=refused, stderr="", timed_out=False)]
        )
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": restart_row(command="kill-session")}),
                dump_json_str(
                    {"job": restart_row(command="kill-session", status="running", node="austinpc")}
                ),
                dump_json_str(
                    {"job": restart_row(command="kill-session", status="failed", node="austinpc")}
                ),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(hub_argv(config_path, repo, mcps)) == 0

        closed = endpoint.arguments[3]
        assert closed["status"] == "failed"
        assert closed["exitCode"] == 1
        assert "KILL - REFUSED" in narrow_json_to_str(closed["detail"])

    def test_a_kill_whose_submitter_is_not_a_label_is_refused_before_it_runs(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        mcps = tmp_path / "mcps-checkout"
        mcps.mkdir()
        runner = FakeRun([])
        _test_hooks.run = runner
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str(
                    {"claimed": restart_row(command="kill-session-hard", submittedBy="--hard; rm")}
                ),
                dump_json_str({"job": restart_row(command="kill-session-hard", status="refused")}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(hub_argv(config_path, repo, mcps)) == 0

        assert runner.calls == []
        assert "SESSION_REQUESTER_INVALID" in narrow_json_to_str(endpoint.arguments[2]["detail"])

    def test_a_skipped_or_failed_restart_closes_failed_with_session_audits_reason(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """A2 on the board task: the busy session is SKIPPED by session-audit
        and this runner reports that, never a success it did not observe."""
        mcps = tmp_path / "mcps-checkout"
        mcps.mkdir()
        _test_hooks.run = FakeRun(
            [
                _test_hooks.CommandResult(
                    returncode=1,
                    stdout=(
                        "ROLLOVER APPLIED - 1 restart(s) attempted: "
                        "0 restarted, 1 skipped, 0 failed\n"
                        "  SKIPPED    mcps-6e (fable-dm-versionsplit-0912) pane %32 "
                        "-- was busy at apply time, not idle; untouched\n"
                    ),
                    stderr="",
                    timed_out=False,
                )
            ]
        )
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": restart_row()}),
                dump_json_str({"job": restart_row(status="running", node="austinpc")}),
                dump_json_str({"job": restart_row(status="failed", node="austinpc")}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(hub_argv(config_path, repo, mcps)) == 0

        closed = endpoint.arguments[3]
        assert closed["status"] == "failed"
        assert closed["exitCode"] == 1
        assert "SKIPPED    mcps-6e" in narrow_json_to_str(closed["detail"])
        assert "untouched" in narrow_json_to_str(closed["detail"])

    def test_without_the_mcps_root_flag_the_job_is_refused_and_nothing_runs(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        runner = FakeRun([])
        _test_hooks.run = runner
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": restart_row()}),
                dump_json_str({"job": restart_row(status="refused")}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(agent_argv(config_path, repo)) == 0

        assert runner.calls == []
        closed = endpoint.arguments[2]
        assert closed["status"] == "refused"
        assert "exitCode" not in closed
        assert "RESTART_ROOT_MISSING" in narrow_json_to_str(closed["detail"])

    def test_a_lawless_target_is_refused_before_session_audit_runs(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        mcps = tmp_path / "mcps-checkout"
        mcps.mkdir()
        runner = FakeRun([])
        _test_hooks.run = runner
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": restart_row(sessionTarget="mcps-99; rm -rf /")}),
                dump_json_str({"job": restart_row(status="refused")}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(hub_argv(config_path, repo, mcps)) == 0

        assert runner.calls == []
        closed = endpoint.arguments[2]
        assert "RESTART_TARGET_INVALID" in narrow_json_to_str(closed["detail"])

    def test_a_row_with_no_target_is_refused_by_name(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        mcps = tmp_path / "mcps-checkout"
        mcps.mkdir()
        runner = FakeRun([])
        _test_hooks.run = runner
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": restart_row(sessionTarget=None)}),
                dump_json_str({"job": restart_row(status="refused")}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(hub_argv(config_path, repo, mcps)) == 0

        assert runner.calls == []
        assert "RESTART_TARGET_MISSING" in narrow_json_to_str(endpoint.arguments[2]["detail"])
