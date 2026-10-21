"""The restart lane at the tick level: a claimed ``restart-session`` job
runs HERE, synchronously, and reports session-audit's own verdict (MCPs
mig 507, board task ccec3417).

Split by role like ``test_agent_rebuild.py``: same fixtures, same fakes --
the queue speaks the real wire shape and the session-audit invocation goes
through the same command hook every dispatch test uses. Every verb first
verifies the active sealed session-verbs release (MCPs board task
c7c2527d), planted by ``tests._session_release_fixtures``.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str, narrow_json_to_str

from fleet.cli import agent
from fleet.core import _test_hooks, restart, sealed_release, session_argv, session_release
from tests._agent_cli import (
    HELD_THEN_CLAIMED,
    HUB_HOST,
    NOTHING_HELD,
    agent_argv,
    bind_hub_tick,
    sessions_argv,
    tick_run_id,
)
from tests._hub_tick import UNROLLED_TICK, unrolled_calls
from tests._queue_fakes import FakeQueue, queue_job
from tests._session_release_fixtures import (
    RELEASE_ID,
    REVISION,
    plant_release,
    point_environment,
    verify_call,
    verify_reply,
)
from tests.conftest import FakeRun, failed

TARGET = "934d9975-0d65-4e68-83de-b74f8c4df0c4"
SUBMITTER = "fable-dm-versionsplit-0912"
#: The board task a handoff's fresh worker starts on (MCPs mig 831).
HANDOFF_TASK = "553cc92f-39d0-49f3-a631-0d6108a2c5ec"


@pytest.fixture(name="credentials_in_env", autouse=True)
def _credentials_in_env() -> None:
    """Give every test the two variables the agent refuses to run without.

    And the machine and unit it runs in, which a hub lane job is started on.
    """
    bind_hub_tick()


def opening(mode: str, returncode: int) -> str:
    """The start of a session job's closing detail.

    Args:
        mode: The session-audit mode that ran.
        returncode: Its exit code.

    Returns:
        The mode, the planted release and revision, and the exit code.
    """
    return (
        f"session-audit {mode} from sealed release {RELEASE_ID} "
        f"(revision {REVISION}) exited {returncode}:"
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
        "submittedBy": SUBMITTER,
        # A session verb acts on the hub, not on a commit (MCPs mig 532).
        "sha": None,
    }
    fields.update(overrides)
    return queue_job(**fields)


def ran_queue(command: str, closed: str) -> FakeQueue:
    """The queue's three answers for a session job that ran.

    Args:
        command: The row's verb.
        closed: The status it closes with.

    Returns:
        The claim, the start report and the close report.
    """
    return FakeQueue(
        [
            NOTHING_HELD,
            dump_json_str({"claimed": restart_row(command=command)}),
            dump_json_str({"job": restart_row(command=command, status="running", node="austinpc")}),
            dump_json_str({"job": restart_row(command=command, status=closed, node="austinpc")}),
        ]
    )


def refused_queue(**claimed: JSONValue) -> FakeQueue:
    """The queue's two answers for a session job refused before it ran.

    Args:
        **claimed: Fields to vary on the claimed row.

    Returns:
        The claim and the refusal report.
    """
    return FakeQueue(
        [
            NOTHING_HELD,
            dump_json_str({"claimed": restart_row(**claimed)}),
            dump_json_str({"job": restart_row(status="refused")}),
        ]
    )


def verb_reply(returncode: int, stdout: str) -> _test_hooks.CommandResult:
    """session-audit's answer.

    Args:
        returncode: Its exit code.
        stdout: What it printed.

    Returns:
        The result.
    """
    return _test_hooks.CommandResult(
        returncode=returncode, stdout=stdout, stderr="", timed_out=False
    )


class TestRestartLane:
    def test_a_claimed_restart_verifies_the_release_runs_it_and_closes_with_its_verdict(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        release = plant_release(tmp_path)
        runner = FakeRun(
            [
                verify_reply(tmp_path),
                verb_reply(
                    0,
                    "ROLLOVER APPLIED - 1 restart(s) attempted: "
                    "1 restarted, 0 skipped, 0 failed\n"
                    "  RESTARTED  mcps-99 (opus-board-triage-0912) pane %16 "
                    "-- now pid 41324 as mcps-d4 on 2.1.270\n",
                ),
                *UNROLLED_TICK,
            ]
        )
        _test_hooks.run = runner
        endpoint = ran_queue("restart-session", "passed")
        _test_hooks.http_post = endpoint

        # No --mcps-root: a session verb never reads the MCPs checkout.
        assert agent.main(agent_argv(config_path, repo)) == 0

        assert endpoint.tools == [*HELD_THEN_CLAIMED, "dispatch_report", "dispatch_report"]
        # The hub lane, with the hub's own tags: none. What keeps a revive
        # out of a queue of checks (board task fd5cabfa, A5).
        assert endpoint.arguments[1]["lane"] == "hub"
        assert endpoint.arguments[1]["tags"] == []
        # The release's own verifier runs first, then the verb runs the
        # release's session-audit against its register (MCPs c7c2527d).
        assert runner.calls == [
            verify_call(tmp_path),
            session_argv.restart_argv(release["root"], release["registry_dir"], HUB_HOST, TARGET),
            *unrolled_calls(repo),
        ]
        # The child inherits neither this agent's poetry venv nor any
        # PYTHONPATH, and writes no bytecode into the sealed release.
        assert runner.unset_env[:2] == [(), ("VIRTUAL_ENV", "PYTHONPATH")]
        assert runner.set_env[:2] == [(), (("PYTHONDONTWRITEBYTECODE", "1"),)]
        # And each carries its deadline, so a verifier or pane that stops
        # answering closes the job instead of holding the tick.
        assert (
            runner.timeouts[:2]
            == [sealed_release.VERIFY_TIMEOUT_SECONDS, restart.SESSION_JOB_TIMEOUT_SECONDS]
            == [300, 600]
        )
        started = endpoint.arguments[2]
        assert started["action"] == "start"
        assert started["node"] == "austinpc"
        assert started["runId"] == tick_run_id()
        closed = endpoint.arguments[3]
        assert closed["action"] == "close"
        assert closed["status"] == "passed"
        assert closed["exitCode"] == 0
        detail = narrow_json_to_str(closed["detail"])
        # The detail names the release and revision that ran, so a closure
        # can show its fix was the code that acted.
        assert detail.startswith(opening("rollover", 0))
        # The session-audit outcome line reaches the queue VERBATIM: it is
        # what the submitter reads, and it names the new pid.
        assert "RESTARTED  mcps-99" in detail
        assert "now pid 41324" in detail

    def test_a_sessions_tick_runs_the_verb_and_reports_it_started_on_its_machine(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """MCPs board task c39ee056: a session verb names no node in the
        queue, so the runner holding the sessions duty says where it ran."""
        release = plant_release(tmp_path)
        runner = FakeRun([verify_reply(tmp_path), verb_reply(0, "RESTARTED  mcps-99\n")])
        _test_hooks.run = runner
        endpoint = FakeQueue(
            [
                NOTHING_HELD,
                dump_json_str({"claimed": restart_row(requestedNode=None)}),
                dump_json_str({"job": restart_row(status="running", node="ziggurat")}),
                dump_json_str({"job": restart_row(status="passed", node="ziggurat")}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(sessions_argv(config_path, repo, "ziggurat")) == 0

        assert endpoint.tools == [*HELD_THEN_CLAIMED, "dispatch_report", "dispatch_report"]
        assert endpoint.arguments[1]["lane"] == "sessions"
        assert endpoint.arguments[2]["node"] == "ziggurat"
        assert endpoint.arguments[2]["runId"] == tick_run_id()
        # The verb and nothing after it: a sessions tick has no publish pass.
        # It acts on the host the tick runs as, which a tick on the sessions
        # duty's machine names for a session host it reaches in to (MCPs
        # board task 78a1eca7).
        assert runner.calls == [
            verify_call(tmp_path),
            session_argv.restart_argv(release["root"], release["registry_dir"], "ziggurat", TARGET),
        ]

    def test_a_claimed_revive_runs_session_audits_revive_mode_with_the_submitter(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """The second session verb (MCPs mig 525, board task 1fe89973) rides the
        same job path: one invocation, session-audit's own REVIVE line back."""
        release = plant_release(tmp_path)
        runner = FakeRun(
            [
                verify_reply(tmp_path),
                verb_reply(
                    0,
                    f"REVIVE - REVIVED session {TARGET}: now pid 15280 as mcps-b3 "
                    "on 2.1.270 in main:6; brief typed\n"
                    "  departure  last ledger event: end (other) at 2026-09-16 10:31Z\n",
                ),
                *UNROLLED_TICK,
            ]
        )
        _test_hooks.run = runner
        endpoint = ran_queue("revive-session", "passed")
        _test_hooks.http_post = endpoint

        assert agent.main(agent_argv(config_path, repo)) == 0

        assert runner.calls == [
            verify_call(tmp_path),
            session_argv.revive_argv(
                release["root"], release["registry_dir"], HUB_HOST, TARGET, SUBMITTER
            ),
            *unrolled_calls(repo),
        ]
        assert runner.unset_env[1] == restart.SESSION_ENVIRONMENT_EXCLUDED
        started = endpoint.arguments[2]
        assert started["runId"] == tick_run_id()
        closed = endpoint.arguments[3]
        assert closed["status"] == "passed"
        detail = narrow_json_to_str(closed["detail"])
        assert detail.startswith(opening("revive", 0))
        assert "REVIVE - REVIVED" in detail
        assert "now pid 15280" in detail

    @pytest.mark.parametrize("command", ["revive-session", "kill-session-hard"])
    def test_a_verb_whose_submitter_is_not_a_label_is_refused_before_it_runs(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path, command: str
    ) -> None:
        plant_release(tmp_path)
        runner = FakeRun([verify_reply(tmp_path), *UNROLLED_TICK])
        _test_hooks.run = runner
        endpoint = refused_queue(command=command, submittedBy="--hard; Not A Label")
        _test_hooks.http_post = endpoint

        assert agent.main(agent_argv(config_path, repo)) == 0

        # The release was verified; session-audit never ran.
        assert runner.calls == [verify_call(tmp_path), *unrolled_calls(repo)]
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
        release = plant_release(tmp_path)
        line = f"KILL - {outcome.format(target=TARGET)}\n"
        runner = FakeRun([verify_reply(tmp_path), verb_reply(0, line), *UNROLLED_TICK])
        _test_hooks.run = runner
        endpoint = ran_queue(command, "passed")
        _test_hooks.http_post = endpoint

        assert agent.main(agent_argv(config_path, repo)) == 0

        assert runner.calls == [
            verify_call(tmp_path),
            session_argv.kill_argv(
                release["root"],
                release["registry_dir"],
                HUB_HOST,
                TARGET,
                SUBMITTER,
                hard=hard,
            ),
            *unrolled_calls(repo),
        ]
        assert runner.set_env[1] == restart.SESSION_ENVIRONMENT_SET
        assert endpoint.arguments[2]["runId"] == tick_run_id()
        closed = endpoint.arguments[3]
        assert closed["status"] == "passed"
        detail = narrow_json_to_str(closed["detail"])
        assert detail == f"{opening('kill', 0)} {line.strip()}"

    def test_a_claimed_compact_runs_session_audits_compact_mode_and_closes_with_its_line(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """MCPs mig 564, board task 01f31e4a: the compact rides the one
        session job path, runs ``session-audit compact`` with the submitter,
        and the closing detail carries the executor's outcome line, which
        the room's supervisor lifts into its digest."""
        release = plant_release(tmp_path)
        line = f"COMPACT - COMPACTED session {TARGET}: 640000 -> 90000 tokens\n"
        runner = FakeRun([verify_reply(tmp_path), verb_reply(0, line), *UNROLLED_TICK])
        _test_hooks.run = runner
        endpoint = ran_queue("compact-session", "passed")
        _test_hooks.http_post = endpoint

        assert agent.main(agent_argv(config_path, repo)) == 0

        assert runner.calls == [
            verify_call(tmp_path),
            session_argv.compact_argv(
                release["root"], release["registry_dir"], HUB_HOST, TARGET, SUBMITTER
            ),
            *unrolled_calls(repo),
        ]
        assert endpoint.arguments[2]["runId"] == tick_run_id()
        closed = endpoint.arguments[3]
        assert closed["status"] == "passed"
        detail = narrow_json_to_str(closed["detail"])
        assert detail == f"{opening('compact', 0)} {line.strip()}"

    def test_a_claimed_move_runs_session_audits_move_mode_and_closes_with_its_line(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """MCPs mig 826, board task 4c26ab11: the move rides the one session
        job path, runs ``session-audit move`` with the submitter, and the
        closing detail carries the executor's outcome line."""
        release = plant_release(tmp_path)
        line = f"MOVE - MOVED session {TARGET}: restarted through the account launcher\n"
        runner = FakeRun([verify_reply(tmp_path), verb_reply(0, line), *UNROLLED_TICK])
        _test_hooks.run = runner
        endpoint = ran_queue("move-session", "passed")
        _test_hooks.http_post = endpoint

        assert agent.main(agent_argv(config_path, repo)) == 0

        assert runner.calls == [
            verify_call(tmp_path),
            session_argv.move_argv(
                release["root"], release["registry_dir"], HUB_HOST, TARGET, SUBMITTER
            ),
            *unrolled_calls(repo),
        ]
        assert endpoint.arguments[2]["runId"] == tick_run_id()
        closed = endpoint.arguments[3]
        assert closed["status"] == "passed"
        detail = narrow_json_to_str(closed["detail"])
        assert detail == f"{opening('move', 0)} {line.strip()}"

    def test_a_claimed_handoff_runs_session_audits_handoff_mode_with_its_task(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """MCPs mig 831, board task 553cc92f: the handoff rides the one session
        job path and runs ``session-audit handoff`` with the row's task."""
        release = plant_release(tmp_path)
        line = f"HANDOFF - STARTED opus-coordination-w1-1020 for task {HANDOFF_TASK}\n"
        runner = FakeRun([verify_reply(tmp_path), verb_reply(0, line), *UNROLLED_TICK])
        _test_hooks.run = runner
        row = {"command": "handoff-session", "taskId": HANDOFF_TASK}
        endpoint = FakeQueue(
            [
                NOTHING_HELD,
                dump_json_str({"claimed": restart_row(**row)}),
                dump_json_str({"job": restart_row(**row, status="running", node="austinpc")}),
                dump_json_str({"job": restart_row(**row, status="passed", node="austinpc")}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(agent_argv(config_path, repo)) == 0

        assert runner.calls == [
            verify_call(tmp_path),
            session_argv.handoff_argv(
                release["root"], release["registry_dir"], HUB_HOST, TARGET, HANDOFF_TASK, SUBMITTER
            ),
            *unrolled_calls(repo),
        ]
        assert endpoint.arguments[2]["runId"] == tick_run_id()
        detail = narrow_json_to_str(endpoint.arguments[3]["detail"])
        assert detail == f"{opening('handoff', 0)} {line.strip()}"

    def test_an_approved_exit_runs_the_graceful_kill_under_its_own_run_verb(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """MCPs board task 5aa8ed06: a session's approved self-exit is the
        graceful kill's keystrokes, never the hard verb, and its run id says
        exit so the job reads as the session's own ending."""
        release = plant_release(tmp_path)
        line = f"KILL - ENDED session {TARGET} (graceful): /exit typed into pane %7\n"
        runner = FakeRun([verify_reply(tmp_path), verb_reply(0, line), *UNROLLED_TICK])
        _test_hooks.run = runner
        endpoint = ran_queue("exit-session", "passed")
        _test_hooks.http_post = endpoint

        assert agent.main(agent_argv(config_path, repo)) == 0

        assert runner.calls[1] == session_argv.kill_argv(
            release["root"], release["registry_dir"], HUB_HOST, TARGET, SUBMITTER, hard=False
        )
        assert endpoint.arguments[2]["runId"] == tick_run_id()
        assert narrow_json_to_str(endpoint.arguments[3]["detail"]) == (
            f"{opening('kill', 0)} {line.strip()}"
        )

    def test_a_kill_session_audit_did_not_carry_out_closes_failed(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        plant_release(tmp_path)
        refused = f"KILL - REFUSED session {TARGET} (graceful): nothing typed: busy right now\n"
        _test_hooks.run = FakeRun([verify_reply(tmp_path), verb_reply(1, refused), *UNROLLED_TICK])
        endpoint = ran_queue("kill-session", "failed")
        _test_hooks.http_post = endpoint

        assert agent.main(agent_argv(config_path, repo)) == 0

        closed = endpoint.arguments[3]
        assert closed["status"] == "failed"
        assert closed["exitCode"] == 1
        assert "KILL - REFUSED" in narrow_json_to_str(closed["detail"])

    def test_a_skipped_or_failed_restart_closes_failed_with_session_audits_reason(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """A2 on the board task: the busy session is SKIPPED by session-audit
        and this runner reports that, never a success it did not observe."""
        plant_release(tmp_path)
        _test_hooks.run = FakeRun(
            [
                verify_reply(tmp_path),
                verb_reply(
                    1,
                    "ROLLOVER APPLIED - 1 restart(s) attempted: "
                    "0 restarted, 1 skipped, 0 failed\n"
                    "  SKIPPED    mcps-6e (fable-dm-versionsplit-0912) pane %32 "
                    "-- was busy at apply time, not idle; untouched\n",
                ),
                *UNROLLED_TICK,
            ]
        )
        endpoint = ran_queue("restart-session", "failed")
        _test_hooks.http_post = endpoint

        assert agent.main(agent_argv(config_path, repo)) == 0

        closed = endpoint.arguments[3]
        assert closed["status"] == "failed"
        assert closed["exitCode"] == 1
        assert "SKIPPED    mcps-6e" in narrow_json_to_str(closed["detail"])
        assert "untouched" in narrow_json_to_str(closed["detail"])


class TestReleaseRefusals:
    """No fallback (MCPs board task c7c2527d): a verb with no verified
    sealed release is refused by code, and nothing runs, not even the
    verifier when there is no release to verify."""

    def test_with_no_active_release_the_job_is_refused_and_nothing_runs(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        point_environment(tmp_path)
        runner = FakeRun([*UNROLLED_TICK])
        _test_hooks.run = runner
        endpoint = refused_queue(command="kill-session")
        _test_hooks.http_post = endpoint

        assert agent.main(agent_argv(config_path, repo)) == 0

        assert runner.calls == [*unrolled_calls(repo)]
        assert endpoint.tools == [*HELD_THEN_CLAIMED, "dispatch_report"]
        closed = endpoint.arguments[2]
        assert closed["status"] == "refused"
        assert "exitCode" not in closed
        detail = narrow_json_to_str(closed["detail"])
        assert detail.startswith(f"{session_release.NOT_ACTIVE_CODE}: ")
        assert "nothing was run" in detail

    def test_a_release_that_fails_its_seal_refuses_the_job_and_no_verb_runs(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        plant_release(tmp_path)
        runner = FakeRun(
            [failed(1, "RELEASE_DIGEST_MISMATCH: the payload changed"), *UNROLLED_TICK]
        )
        _test_hooks.run = runner
        endpoint = refused_queue(command="exit-session")
        _test_hooks.http_post = endpoint

        assert agent.main(agent_argv(config_path, repo)) == 0

        assert runner.calls == [verify_call(tmp_path), *unrolled_calls(repo)]
        closed = endpoint.arguments[2]
        assert closed["status"] == "refused"
        detail = narrow_json_to_str(closed["detail"])
        assert detail.startswith(f"{sealed_release.UNSEALED_CODE}: ")
        assert "RELEASE_DIGEST_MISMATCH: the payload changed" in detail

    def test_a_lawless_target_is_refused_before_the_release_is_even_read(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        plant_release(tmp_path)
        runner = FakeRun([*UNROLLED_TICK])
        _test_hooks.run = runner
        endpoint = refused_queue(sessionTarget="mcps-99; rm -rf /")
        _test_hooks.http_post = endpoint

        assert agent.main(agent_argv(config_path, repo)) == 0

        assert runner.calls == [*unrolled_calls(repo)]
        closed = endpoint.arguments[2]
        assert "RESTART_TARGET_INVALID" in narrow_json_to_str(closed["detail"])

    def test_a_row_with_no_target_is_refused_by_name(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        plant_release(tmp_path)
        runner = FakeRun([*UNROLLED_TICK])
        _test_hooks.run = runner
        endpoint = refused_queue(sessionTarget=None)
        _test_hooks.http_post = endpoint

        assert agent.main(agent_argv(config_path, repo)) == 0

        assert runner.calls == [*unrolled_calls(repo)]
        assert "RESTART_TARGET_MISSING" in narrow_json_to_str(endpoint.arguments[2]["detail"])

    def test_a_handoff_naming_no_task_is_refused_before_the_release_is_read(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """MCPs mig 831: a handoff whose row names no task starts no worker."""
        plant_release(tmp_path)
        runner = FakeRun([*UNROLLED_TICK])
        _test_hooks.run = runner
        endpoint = refused_queue(command="handoff-session", taskId=None)
        _test_hooks.http_post = endpoint

        assert agent.main(agent_argv(config_path, repo)) == 0

        assert runner.calls == [*unrolled_calls(repo)]
        assert "HANDOFF_TASK_MISSING" in narrow_json_to_str(endpoint.arguments[2]["detail"])
