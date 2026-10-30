"""A settle closes the queue before it changes anything (MCPs board task 8993c306).

Until 2026-10-05 a settle retired the run's directory and finished its
ledger row before it closed the queue job, so a close the queue did not
answer left a running job with no live row behind it, which no later pass
settles, and a directory whose result was gone. These cases run the real
node runner (:func:`fleet.cli.node_agent.main`) on a launched run whose
result is written, with the queue bound through the production transport
(:func:`fleet.core.queue_transport.answering`) and its close left
unanswered: nothing on this machine or the node changes, the start ends
cleanly, and the next start settles the same run with the same verdict.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import dump_json_str, narrow_json_to_str

from fleet.cli import node_agent
from fleet.core import _test_hooks
from fleet.core.queue_transport import answering
from tests._node_agent_fixtures import (
    PASSING_TAIL,
    PROBED,
    VERDICT_TASK,
    _credentials_in_env,
    _sourced_config,
    held_answer,
    launch,
    node_argv,
)
from tests._queue_fakes import FakeQueue, Unanswered, queue_job
from tests.conftest import DEMO_RUN_ID, FakeRun, failed, ok, retire_replies

__all__ = ["_credentials_in_env", "_sourced_config"]

#: The node's answers up to the verdict: the result read and the tail read,
#: each a script sent and then run.
READ_AND_TAIL = (ok(""), ok("0 1757000060"), ok(""), ok(PASSING_TAIL))


def _ledger(config_path: pathlib.Path) -> str:
    """The runner's ledger.

    Args:
        config_path: The workspace document beside it.

    Returns:
        Its text.
    """
    return (config_path.parent / "runs" / "ledger.jsonl").read_text(encoding="utf-8")


class TestACloseTheQueueDidNotAnswer:
    def test_leaves_the_row_live_and_the_run_unretired_and_the_next_start_settles_it(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        launch(sourced_config)
        launched = _ledger(sourced_config)
        refused_node = FakeRun(list(READ_AND_TAIL))
        _test_hooks.run = refused_node
        refusing = FakeQueue([held_answer(taskId=VERDICT_TASK), Unanswered()])
        _test_hooks.http_post = answering(refusing)

        with caplog.at_level("INFO"):
            refused_exit = node_agent.main(node_argv(sourced_config))

        assert refused_exit == 0
        assert refusing.tools == ["dispatch_list", "dispatch_report"]
        assert refusing.arguments[1]["action"] == "close"
        # Neither the retire nor the generations' removal ran, and the row is
        # as launched.
        assert len(refused_node.calls) == len(READ_AND_TAIL)
        assert _ledger(sourced_config) == launched
        assert any(
            record.getMessage().startswith(
                "lavender: the queue did not answer its collect pass; it runs again at the "
                "next poll: "
            )
            for record in caplog.records
        )

        settling_node = FakeRun([*READ_AND_TAIL, *retire_replies(), *PROBED])
        _test_hooks.run = settling_node
        answering_queue = FakeQueue(
            [
                held_answer(taskId=VERDICT_TASK),
                dump_json_str({"job": queue_job(status="passed")}),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = answering(answering_queue)

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert answering_queue.tools == ["dispatch_list", "dispatch_report", "dispatch_claim"]
        refused_line = narrow_json_to_str(refusing.arguments[1]["detail"])
        settled_line = narrow_json_to_str(answering_queue.arguments[1]["detail"])
        assert settled_line == refused_line
        assert settled_line.endswith(
            f"log=lavender:/home/corvis/fleet/stage/logs/{DEMO_RUN_ID}.log run={DEMO_RUN_ID} "
            "workers=7 slowest=unread ci=unread"
        )
        retire_path = f"/home/corvis/fleet/stage/retire-{DEMO_RUN_ID}.sh"
        assert [retire_path in " ".join(call) for call in settling_node.calls[:6]] == [
            False,
            False,
            False,
            False,
            True,
            True,
        ]
        assert "passed" in _ledger(sourced_config)


#: A node that does not answer the retire's send: the retire is owed.
UNANSWERED_RETIRE = failed(255, "Connection timed out during banner exchange")


class TestASettleLogsEachFactOnce:
    """The close logs the verdict (fleet.cli.job_close, MCPs board task
    af8b5662) and the collect pass logs what followed it, so a settle that
    closed through job_close says each once rather than the verdict twice."""

    @pytest.mark.parametrize(
        ("after_close", "followed"),
        [
            ([*retire_replies(), *PROBED], "its row is finished and its directory retired"),
            (
                [UNANSWERED_RETIRE, *PROBED],
                "its row is finished and the retire is owed until lavender answers",
            ),
        ],
    )
    def test_the_verdict_once_then_what_followed(
        self,
        sourced_config: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
        after_close: list[_test_hooks.CommandResult],
        followed: str,
    ) -> None:
        launch(sourced_config)
        _test_hooks.run = FakeRun([*READ_AND_TAIL, *after_close])
        answering_queue = FakeQueue(
            [
                held_answer(taskId=VERDICT_TASK),
                dump_json_str({"job": queue_job(status="passed")}),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = answering(answering_queue)

        with caplog.at_level("INFO"):
            assert node_agent.main(node_argv(sourced_config)) == 0

        verdict = narrow_json_to_str(answering_queue.arguments[1]["detail"])
        messages = [record.getMessage() for record in caplog.records]
        verdict_lines = [message for message in messages if verdict in message]
        followed_lines = [m for m in messages if m.endswith(f"run={DEMO_RUN_ID}: {followed}")]
        assert len(verdict_lines) == 1
        assert verdict_lines[0].endswith(f"run={DEMO_RUN_ID}: {verdict}")
        assert len(followed_lines) == 1


#: The line systemd's ``ExecStopPost=`` appends for a unit ended for memory
#: (MCPs board tasks c8585623 and d111690a): the ending lavender's helper
#: gave a unit filled past its cap on 2026-10-25T14:50:49Z.
MEMORY_ENDED = (
    "FLEET_UNIT_ENDED: unit fleet-MCPs-mcp-shared-colossus-1792939198 ended with systemd "
    "result oom-kill (killed TERM) before the build wrote its status; recorded exit 143"
)

#: The line for a unit a signal ended, which says nothing about memory.
SIGNAL_ENDED = (
    "FLEET_UNIT_ENDED: unit fleet-MCPs-mcp-shared-diphtheria-1791171734 ended with systemd "
    "result signal (killed KILL) before the build wrote its status; recorded exit 137"
)


def _killed_tail(ending: str) -> str:
    """The tail of a build the unit-end script closed: 859e7ab3's transcript, then the line.

    Args:
        ending: The unit-end line.

    Returns:
        The tail.
    """
    return (
        "fleet-prepare: no stored install for tree key e7fd2baa9e87; running npm ci\n"
        "npm warn deprecated node-domexception@1.0.0: Use your platform's native DOMException\n"
        f"{ending}\n"
    )


def _settle_killed(
    sourced_config: pathlib.Path, *, ending: str, recorded: int, answered: str
) -> FakeQueue:
    """Settle a launched run whose unit ended with ``ending``, the queue answering ``answered``.

    Args:
        sourced_config: The workspace document.
        ending: The unit-end line the tail carries.
        recorded: The exit the unit-end script recorded.
        answered: The queue's answer to the report.

    Returns:
        The queue, holding every call the runner made.
    """
    launch(sourced_config)
    _test_hooks.run = FakeRun(
        [
            ok(""),
            ok(f"{recorded} 1757000060"),
            ok(""),
            ok(_killed_tail(ending)),
            *retire_replies(),
            *PROBED,
        ]
    )
    answering_queue = FakeQueue(
        [held_answer(taskId=VERDICT_TASK), answered, dump_json_str({"claimed": None})]
    )
    _test_hooks.http_post = answering(answering_queue)
    assert node_agent.main(node_argv(sourced_config)) == 0
    return answering_queue


#: Where every verdict line here ends, before the unit's ending.
VERDICT_END = (
    f"log=lavender:/home/corvis/fleet/stage/logs/{DEMO_RUN_ID}.log run={DEMO_RUN_ID} workers=7 "
    "slowest=unread ci=unread"
)


class TestABuildItsUnitClosed:
    def test_a_signal_s_kill_closes_failed_with_the_unit_s_ending_on_the_line(
        self, sourced_config: pathlib.Path
    ) -> None:
        answered = dump_json_str({"job": queue_job(status="failed", exitCode=137)})
        answering_queue = _settle_killed(
            sourced_config, ending=SIGNAL_ENDED, recorded=137, answered=answered
        )

        close = answering_queue.arguments[1]
        assert close["action"] == "close"
        assert close["status"] == "failed"
        assert close["exitCode"] == 137
        assert narrow_json_to_str(close["detail"]).endswith(f"{VERDICT_END} {SIGNAL_ENDED}")
        assert "failed" in _ledger(sourced_config)


class TestABuildEndedForMemory:
    """A check yields memory to the stack (MCPs board task d111690a), so a unit
    the kernel or systemd-oomd ended for memory is reported lost: the queue
    requeues it the first time and fails it the second."""

    def test_the_first_kill_is_given_back_and_the_queue_requeues_it(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        answered = dump_json_str({"job": queue_job(status="queued")})
        with caplog.at_level("INFO"):
            answering_queue = _settle_killed(
                sourced_config, ending=MEMORY_ENDED, recorded=143, answered=answered
            )

        assert answering_queue.tools == ["dispatch_list", "dispatch_report", "dispatch_claim"]
        report = answering_queue.arguments[1]
        assert report["action"] == "lost"
        assert "status" not in report
        assert "exitCode" not in report
        detail = narrow_json_to_str(report["detail"])
        assert detail.startswith("FLEET-CHECK ")
        assert " exit=143 " in detail
        assert detail.endswith(f"{VERDICT_END} {MEMORY_ENDED}")
        # The run itself ended: this machine's row for it is finished.
        assert "failed" in _ledger(sourced_config)
        assert any(
            record.getMessage().endswith(f": given back, the queue answered queued: {detail}")
            for record in caplog.records
        )

    def test_a_job_given_back_before_is_failed_by_the_queue(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        answered = dump_json_str({"job": queue_job(status="failed")})
        with caplog.at_level("INFO"):
            answering_queue = _settle_killed(
                sourced_config, ending=MEMORY_ENDED, recorded=143, answered=answered
            )

        assert answering_queue.arguments[1]["action"] == "lost"
        assert any(
            ": given back, the queue answered failed: " in record.getMessage()
            for record in caplog.records
        )
