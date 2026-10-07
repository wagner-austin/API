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
from tests.conftest import DEMO_RUN_ID, FakeRun, ok, retire_replies

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
        # Neither the retire nor the sweep ran, and the row is as launched.
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
            f"log=lavender:C:/fleet/stage/logs/{DEMO_RUN_ID}.log run={DEMO_RUN_ID}"
        )
        retire_path = f"C:/fleet/stage/retire-{DEMO_RUN_ID}.ps1"
        assert [retire_path in " ".join(call) for call in settling_node.calls[:6]] == [
            False,
            False,
            False,
            False,
            True,
            True,
        ]
        assert "passed" in _ledger(sourced_config)


#: The tail of a build the unit-end script closed (MCPs board task
#: c8585623): 859e7ab3's transcript as it stopped, then the line systemd's
#: ``ExecStopPost=`` appends for an oom-kill.
KILLED_TAIL = (
    "fleet-prepare: no stored install for tree key e7fd2baa9e87; running npm ci\n"
    "npm warn deprecated node-domexception@1.0.0: Use your platform's native DOMException\n"
    "FLEET_UNIT_ENDED: unit fleet-MCPs-mcp-shared-diphtheria-1791171734 ended with systemd "
    "result oom-kill (killed KILL) before the build wrote its status; recorded exit 137\n"
)


class TestABuildItsUnitClosed:
    def test_closes_failed_with_the_unit_s_ending_on_the_line(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        _test_hooks.run = FakeRun(
            [ok(""), ok("137 1757000060"), ok(""), ok(KILLED_TAIL), *retire_replies(), *PROBED]
        )
        answering_queue = FakeQueue(
            [
                held_answer(taskId=VERDICT_TASK),
                dump_json_str({"job": queue_job(status="failed", exitCode=137)}),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = answering(answering_queue)

        assert node_agent.main(node_argv(sourced_config)) == 0

        close = answering_queue.arguments[1]
        assert close["action"] == "close"
        assert close["status"] == "failed"
        assert close["exitCode"] == 137
        assert narrow_json_to_str(close["detail"]).endswith(
            f"log=lavender:C:/fleet/stage/logs/{DEMO_RUN_ID}.log run={DEMO_RUN_ID} "
            "FLEET_UNIT_ENDED: unit fleet-MCPs-mcp-shared-diphtheria-1791171734 ended with "
            "systemd result oom-kill (killed KILL) before the build wrote its status; "
            "recorded exit 137"
        )
        assert "failed" in _ledger(sourced_config)
