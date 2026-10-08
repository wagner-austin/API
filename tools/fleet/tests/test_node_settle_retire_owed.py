"""A settle's retire its node did not answer is owed, not raised (MCPs board task 8776b828).

Until this, a settle whose retire met an ssh that timed out raised
``NODE_UNREACHABLE`` after the queue job and the ledger row had both closed,
so no later pass retried it and the raise ended the serve. These cases run
the real node runner (:func:`fleet.cli.node_agent.main`) three times on one
launched run: the first start settles it while the node misses the retire
and goes on to its claim pass, the second start finds the node answering
and retires the run, and the third sends nothing, since the run's directory
is retired once.
"""

from __future__ import annotations

import pathlib

from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_str

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
from tests._queue_fakes import FakeQueue, queue_job
from tests.conftest import DEMO_RUN_ID, FakeRun, failed, ok, retire_replies

__all__ = ["_credentials_in_env", "_sourced_config"]

#: What ssh said when lavender-wsl missed its reads on 2026-10-07.
BANNER_TIMEOUT = "Connection timed out during banner exchange"

#: Where the runner's retire script goes on the demo node.
RETIRE_PATH = f"C:/fleet/stage/retire-{DEMO_RUN_ID}.ps1"


def _retires(config_path: pathlib.Path) -> list[tuple[str, str, str]]:
    """The runner's retire record, as (run, node, state) per line.

    Args:
        config_path: The workspace document beside it.

    Returns:
        One tuple per line, in order.
    """
    text = (config_path.parent / "runs" / "retires.jsonl").read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        value = load_json_str(line)
        assert isinstance(value, dict)
        lines.append(
            (
                narrow_json_to_str(value["run_id"]),
                narrow_json_to_str(value["node"]),
                narrow_json_to_str(value["state"]),
            )
        )
    return lines


def _sent_retire(run: FakeRun) -> int:
    """How many calls carried the run's retire script.

    Args:
        run: The node's fake.

    Returns:
        The count, two per retire sent (the script written, then run).
    """
    return sum(RETIRE_PATH in " ".join(call) for call in run.calls)


def _idle_queue() -> FakeQueue:
    """The queue's answers to a start holding nothing: no job held, nothing to claim.

    Returns:
        The fake.
    """
    return FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])


class TestARetireTheNodeDidNotAnswer:
    def test_leaves_the_serve_running_and_a_later_pass_retires_the_run_once(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        missed = FakeRun(
            [
                ok(""),  # result: send the script
                ok("0 1757000060"),  # result: run it
                ok(""),  # tail: send the script
                ok(PASSING_TAIL),  # tail: run it
                failed(255, BANNER_TIMEOUT),  # retire: send the script, missed
                *PROBED,
            ]
        )
        _test_hooks.run = missed
        settling = FakeQueue(
            [
                held_answer(taskId=VERDICT_TASK),
                dump_json_str({"job": queue_job(status="passed")}),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = answering(settling)

        assert node_agent.main(node_argv(sourced_config)) == 0

        # The run closed on both sides, the claim pass still ran after it,
        # and no sweep was sent to a node that had just missed the retire.
        assert settling.tools == ["dispatch_list", "dispatch_report", "dispatch_claim"]
        assert settling.arguments[1]["status"] == "passed"
        assert _sent_retire(missed) == 1
        assert not any("fleet-venv-sweep" in " ".join(call) for call in missed.calls)
        assert _retires(sourced_config) == [(DEMO_RUN_ID, "lavender", "owed")]

        answered = FakeRun([*retire_replies(), *PROBED])
        _test_hooks.run = answered
        later = _idle_queue()
        _test_hooks.http_post = answering(later)

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert later.tools == ["dispatch_list", "dispatch_claim"]
        assert _sent_retire(answered) == 2
        assert any("fleet-venv-sweep" in " ".join(call) for call in answered.calls)
        assert _retires(sourced_config) == [
            (DEMO_RUN_ID, "lavender", "owed"),
            (DEMO_RUN_ID, "lavender", "retired"),
        ]

        after = FakeRun(list(PROBED))
        _test_hooks.run = after
        _test_hooks.http_post = answering(_idle_queue())

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert _sent_retire(after) == 0
        assert len(after.calls) == len(PROBED)
        assert len(_retires(sourced_config)) == 2
