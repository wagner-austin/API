"""A claimed job whose staging cannot finish is refused, not left claimed.

Board task 1e57ebe5, A3. ``prepare`` has always been guarded, so every
fault BEFORE the lease is reported to the queue by name. ``dispatch.start``
was not, and there is no ``failed`` feed row anywhere on that path, so a
fault DURING staging ended the tick with nothing recorded: the queue row
stayed in status claimed until its 3600-second lease ran out, and a
session waiting on the verdict could not tell a stalled run from a slow
one. Measured 2026-09-24, two jobs sat exactly that way for 32 and 14
minutes.

The case drives the real tick through the real staging code and fails the
one command that carries the payload, which is where the 2026-09-24
incident actually died.
"""

from __future__ import annotations

import pathlib

from platform_core.errors import FleetErrorCode
from platform_core.json_utils import dump_json_str, narrow_json_to_str

from fleet.cli import node_agent
from fleet.core import _test_hooks
from tests._node_agent_fixtures import (
    PROBED,
    _credentials_in_env,
    _sourced_config,
    node_argv,
    prebuilt_export,
    sourced_document,
)
from tests._queue_fakes import FakeQueue, queue_job
from tests.conftest import FakeRun, ok, timed_out

__all__ = ["_credentials_in_env", "_sourced_config", "sourced_document"]

#: The deadline the transport gives one ssh, quoted so the refusal's text
#: can be asserted against the number the code actually carries.
SSH_DEADLINE_SECONDS = 120

#: Every command the tick runs up to and including the send that carries the
#: payload, which is the one that fails here. The steps are the same ones
#: :func:`tests._node_agent_fixtures.claim_replies` lists; this stops early
#: rather than repeating the rest, because nothing after it should run.
UP_TO_THE_PAYLOAD: tuple[_test_hooks.CommandResult, ...] = (
    *PROBED,  # the probe, before any claim
    ok(""),  # git init --bare (the mirror is new)
    ok(""),  # git cat-file -e: the sha is present
    ok(""),  # git archive -o
    ok(""),  # stage: send mkdir script
    ok(""),  # stage: run mkdir
    timed_out(SSH_DEADLINE_SECONDS),  # stage: send the payload, ended at its deadline
)


class TestAStagingFaultIsReported:
    def test_a_payload_send_that_ends_at_its_deadline_is_refused_by_name(
        self, sourced_config: pathlib.Path
    ) -> None:
        """The queue is told, rather than the row being left claimed.

        Before A3 this tick ended with the AppError escaping and the job
        sitting in status claimed for the rest of its lease, which is the
        shape a waiting session cannot distinguish from a slow run.
        """
        # The archive the faked ``git archive`` would have written, where the
        # tick reads it: the payload has to be real for staging to carry it.
        prebuilt_export(sourced_config)
        _test_hooks.run = FakeRun(list(UP_TO_THE_PAYLOAD))
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": queue_job(status="claimed")}),
                dump_json_str({"job": queue_job(status="refused")}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        closed = endpoint.arguments[2]
        assert closed["action"] == "close"
        assert closed["status"] == "refused"
        assert "exitCode" not in closed
        detail = narrow_json_to_str(closed["detail"])
        assert FleetErrorCode.NODE_UNREACHABLE in detail
        # The reason reaches the queue, not just the fact: a refusal that
        # said only "it failed" would send the reader to the node.
        assert f"timed out after {SSH_DEADLINE_SECONDS} s" in detail
