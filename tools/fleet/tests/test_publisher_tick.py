"""A hub publisher tick's wire shape and the queue call that records it (MCPs 9f1367d5)."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import InvalidJsonError, JSONTypeError, dump_json_str

from fleet.contracts.publisher_tick import (
    PublisherTick,
    decode_ticks_counted,
    encode_publisher_tick,
)
from fleet.core import _test_hooks, tick_report
from tests._queue_fakes import QUEUE_CREDENTIALS, RUNNER_IDENTITY, FakeQueue, FakeRefusingQueue

#: hpc-wake's tick on the hub after the 2026-10-16 09:38Z align.
HPC_WAKE = PublisherTick(
    host="austinpc",
    publisher="hpc-wake",
    repo="API",
    head_sha="e164f17a74852359b8c84370b5c5216cfe89f10e",
    dirty=False,
    exit_code=0,
)


class TestTheWireShape:
    def test_it_names_each_field_as_the_tool_does(self) -> None:
        assert encode_publisher_tick(HPC_WAKE) == {
            "host": "austinpc",
            "publisher": "hpc-wake",
            "repo": "API",
            "headSha": "e164f17a74852359b8c84370b5c5216cfe89f10e",
            "dirty": False,
            "exitCode": 0,
        }

    def test_a_dirty_failing_tick_keeps_both_facts(self) -> None:
        encoded = encode_publisher_tick(
            PublisherTick(
                host="austinpc",
                publisher="ci-wake",
                repo="API",
                head_sha="27df272f01d076ad18c1c5c0152879dfa069e048",
                dirty=True,
                exit_code=-1,
            )
        )

        assert (encoded["dirty"], encoded["exitCode"]) == (True, -1)


class TestTheCountedAnswer:
    def test_it_reads_the_stored_state_s_tick_count(self) -> None:
        answer = dump_json_str({"tick": {"publisher": "hpc-wake", "ticks": 3}})

        assert decode_ticks_counted(answer) == 3

    def test_an_answer_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(
            JSONTypeError, match="dispatch_publisher_tick must answer an object, got list"
        ):
            decode_ticks_counted("[]")

    def test_an_answer_without_a_count_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="ticks"):
            decode_ticks_counted(dump_json_str({"tick": {"publisher": "hpc-wake"}}))

    def test_an_answer_that_is_not_json_is_refused(self) -> None:
        with pytest.raises(InvalidJsonError):
            decode_ticks_counted("stored")


class TestRecordingAPublisherTick:
    def test_it_sends_the_tick_under_the_given_identity(self) -> None:
        endpoint = FakeQueue([dump_json_str({"tick": {"ticks": 1}})])
        _test_hooks.http_post = endpoint

        counted = tick_report.record_publisher_tick(
            QUEUE_CREDENTIALS, HPC_WAKE, identity=RUNNER_IDENTITY
        )

        assert counted == 1
        assert endpoint.tools == ["dispatch_publisher_tick"]
        assert endpoint.arguments == [{**encode_publisher_tick(HPC_WAKE), **RUNNER_IDENTITY}]

    def test_a_refusal_propagates(self) -> None:
        _test_hooks.http_post = FakeRefusingQueue("headSha: Invalid")

        with pytest.raises(AppError, match="headSha: Invalid"):
            tick_report.record_publisher_tick(QUEUE_CREDENTIALS, HPC_WAKE, identity=RUNNER_IDENTITY)
