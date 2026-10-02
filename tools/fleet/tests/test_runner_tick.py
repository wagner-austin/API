"""The runner tick's wire shape and the queue call that records it (MCPs 939ec5c7, A4)."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import InvalidJsonError, JSONTypeError, JSONValue, dump_json_str

from fleet.contracts.runner_tick import (
    RunnerTick,
    TickLoad,
    decode_recorded_at,
    decode_runner_tick,
    encode_runner_tick,
)
from fleet.contracts.tags import NodeTag
from fleet.core import _test_hooks, tick_report
from tests._queue_fakes import (
    QUEUE_CREDENTIALS,
    RUNNER_IDENTITY,
    TICKED_AT,
    FakeQueue,
    FakeRefusingQueue,
)

#: sedona's tick when its runner could claim two projects.
SEDONA = RunnerTick(
    node="sedona",
    elevated=False,
    tags=(NodeTag.CXX, NodeTag.GPU, NodeTag.WINDOWS),
    fits=("libs/platform_core", "tools/fleet"),
    load=TickLoad(runs=1, workers=8, free_ram_gb=9.5),
    claiming=True,
    verdict="sedona asked for 2 fitting project(s); nothing in the node lane matched",
)

#: A node that did not answer its probe.
SILENT = RunnerTick(
    node="sedona",
    elevated=True,
    tags=(),
    fits=(),
    load=None,
    claiming=False,
    verdict="did not answer; claiming nothing: ssh timed out",
)


class TestTheWireShape:
    def test_a_tick_encodes_to_dispatch_tick_s_arguments(self) -> None:
        assert encode_runner_tick(SEDONA) == {
            "node": "sedona",
            "elevated": False,
            "tags": ["cxx", "gpu", "windows"],
            "fits": ["libs/platform_core", "tools/fleet"],
            "claiming": True,
            "verdict": SEDONA["verdict"],
            "load": {"runs": 1, "workers": 8, "freeRamGb": 9.5},
        }

    def test_a_silent_node_omits_its_load(self) -> None:
        assert "load" not in encode_runner_tick(SILENT)

    @pytest.mark.parametrize("tick", [SEDONA, SILENT], ids=["with load", "without load"])
    def test_it_round_trips(self, tick: RunnerTick) -> None:
        assert decode_runner_tick(encode_runner_tick(tick)) == tick

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ([], "runner tick must be a JSON object, got list"),
            ({**encode_runner_tick(SEDONA), "load": 3}, "load must be a JSON object, got int"),
            ({**encode_runner_tick(SEDONA), "tags": ["windows", "windows"]}, "repeats 'windows'"),
            ({**encode_runner_tick(SEDONA), "tags": ["quantum"]}, "tags[0] must be one of"),
        ],
        ids=["not an object", "load not an object", "a repeated tag", "a tag outside the set"],
    )
    def test_a_malformed_tick_is_refused_by_name(self, value: JSONValue, message: str) -> None:
        with pytest.raises(JSONTypeError, match=message.replace("[", r"\[").replace("]", r"\]")):
            decode_runner_tick(value)


class TestTheRecordedAnswer:
    def test_it_reads_when_the_queue_stored_the_tick(self) -> None:
        answer = dump_json_str({"tick": {"runner": "fleet-node-sedona", "tickedAt": TICKED_AT}})

        assert decode_recorded_at(answer) == TICKED_AT

    def test_an_answer_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="dispatch_tick must answer an object, got list"):
            decode_recorded_at("[]")

    def test_an_answer_that_is_not_json_is_refused(self) -> None:
        with pytest.raises(InvalidJsonError):
            decode_recorded_at("stored")


class TestRecordingATick:
    def test_it_sends_the_tick_under_the_runner_s_identity(self) -> None:
        endpoint = FakeQueue([])
        _test_hooks.http_post = endpoint

        assert tick_report.record_tick(QUEUE_CREDENTIALS, SEDONA, identity=RUNNER_IDENTITY) == (
            TICKED_AT
        )
        assert endpoint.ticks == [{**encode_runner_tick(SEDONA), **RUNNER_IDENTITY}]

    def test_a_refusal_propagates(self) -> None:
        _test_hooks.http_post = FakeRefusingQueue("verdict must be at most 2000 characters")

        with pytest.raises(AppError, match="verdict must be at most 2000 characters"):
            tick_report.record_tick(QUEUE_CREDENTIALS, SEDONA, identity=RUNNER_IDENTITY)
