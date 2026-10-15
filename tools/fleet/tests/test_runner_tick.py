"""The runner tick's wire shape and the queue call that records it (MCPs 939ec5c7, A4)."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import (
    InvalidJsonError,
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
)

from fleet.contracts.action_cache import ActionCacheCounts
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
    load=TickLoad(runs=1, workers=8, free_ram_gb=9.5, free_disk_gb=412.5, action_cache=None),
    claiming=True,
    verdict="sedona asked for 2 fitting project(s); nothing in the node lane matched",
)

#: lavender-wsl's tick on 2026-10-15, its eight runners reading the cache.
LAVENDER = RunnerTick(
    node="lavender-wsl",
    elevated=False,
    tags=(NodeTag.LINUX, NodeTag.DOCKER),
    fits=("tools/fleet-execution-linux",),
    load=TickLoad(
        runs=2,
        workers=4,
        free_ram_gb=11.0,
        free_disk_gb=150.0,
        action_cache=ActionCacheCounts(runners=8, reading=8, served=8, missed=0),
    ),
    claiming=True,
    verdict="lavender-wsl asked for 1 fitting project(s)",
)

#: Counts that cannot describe one set of runners: more reading than counted.
OVERCOUNTED: JSONObject = {"runners": 8, "reading": 9, "served": 0, "missed": 0}


def _sedona_load(action_cache: JSONValue) -> JSONObject:
    """sedona's encoded load carrying the given ``actionCache`` value.

    Args:
        action_cache: What the load's ``actionCache`` holds.

    Returns:
        The load as the wire carries it.
    """
    return {
        "runs": 1,
        "workers": 8,
        "freeRamGb": 9.5,
        "freeDiskGb": 412.5,
        "actionCache": action_cache,
    }


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
            "load": {
                "runs": 1,
                "workers": 8,
                "freeRamGb": 9.5,
                "freeDiskGb": 412.5,
                "actionCache": None,
            },
        }

    def test_a_runner_host_s_action_cache_counts_ride_in_its_load(self) -> None:
        assert encode_runner_tick(LAVENDER)["load"] == {
            "runs": 2,
            "workers": 4,
            "freeRamGb": 11.0,
            "freeDiskGb": 150.0,
            "actionCache": {"runners": 8, "reading": 8, "served": 8, "missed": 0},
        }

    def test_a_silent_node_omits_its_load(self) -> None:
        assert "load" not in encode_runner_tick(SILENT)

    @pytest.mark.parametrize(
        "tick", [SEDONA, LAVENDER, SILENT], ids=["with load", "with cache counts", "without load"]
    )
    def test_it_round_trips(self, tick: RunnerTick) -> None:
        assert decode_runner_tick(encode_runner_tick(tick)) == tick

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ([], "runner tick must be a JSON object, got list"),
            ({**encode_runner_tick(SEDONA), "load": 3}, "load must be a JSON object, got int"),
            ({**encode_runner_tick(SEDONA), "tags": ["windows", "windows"]}, "repeats 'windows'"),
            ({**encode_runner_tick(SEDONA), "tags": ["quantum"]}, "tags[0] must be one of"),
            (
                {
                    **encode_runner_tick(SEDONA),
                    "load": {"runs": 1, "workers": 8, "freeRamGb": 9.5, "freeDiskGb": 412.5},
                },
                "load.actionCache is missing",
            ),
            (
                {**encode_runner_tick(SEDONA), "load": _sedona_load(8)},
                "actionCache must be a JSON object or null, got int",
            ),
            (
                {**encode_runner_tick(SEDONA), "load": _sedona_load(OVERCOUNTED)},
                "do not describe one set of runners: 8 runner.s., 9 reading, 0 served, 0 missed",
            ),
        ],
        ids=[
            "not an object",
            "load not an object",
            "a repeated tag",
            "a tag outside the set",
            "no action cache key",
            "action cache not an object",
            "action cache over-counted",
        ],
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
