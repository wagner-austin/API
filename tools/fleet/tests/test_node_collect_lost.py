"""A run whose queue job left its runner is stopped and closed lost (board task fd402617).

The defect, measured 2026-09-30: MCPs/search's job baca2609 was claimed by
lavender-wsl, its claim lapsed during that node's memory stall, and diphtheria
reclaimed it and passed it. lavender-wsl's ledger row stayed ``running`` for
seven hours, and every tick refused all work against it with
NODE_OWNER_RESERVED, because the job no longer named lavender-wsl's runner
and was never cancelled, so neither half of the collect pass looked at it.

Each test drives one whole tick through ``node_agent.main`` against the fake
queue and the fake ssh runner, and asserts what reached the node, the queue
and the ledger. The demo run is launched at ``DEMO_NOW`` by lavender's runner.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONObject, dump_json_str

from fleet.cli import _config, node_agent
from fleet.contracts.ledger import LedgerEntry
from fleet.core import _test_hooks, claim_window, dialect, leases, names, records
from tests._node_agent_fixtures import (
    PROBED,
    _credentials_in_env,
    _sourced_config,
    launch,
    node_argv,
)
from tests._queue_fakes import (
    DEFAULT_JOB_ID,
    FakeQueue,
    listing_page,
    queue_job,
    trail_answer,
)
from tests.conftest import DEMO_NOW, DEMO_RUN_ID, FakeRun, ok, retire_replies

__all__ = ["_credentials_in_env", "_sourced_config"]

#: This node's runner, the one that launched the demo run.
LAVENDER = "fleet-node-lavender"

#: The runner that took the job over once lavender's claim lapsed.
DIPHTHERIA = "fleet-node-diphtheria"

#: When diphtheria reclaimed it: the moment lavender's hour-long claim ran out.
RECLAIMED = DEMO_NOW + claim_window.CLAIM_LEASE_SECONDS

#: The job as the queue holds it after the takeover: passed on diphtheria,
#: every field that named lavender overwritten.
TAKEN_OVER: JSONObject = queue_job(
    status="passed",
    node="diphtheria",
    runId="libs-demo-diphtheria-1757003600",
    claimedBy=DIPHTHERIA,
)

#: Its trail: lavender's claim, then diphtheria's.
TAKEN_OVER_TRAIL = trail_answer(TAKEN_OVER, [(LAVENDER, DEMO_NOW), (DIPHTHERIA, RECLAIMED)])


def _stop_body(config_path: pathlib.Path) -> str:
    """The stop script lavender is sent for the demo run.

    Args:
        config_path: The workspace document.

    Returns:
        The script's text, as the dialect renders it.
    """
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    node = loaded.workspace["nodes"]["lavender"]
    return dialect.for_platform(node["platform"]).stop_script(
        target=names.dispatch_directory(node["stage_root"], DEMO_RUN_ID), run_id=DEMO_RUN_ID
    )


def _last_row(config_path: pathlib.Path) -> LedgerEntry:
    """The newest ledger row.

    Args:
        config_path: The workspace document.

    Returns:
        The row.
    """
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    return records.read_ledger(loaded.ledger)[-1]


def _lease_released(config_path: pathlib.Path) -> bool:
    """Whether the demo run holds no lease at ``DEMO_NOW``, when its own is unexpired.

    Args:
        config_path: The workspace document.

    Returns:
        True when no lease names the run.
    """
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    return leases.find_by_run(loaded.leases, run_id=DEMO_RUN_ID, now_unix=DEMO_NOW) is None


def _tick(
    config_path: pathlib.Path, answers: list[str], *, stops: bool
) -> tuple[FakeQueue, FakeRun]:
    """Run one tick in which lavender holds nothing and nothing was cancelled.

    Args:
        config_path: The workspace document.
        answers: What the queue answers after the held and cancelled listings,
            before the claim.
        stops: Whether the node is expected to be sent a stop.

    Returns:
        The queue and the ssh runner, for assertions.
    """
    replies = [ok(""), ok("stopped"), *retire_replies(), *PROBED] if stops else list(PROBED)
    runner = FakeRun(replies)
    _test_hooks.run = runner
    endpoint = FakeQueue(
        [
            dump_json_str({"jobs": []}),
            listing_page([], None),
            *answers,
            dump_json_str({"claimed": None}),
        ]
    )
    _test_hooks.http_post = endpoint
    assert node_agent.main(node_argv(config_path)) == 0
    return endpoint, runner


class TestTakenOver:
    def test_the_measured_sequence_stops_the_run_closes_it_lost_and_frees_the_slot(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)

        endpoint, runner = _tick(
            sourced_config, [listing_page([TAKEN_OVER], None), TAKEN_OVER_TRAIL], stops=True
        )

        assert endpoint.tools == [
            "dispatch_list",
            "dispatch_list",
            "dispatch_list",
            "dispatch_get",
            "dispatch_claim",
        ]
        assert endpoint.arguments[3] == {"jobId": DEFAULT_JOB_ID}
        assert runner.stdin[0] == _stop_body(sourced_config).encode("utf-8")
        last = _last_row(sourced_config)
        assert last["run_id"] == DEMO_RUN_ID
        assert last["outcome"] == "lost"
        assert last["detail"] == (
            f"queue job {DEFAULT_JOB_ID} left {LAVENDER} while this run was live: it is "
            f"passed and held by {DIPHTHERIA}; stopped by {LAVENDER}; was dispatched by "
            "opus-dispatch-0905"
        )
        assert _lease_released(sourced_config)
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
        assert records.live_runs(loaded.ledger, node="lavender") == 0

    def test_a_job_back_in_the_queue_with_no_holder_is_lost_to_this_runner_too(
        self, sourced_config: pathlib.Path
    ) -> None:
        """Between the lapse and a reclaim the job sits queued and unheld; the
        run can no longer report to it, so it is stopped then as well."""
        launch(sourced_config)
        requeued = queue_job(status="queued")

        _tick(
            sourced_config,
            [listing_page([requeued], None), trail_answer(requeued, [(LAVENDER, DEMO_NOW)])],
            stops=True,
        )

        assert _last_row(sourced_config)["detail"] == (
            f"queue job {DEFAULT_JOB_ID} left {LAVENDER} while this run was live: it is "
            f"queued and held by no runner; stopped by {LAVENDER}; was dispatched by "
            "opus-dispatch-0905"
        )

    def test_pages_past_other_sessions_jobs_without_reading_their_trails(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        other_session = queue_job(
            id="99999999-9999-4999-8999-999999999999",
            sessionId="22222222-bbbb-4bbb-8bbb-222222222222",
            status="passed",
        )

        endpoint, _ = _tick(
            sourced_config,
            [
                listing_page([other_session], 100),
                listing_page([TAKEN_OVER], None),
                TAKEN_OVER_TRAIL,
            ],
            stops=True,
        )

        assert [arguments.get("offset") for arguments in endpoint.arguments[2:4]] == [0, 100]
        assert endpoint.tools.count("dispatch_get") == 1
        assert _last_row(sourced_config)["outcome"] == "lost"


class TestLeftAlone:
    def test_a_claim_by_another_runner_alone_does_not_make_the_run_this_runners(
        self, sourced_config: pathlib.Path
    ) -> None:
        """Only a claim THIS runner took ties a job to a run on its node."""
        launch(sourced_config)

        _tick(
            sourced_config,
            [listing_page([TAKEN_OVER], None), trail_answer(TAKEN_OVER, [(DIPHTHERIA, DEMO_NOW)])],
            stops=False,
        )

        assert _last_row(sourced_config)["outcome"] == "running"
        assert not _lease_released(sourced_config)

    def test_two_jobs_that_could_have_launched_it_are_refused_not_guessed_between(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        twin = {**TAKEN_OVER, "id": "88888888-8888-4888-8888-888888888888"}
        _test_hooks.run = FakeRun(list(PROBED))
        _test_hooks.http_post = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                listing_page([], None),
                listing_page([TAKEN_OVER, twin], None),
                TAKEN_OVER_TRAIL,
                trail_answer(twin, [(LAVENDER, DEMO_NOW)]),
            ]
        )

        with pytest.raises(AppError) as raised:
            node_agent.main(node_argv(sourced_config))

        assert raised.value.code is FleetErrorCode.DISPATCH_CLAIM_AMBIGUOUS
        assert raised.value.message == (
            f"{DEMO_RUN_ID} has no job held by {LAVENDER}, and 2 jobs could have launched it: "
            f"{DEFAULT_JOB_ID}, 88888888-8888-4888-8888-888888888888"
        )
        assert _last_row(sourced_config)["outcome"] == "running"
