"""A serve claims only while ``refs/fleet/rolled`` names the commit it runs.

MCPs board task 1fd01f09: job 0fa14fab, queued after ``make fleet-roll``
returned, was claimed 16 s later by pendragon's serve on the registry before
the roll and failed on it. These drive lavender's real serve
(:func:`fleet.cli.node_agent.main`) from a registry laid out as a rolled
extraction, its git calls and the queue answered from scripts: a ref that
moved before the claim means the queue is asked for nothing, a ref that moved
by the claim's answer means the job is given back with nothing staged, and
either way the serve claims nothing more and hands over naming the move.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import dump_json_str, narrow_json_to_str

from fleet.cli import node_agent
from fleet.cli import run as run_cli
from fleet.cli.node_roll import MOVED_CODE, RollGuard
from fleet.core import _test_hooks, records, rolled, run_registry, scratch, staging
from fleet.core.rolled import REGISTRY_DIRECTORY
from tests._node_agent_fixtures import (
    PROBED,
    _credentials_in_env,
    _sourced_config,
    claim_replies,
    handed_a_job,
    node_argv,
    prebuilt_export,
    sourced_document,
)
from tests._queue_fakes import DEFAULT_JOB_ID, FakeQueue, queue_job, tick_body
from tests._registry import write_registry
from tests._roll_reads import NEXT_COMMIT, ROLLED_COMMIT, roll_read
from tests.conftest import FakeRun, workspace_document

__all__ = ["_credentials_in_env", "_sourced_config"]

#: What a serve that runs ROLLED_COMMIT says once the ref names NEXT_COMMIT.
MOVED = f"refs/fleet/rolled moved from {ROLLED_COMMIT} to {NEXT_COMMIT}"


class TestTheGuard:
    def test_reads_the_ref_each_time_while_it_names_the_serves_commit(self) -> None:
        reads: list[str] = []

        def ref() -> str:
            reads.append(ROLLED_COMMIT)
            return ROLLED_COMMIT

        guard = RollGuard(runs=ROLLED_COMMIT, rolled=ref)

        assert (guard.moved(), guard.moved(), guard.latched()) == (None, None, None)
        assert len(reads) == 2

    def test_latches_the_first_move_and_reads_nothing_after(self) -> None:
        answers = [NEXT_COMMIT]
        guard = RollGuard(runs=ROLLED_COMMIT, rolled=lambda: answers.pop(0))

        assert guard.moved() == MOVED
        assert guard.moved() == MOVED
        assert guard.latched() == MOVED
        assert guard.runs == ROLLED_COMMIT

    def test_a_ref_that_no_longer_resolves_stops_the_claiming_too(self) -> None:
        refusal = "FLEET_ROLL_FETCH_FAILED: git fetch fleet refused"
        guard = RollGuard(runs=ROLLED_COMMIT, rolled=lambda: refusal)

        assert guard.moved() == f"refs/fleet/rolled moved from {ROLLED_COMMIT} to {refusal}"


class TestTheServesCommit:
    def test_is_the_extraction_its_registry_was_read_from(self, config_path: pathlib.Path) -> None:
        assert rolled.extraction_commit(config_path) == ROLLED_COMMIT
        assert run_cli.registry_commit(config_path) == ROLLED_COMMIT

    def test_a_checkout_s_registry_is_no_commit_s(self, tmp_path: pathlib.Path) -> None:
        checkout = tmp_path / "API" / pathlib.PurePosixPath(REGISTRY_DIRECTORY)

        assert rolled.extraction_commit(checkout) is None
        assert rolled.extraction_commit(tmp_path / "registry") is None
        assert run_cli.registry_commit(checkout) == run_registry.WORKING_TREE

    def test_a_directory_named_like_a_commit_outside_the_extractions_is_not_one(
        self, config_path: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        elsewhere = tmp_path / ROLLED_COMMIT / pathlib.PurePosixPath(REGISTRY_DIRECTORY)
        misnamed = (
            scratch.directory(scratch.EXTRACTIONS_DIR)
            / "turn.lock"
            / pathlib.PurePosixPath(REGISTRY_DIRECTORY)
        )

        assert rolled.extraction_commit(elsewhere) is None
        assert rolled.extraction_commit(misnamed) is None

    def test_a_serve_from_a_checkout_is_refused_before_it_asks_the_queue(
        self, tmp_path: pathlib.Path, config_path: pathlib.Path
    ) -> None:
        checkout = write_registry(tmp_path / "registry", workspace_document())
        endpoint = FakeQueue([])
        _test_hooks.http_post = endpoint

        with pytest.raises(ValueError, match="is not a rolled extraction's registry"):
            node_agent.main(node_argv(checkout))

        assert endpoint.tools == []


class TestARollBeforeTheClaim:
    def test_asks_the_queue_for_nothing_and_records_why(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        runner = FakeRun([*PROBED, *roll_read(sourced_config, NEXT_COMMIT)])
        _test_hooks.run = runner
        endpoint = FakeQueue([dump_json_str({"jobs": []})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools == ["dispatch_list"]
        verdict = (
            f"asked the queue for nothing: {MOVED}; the next serve claims on the rolled commit"
        )
        tick = tick_body(endpoint.ticks[0])
        assert (tick["claiming"], tick["verdict"]) == (False, verdict)
        assert f"lavender {verdict}" in caplog.text


class TestARollBetweenTheAskAndTheAnswer:
    """The window job 0fa14fab was claimed in: the read before the claim
    found the serve's commit, the queue then handed over a job queued after
    the roll, and the read after the claim finds the move."""

    def _given_back(self, sourced_config: pathlib.Path) -> tuple[FakeQueue, FakeRun]:
        runner = FakeRun(
            [*PROBED, *roll_read(sourced_config), *roll_read(sourced_config, NEXT_COMMIT)]
        )
        _test_hooks.run = runner
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": queue_job(status="claimed")}),
                dump_json_str({"job": queue_job(status="queued")}),
            ]
        )
        _test_hooks.http_post = endpoint
        assert node_agent.main(node_argv(sourced_config)) == 0
        return endpoint, runner

    def test_gives_the_job_back_with_nothing_staged(self, sourced_config: pathlib.Path) -> None:
        endpoint, runner = self._given_back(sourced_config)

        assert endpoint.tools == ["dispatch_list", "dispatch_claim", "dispatch_report"]
        given = endpoint.arguments[2]
        assert (given["action"], given["jobId"]) == ("lost", DEFAULT_JOB_ID)
        assert narrow_json_to_str(given["detail"]) == (
            f"{MOVED_CODE}: lavender's serve was handed {DEFAULT_JOB_ID} after {MOVED}; "
            "nothing was staged, so the job is given back for a serve on the rolled commit"
        )
        # The probes and the two reads, and no mirror, export or lease.
        assert len(runner.calls) == len(handed_a_job(sourced_config))
        assert records.read_ledger(sourced_config.parent / "runs" / "ledger.jsonl") == ()
        assert not (sourced_config.parent / "runs" / "leases.json").exists()

    def test_the_serve_claims_nothing_more_and_hands_over_naming_the_move(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        document = sourced_document(())
        document["node_serve_seconds"] = 1200
        write_registry(sourced_config, document)

        with caplog.at_level("INFO"):
            endpoint, _ = self._given_back(sourced_config)

        # No listing between the boundaries and no claim after the give-back.
        assert endpoint.tools == ["dispatch_list", "dispatch_claim", "dispatch_report"]
        handed_over = caplog.records[-1].getMessage()
        assert handed_over.endswith(f"before the 2025-09-04T15:36:00+00:00 fire: {MOVED}")


class TestALaunchRecordsTheServesCommit:
    def test_the_launch_script_writes_it_beside_the_run(self, sourced_config: pathlib.Path) -> None:
        payload = prebuilt_export(sourced_config)
        run = FakeRun(
            claim_replies(staging.digest(payload), registry=sourced_config, commit_present=True)
        )
        _test_hooks.run = run
        _test_hooks.http_post = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": queue_job(status="claimed")}),
                dump_json_str({"job": queue_job(status="running", node="lavender")}),
            ]
        )

        assert node_agent.main(node_argv(sourced_config)) == 0

        recorded = [
            payload.decode("utf-8")
            for payload in run.stdin
            if payload is not None and run_registry.REGISTRY_COMMIT_NAME.encode() in payload
        ]
        assert len(recorded) == 1
        assert f"printf '%s\\n' '{ROLLED_COMMIT}' > '" in recorded[0]
