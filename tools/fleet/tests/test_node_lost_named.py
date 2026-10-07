"""The job that names a run is the one that launched it (MCPs board task c1d48330).

On 2026-10-07 lavender-wsl's collect pass listed job 641605e5 as claimed at
03:47:50Z, its launch reported the start at 03:47:53Z, and the pass then
took the started run for a lost one, found two of the same session's jobs
that could have launched it, raised DISPATCH_CLAIM_AMBIGUOUS and ended the
serve. Each case here asks :func:`fleet.cli.node_lost.launching_job` about
the demo run lavender launched at ``DEMO_NOW``, against the fake queue.
"""

from __future__ import annotations

import pathlib

from platform_core.json_utils import JSONObject

from fleet.cli import _config, node_lost
from fleet.contracts.dispatch import DispatchJob, decode_job
from fleet.contracts.ledger import LedgerEntry
from fleet.core import _test_hooks, queue, records
from tests._node_agent_fixtures import _credentials_in_env, _sourced_config, launch
from tests._queue_fakes import FakeQueue, listing_page, queue_job, trail_answer
from tests.conftest import DEMO_NOW, DEMO_RUN_ID, FakeRun

__all__ = ["_credentials_in_env", "_sourced_config"]

#: This node's runner, the one that launched the demo run.
LAVENDER = "fleet-node-lavender"

#: A job of the same session and project that lavender also claimed in the
#: window the run began in, and that has since left it.
TWIN: JSONObject = queue_job(
    id="6ee9913e-adae-48bb-b601-3d883ab41a04",
    status="passed",
    node="diphtheria",
    runId="libs-demo-diphtheria-1757003600",
    claimedBy="fleet-node-diphtheria",
)


def _launched_row(config_path: pathlib.Path) -> LedgerEntry:
    """Launch the demo run and read its live row.

    Args:
        config_path: The workspace document.

    Returns:
        The row.
    """
    launch(config_path)
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    return records.read_ledger(loaded.ledger)[-1]


def _ask(row: LedgerEntry, answers: list[str]) -> tuple[DispatchJob | None, FakeQueue]:
    """Ask which job launched the row, against scripted queue answers.

    Args:
        row: The run's live row.
        answers: The submitter's pages and the trails read, in order.

    Returns:
        The job found, and the queue for its calls.
    """
    _test_hooks.run = FakeRun([])
    endpoint = FakeQueue(answers)
    _test_hooks.http_post = endpoint
    found = node_lost.launching_job(queue.load_credentials(), row, agent=LAVENDER)
    return found, endpoint


class TestAJobThatNamesTheRun:
    def test_is_its_launcher_though_another_of_its_session_also_has_a_claim_in_the_window(
        self, sourced_config: pathlib.Path
    ) -> None:
        """The measured race: the start landed between the listing and this ask."""
        row = _launched_row(sourced_config)
        started = queue_job(
            status="running", node="lavender", runId=DEMO_RUN_ID, claimedBy=LAVENDER
        )

        found, endpoint = _ask(
            row,
            [listing_page([TWIN, started], None), trail_answer(TWIN, [(LAVENDER, DEMO_NOW)])],
        )

        launcher = decode_job(started, answer="")
        assert found == launcher
        assert endpoint.tools == ["dispatch_list", "dispatch_get"]
        assert node_lost.still_held(launcher, agent=LAVENDER) is True


class TestAJobAnotherRunnerHoldsThatNamesTheRun:
    def test_leaves_the_run_to_that_runner_and_reads_no_trail(
        self, sourced_config: pathlib.Path
    ) -> None:
        """serendipity's elevated runner at 04:27:09Z on 2026-10-07 stopped
        its sibling runner's live run of job 2042e8ef as lost; the run is
        the holder's, never this runner's to stop."""
        row = _launched_row(sourced_config)
        sibling = queue_job(
            status="running",
            node="lavender",
            runId=DEMO_RUN_ID,
            claimedBy="fleet-node-lavender-elevated",
        )

        found, endpoint = _ask(row, [listing_page([sibling, TWIN], None)])

        assert found is None
        assert endpoint.tools == ["dispatch_list"]


class TestAJobThisRunnerHoldsForAnotherRun:
    def test_is_no_candidate_and_its_trail_is_not_read(self, sourced_config: pathlib.Path) -> None:
        row = _launched_row(sourced_config)
        other = queue_job(
            id="77777777-7777-4777-8777-777777777777",
            status="running",
            node="lavender",
            runId="libs-demo-lavender-1757000900",
            claimedBy=LAVENDER,
        )

        found, endpoint = _ask(
            row,
            [listing_page([other, TWIN], None), trail_answer(TWIN, [(LAVENDER, DEMO_NOW)])],
        )

        assert found == decode_job(TWIN, answer="")
        assert endpoint.tools == ["dispatch_list", "dispatch_get"]


class TestTwoCandidatesOneStillUnstarted:
    def test_answers_the_unstarted_one_so_the_run_is_not_shown_lost(
        self, sourced_config: pathlib.Path
    ) -> None:
        """The start has not landed yet: the run may be the held claim's, so
        it is left alone, and the next pass finds the job naming it."""
        row = _launched_row(sourced_config)
        unstarted = queue_job(status="claimed", claimedBy=LAVENDER)

        found, endpoint = _ask(
            row,
            [
                listing_page([TWIN, unstarted], None),
                trail_answer(TWIN, [(LAVENDER, DEMO_NOW)]),
                trail_answer(unstarted, [(LAVENDER, DEMO_NOW)]),
            ],
        )

        held = decode_job(unstarted, answer="")
        assert found == held
        assert endpoint.tools == ["dispatch_list", "dispatch_get", "dispatch_get"]
        assert node_lost.still_held(held, agent=LAVENDER) is True
