"""Grouping terminal dispatches into posts, and selecting which to announce.

Pure functions, so every test here builds ledger rows through fleet's own
decoder and asserts on text. The rows are decoded rather than constructed as
dicts so a change to ``LedgerEntry`` fails these tests instead of letting them
assert against a shape production no longer produces.
"""

from __future__ import annotations

from fleet.contracts.ledger import NO_EXIT_CODE, LedgerEntry, decode_ledger_entry
from platform_core.json_utils import JSONObject

from fleet_wake.announce import LINE_CAP, MARKER, announcements, terminal_unannounced


def row(
    run_id: str,
    *,
    outcome: str = "passed",
    project: str = "tools/fleet",
    agent: str = "opus-fleet-0906",
    node: str = "lavender",
    exit_code: int = 0,
    started: int = 1788633781,
    ended: int = 1788633884,
) -> LedgerEntry:
    """Build one ledger row through the production decoder.

    Args:
        run_id: The dispatch's id.
        outcome: How it ended, or ``running``.
        project: Repo-relative project path.
        agent: Board label of the dispatching session.
        node: The node's workspace name.
        exit_code: The recipe's exit status.
        started: When the dispatch began.
        ended: When it reached its outcome.

    Returns:
        The decoded row.
    """
    document: JSONObject = {
        "run_id": run_id,
        "node": node,
        "host": node,
        "project": project,
        "agent": agent,
        "session_id": "acc774c0-3bc3-4cce-9dda-c7a12fb99519",
        "started_unix": started,
        "ended_unix": ended,
        "outcome": outcome,
        "exit_code": exit_code,
        "workers": 12,
        "detail": "",
    }
    return decode_ledger_entry(document)


class TestSelectingWhatToAnnounce:
    def test_a_running_dispatch_is_not_announced(self) -> None:
        """It has not ended, so there is nothing to tell anybody."""
        assert terminal_unannounced((row("a", outcome="running"),), frozenset()) == ()

    def test_an_already_announced_dispatch_is_not_announced_again(self) -> None:
        assert terminal_unannounced((row("a"),), frozenset({"a"})) == ()

    def test_every_failure_outcome_is_announced(self) -> None:
        """THE HALF A BRIDGE IS TEMPTED TO DROP. A notifier that announced
        only passes would be the wedge detector's opposite -- and ``lost`` is
        the one nobody else can report, because a dispatch whose lease expired
        with no result cannot announce its own death."""
        rows = tuple(
            row(outcome, outcome=outcome) for outcome in ("refused", "failed", "cancelled", "lost")
        )

        selected = terminal_unannounced(rows, frozenset())

        assert [entry["run_id"] for entry in selected] == [
            "refused",
            "failed",
            "cancelled",
            "lost",
        ]

    def test_terminality_is_not_re_derived_from_a_local_list(self) -> None:
        """Asserted against fleet's own vocabulary rather than a copy of it:
        every outcome the ledger declares is either live or announced, with
        nothing falling between the two."""
        every_outcome = ("refused", "passed", "failed", "cancelled", "lost", "running")
        rows = tuple(row(outcome, outcome=outcome) for outcome in every_outcome)

        selected = terminal_unannounced(rows, frozenset())

        assert len(selected) == len(every_outcome) - 1
        assert "running" not in {entry["run_id"] for entry in selected}


class TestGrouping:
    def test_one_post_per_agent_and_project(self) -> None:
        """A session fanning a suite across three nodes ends three dispatches
        seconds apart; three notes for one intent buries the feed the moment
        the bridge works."""
        rows = (
            row("a", node="lavender"),
            row("b", node="sedona"),
            row("c", project="libs/platform_core"),
            row("d", agent="opus-other-0906"),
        )

        posts = announcements(rows)

        assert [(post["project"], post["agent"]) for post in posts] == [
            ("libs/platform_core", "opus-fleet-0906"),
            ("tools/fleet", "opus-fleet-0906"),
            ("tools/fleet", "opus-other-0906"),
        ]

    def test_the_body_leads_with_the_marker_and_the_tally(self) -> None:
        """The marker is the first token so ``task_feed(query=...)`` -- the
        one board surface that searches bodies -- finds these without
        depending on any render grammar of the board's own."""
        posts = announcements((row("a"), row("b", outcome="failed", exit_code=1)))

        assert posts[0]["body"].startswith(
            f"{MARKER} tools/fleet: 2 dispatch(es) ended (failed x1, passed x1)"
        )

    def test_each_line_names_the_run_node_outcome_exit_and_elapsed(self) -> None:
        posts = announcements((row("tools-fleet-1788633781"),))

        assert "tools-fleet-1788633781 lavender passed exit 0 103s" in posts[0]["body"]

    def test_a_dispatch_with_no_exit_code_says_so_rather_than_printing_minus_one(
        self,
    ) -> None:
        """A refused dispatch never started and a lost one never reported.
        Spelling that ``exit -1`` would be arithmetic on a number that means
        something else, and every reader would need to know the convention."""
        posts = announcements((row("a", outcome="refused", exit_code=NO_EXIT_CODE),))

        assert "a lavender refused no exit code" in posts[0]["body"]
        assert "-1" not in posts[0]["body"]

    def test_the_dispatching_session_is_tagged_last(self) -> None:
        posts = announcements((row("a"),))

        assert posts[0]["body"].endswith(
            "@opus-fleet-0906 your dispatch(es) reached terminal state"
        )

    def test_a_long_group_is_capped_and_says_how_many_it_dropped(self) -> None:
        """A capped post that did not say so would read as the whole story."""
        rows = tuple(row(f"run-{index}") for index in range(LINE_CAP + 3))

        body = announcements(rows)[0]["body"]

        assert f"{MARKER} tools/fleet: {LINE_CAP + 3} dispatch(es) ended" in body
        assert "run-0 lavender passed" in body
        assert f"run-{LINE_CAP}" not in body
        assert "+3 more, all in the workspace ledger" in body

    def test_output_is_ordered_so_two_runs_post_the_same_thing(self) -> None:
        """A cycle whose output depended on dict ordering would be
        untestable and would read differently on the board run to run."""
        rows = (row("c", project="z/last"), row("a", project="a/first"))

        assert [post["project"] for post in announcements(rows)] == ["a/first", "z/last"]
        assert announcements(rows) == announcements(rows)
