"""The watch acts on what each collect did (MCPs board task c1d48330).

On 2026-10-07 at 02:54:45Z lavender-wsl stopped answering ssh and the
watch's collect of row 24c4e934 answered "did not answer the read"; the
watch counted that as a closed run and stopped reading it, so the row
closed at the next start, 192 s after its check. And row 374f0656's lease
went unrenewed from 02:57:56Z to 03:02:54Z because the 03:00Z pass could
not read it. Each case runs the real :class:`fleet.cli.node_watch.RunWatch`
on a thread of its own against the launched demo run, with the node's ssh
answers and the collect faked, on a one-second poll.
"""

from __future__ import annotations

import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from fleet.cli.node_collected import Collected, CollectOutcome
from fleet.core import _test_hooks
from tests._node_agent_fixtures import _credentials_in_env, _sourced_config, launch
from tests._thread_fakes import await_event
from tests._watch_fixtures import (
    ENDED,
    STILL_RUNNING,
    AnsweringThen,
    collected,
    lavender_watch,
    never_settles,
    poll_every_second,
)
from tests.conftest import DEMO_RUN_ID, FakeRun

__all__ = ["_credentials_in_env", "_sourced_config"]


class TestACollectWhoseNodeDidNotAnswer:
    def test_is_not_counted_closed_and_the_run_is_collected_again_at_the_next_poll(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        loaded = poll_every_second(sourced_config)
        node = FakeRun([*ENDED, *ENDED])
        _test_hooks.run = node
        outcomes = [CollectOutcome.UNREACHABLE, CollectOutcome.SETTLED]
        attempts: list[str] = []
        settled = threading.Event()

        def settle(*, run_id: str) -> Collected:
            attempts.append(run_id)
            outcome = outcomes[len(attempts) - 1]
            if outcome is CollectOutcome.SETTLED:
                settled.set()
            return collected(run_id, outcome)

        watch = lavender_watch(loaded, settle)
        with ThreadPoolExecutor(max_workers=1) as pool:
            watching = pool.submit(watch.watch)
            with watch:
                watch.hold(frozenset({DEMO_RUN_ID}))
                await_event(settled, what="the second collect to settle the run")
            watching.result()

        assert attempts == [DEMO_RUN_ID, DEMO_RUN_ID]
        assert watch.closed() == 1
        assert watch.polls == 2
        assert len(node.calls) == 2 * len(ENDED)


class TestACollectOfARunNoLongerThisRunners:
    def test_lets_the_run_go_uncounted(self, sourced_config: pathlib.Path) -> None:
        """A cancel or a takeover: the next fire's collect pass stops it."""
        launch(sourced_config)
        loaded = poll_every_second(sourced_config)
        _test_hooks.run = FakeRun(list(ENDED))
        answered = threading.Event()

        def settle(*, run_id: str) -> Collected:
            answered.set()
            return collected(run_id, CollectOutcome.NOT_HELD)

        watch = lavender_watch(loaded, settle)
        with ThreadPoolExecutor(max_workers=1) as pool:
            watching = pool.submit(watch.watch)
            with watch:
                watch.hold(frozenset({DEMO_RUN_ID}))
                await_event(answered, what="the collect")
            watching.result()

        assert watch.closed() == 0
        assert watch.still_watched() == 0


class TestAnOwedRenewal:
    def test_is_made_at_the_first_read_that_finds_the_run_going_and_only_then(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        launch(sourced_config)
        loaded = poll_every_second(sourced_config)
        renewed: list[str] = []

        def settle(*, run_id: str) -> Collected:
            renewed.append(run_id)
            return collected(run_id, CollectOutcome.RENEWED)

        watch = lavender_watch(loaded, settle)
        # Closed once the second poll's read of the run has been answered.
        node = AnsweringThen(
            [*STILL_RUNNING, *STILL_RUNNING],
            after=2 * len(STILL_RUNNING) - 1,
            then=watch.close_if_idle,
        )
        _test_hooks.run = node
        with caplog.at_level("INFO"), ThreadPoolExecutor(max_workers=1) as pool:
            watching = pool.submit(watch.watch)
            watch.owe(frozenset({DEMO_RUN_ID}))
            watching.result()

        assert renewed == [DEMO_RUN_ID]
        assert watch.polls == 2
        assert watch.closed() == 0
        assert watch.still_watched() == 1
        assert len(node.runner.calls) == 2 * len(STILL_RUNNING)
        assert (
            f"lavender: {DEMO_RUN_ID} is still running and its renewal is owed; collecting it now"
        ) in [record.getMessage() for record in caplog.records]

    def test_stays_owed_while_the_collect_meets_a_node_that_does_not_answer(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        loaded = poll_every_second(sourced_config)
        _test_hooks.run = FakeRun([*STILL_RUNNING, *STILL_RUNNING])
        outcomes = [CollectOutcome.UNREACHABLE, CollectOutcome.RENEWED]
        attempts: list[str] = []
        renewed = threading.Event()

        def settle(*, run_id: str) -> Collected:
            attempts.append(run_id)
            outcome = outcomes[len(attempts) - 1]
            if outcome is CollectOutcome.RENEWED:
                renewed.set()
            return collected(run_id, outcome)

        watch = lavender_watch(loaded, settle)
        with ThreadPoolExecutor(max_workers=1) as pool:
            watching = pool.submit(watch.watch)
            with watch:
                watch.owe(frozenset({DEMO_RUN_ID}))
                await_event(renewed, what="the second collect to renew the run")
            watching.result()

        assert attempts == [DEMO_RUN_ID, DEMO_RUN_ID]
        assert watch.polls == 2
        assert watch.closed() == 0

    def test_of_a_run_the_ledger_does_not_call_live_holds_nothing(
        self, sourced_config: pathlib.Path
    ) -> None:
        loaded = poll_every_second(sourced_config)
        node = FakeRun([])
        _test_hooks.run = node
        watch = lavender_watch(loaded, never_settles)

        watch.owe(frozenset({"a-run-the-ledger-never-had"}))

        assert watch.still_watched() == 0
        assert node.calls == []
