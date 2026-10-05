"""A watch settles two ended runs beside each other (MCPs board task 8993c306).

On 2026-10-05 lavender-wsl's libs/platform_core run ended at 08:20:00Z and
its settle ran to 08:20:13Z, and row ebb9009c, whose check ended a second
later, was read only after it and closed 22.1 s after its check. Each case
here runs the real :class:`fleet.cli.node_watch.RunWatch` on a thread of its
own over two launched runs that have both ended, every read of the node
answering with a written result. Each settle waits until both have begun,
which a watch settling one run at a time never reaches, and a handover
asked for meanwhile is refused; when both settles raise, the watch keeps
the first failure and raises it once both have finished.
"""

from __future__ import annotations

import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import dump_json_str

from fleet.cli import _config, node_watch
from fleet.core import _test_hooks, records
from tests._node_agent_fixtures import (
    NPM_CI,
    _credentials_in_env,
    _sourced_config,
    launch,
    sourced_document,
)
from tests._thread_fakes import WAIT_SECONDS, await_event
from tests.conftest import DEMO_NOW, DEMO_RUN_ID, FakeRun, ok

__all__ = ["_credentials_in_env", "_sourced_config"]

#: The second run, launched beside the demo run on lavender.
SECOND_RUN = "libs-demo-lavender-second"

#: Both runs.
BOTH = frozenset({DEMO_RUN_ID, SECOND_RUN})


def _two_ended_runs(config_path: pathlib.Path) -> _config.LoadedWorkspace:
    """Launch the demo run, record a second beside it, and say both have ended.

    The poll is one second, and every ssh call answers with a written
    result, which the script's send ignores, so the two reads may come in
    either order.

    Args:
        config_path: The workspace document.

    Returns:
        The workspace, loaded.
    """
    launch(config_path)
    document = sourced_document((NPM_CI,))
    document["node_poll_seconds"] = 1
    config_path.write_text(dump_json_str(document), encoding="utf-8")
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    second = records.read_ledger(loaded.ledger)[-1].copy()
    second["run_id"] = SECOND_RUN
    records.append_ledger(loaded.ledger, second)
    _test_hooks.run = FakeRun([ok(f"0 {DEMO_NOW + 72}")] * 4)
    return loaded


def _watch(loaded: _config.LoadedWorkspace, settle: node_watch.Settle) -> node_watch.RunWatch:
    """Lavender's watch.

    Args:
        loaded: The workspace.
        settle: What it settles an ended run with.

    Returns:
        The watch, holding nothing.
    """
    node = loaded.workspace["nodes"]["lavender"]
    return node_watch.RunWatch(loaded, alias="lavender", node=node, settle=settle)


class TestTwoRunsThatEndTogether:
    def test_are_settled_beside_each_other_and_no_handover_cuts_either_short(
        self, sourced_config: pathlib.Path
    ) -> None:
        loaded = _two_ended_runs(sourced_config)
        begun = {run_id: threading.Event() for run_id in BOTH}
        release = threading.Event()
        settled: list[str] = []

        def settle(*, run_id: str) -> str:
            begun[run_id].set()
            await_event(release, what="the case's go-ahead to finish the settles")
            settled.append(run_id)
            return f"{run_id}: settled"

        watch = _watch(loaded, settle)
        with ThreadPoolExecutor(max_workers=1) as pool:
            watching = pool.submit(watch.watch)
            with watch:
                watch.hold(BOTH)
                for run_id in sorted(BOTH):
                    await_event(begun[run_id], what=f"{run_id}'s settle to begin")
                refused = watch.close_if_idle()
                release.set()
            watching.result()

        assert refused is False
        assert frozenset(settled) == BOTH
        assert watch.closed() == 2
        assert watch.polls == 1


class TestTwoSettlesThatRaise:
    def test_end_the_watch_with_the_first_failure_once_both_have_finished(
        self, sourced_config: pathlib.Path
    ) -> None:
        loaded = _two_ended_runs(sourced_config)
        both_begun = threading.Barrier(2, timeout=WAIT_SECONDS)

        def settle(*, run_id: str) -> str:
            both_begun.wait()
            raise AppError(FleetErrorCode.QUEUE_ANSWER_MALFORMED, f"{run_id}: nonsense")

        watch = _watch(loaded, settle)
        with ThreadPoolExecutor(max_workers=1) as pool:
            watching = pool.submit(watch.watch)
            watch.hold(BOTH)
            with pytest.raises(AppError, match="nonsense"):
                watching.result()

        assert watch.closed() == 0
        assert watch.close_if_idle() is True
