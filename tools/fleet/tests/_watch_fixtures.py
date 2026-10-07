"""What the watch's cases share: the node's answers, the watch, a faked settle.

Lifted out of ``test_node_watch.py`` when the watch's collect outcomes and
owed renewals got cases of their own (MCPs board task c1d48330), so both
files drive the same real :class:`fleet.cli.node_watch.RunWatch` the same way.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Sequence

from platform_core.json_utils import dump_json_str

from fleet.cli import _config, node_watch
from fleet.cli.node_collected import Collected, CollectOutcome
from fleet.core import _test_hooks
from tests._node_agent_fixtures import NPM_CI, sourced_document
from tests.conftest import DEMO_NOW, FakeRun, ok

#: One read of a run's result that finds none: the collect script sent, then
#: run, printing nothing.
STILL_RUNNING = (ok(""), ok(""))

#: The same read once the build has written its result.
ENDED = (ok(""), ok(f"0 {DEMO_NOW + 72}"))


def poll_every_second(config_path: pathlib.Path) -> _config.LoadedWorkspace:
    """Declare a one-second poll and load the workspace as the runner does.

    Args:
        config_path: The workspace document.

    Returns:
        It, loaded.
    """
    document = sourced_document((NPM_CI,))
    document["node_poll_seconds"] = 1
    config_path.write_text(dump_json_str(document), encoding="utf-8")
    return _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})


def lavender_watch(
    loaded: _config.LoadedWorkspace, settle: node_watch.Settle
) -> node_watch.RunWatch:
    """Lavender's watch.

    Args:
        loaded: The workspace.
        settle: What it collects a run with.

    Returns:
        The watch, holding nothing.
    """
    node = loaded.workspace["nodes"]["lavender"]
    return node_watch.RunWatch(loaded, alias="lavender", node=node, settle=settle)


def never_settles(*, run_id: str) -> Collected:
    """A collect a case asserts is never reached.

    Args:
        run_id: The run.

    Raises:
        AssertionError: Always.
    """
    raise AssertionError(f"settled {run_id}")


def collected(run_id: str, outcome: CollectOutcome) -> Collected:
    """What a faked collect answers.

    Args:
        run_id: The run.
        outcome: What it did.

    Returns:
        The collect, as :func:`fleet.cli.node_collect.collect_one_job` answers one.
    """
    return Collected(outcome=outcome, line=f"{run_id}: {outcome}")


class AnsweringThen:
    """A node that answers from a script and does something once a given call is answered.

    Satisfies :class:`~fleet.core._test_hooks.RunProtocol`.

    Attributes:
        runner: The scripted node, whose calls a case reads.
    """

    runner: FakeRun

    def __init__(
        self,
        replies: Sequence[_test_hooks.CommandResult],
        *,
        after: int,
        then: Callable[[], bool],
    ) -> None:
        """Bind the script and the action.

        Args:
            replies: One result per expected call.
            after: The index, from zero, of the call the action follows.
            then: The action.
        """
        self.runner = FakeRun(replies)
        self._after = after
        self._then = then

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        stdin_bytes: bytes | None = None,
        unset_env: Sequence[str] = (),
        set_env: Sequence[tuple[str, str]] = (),
    ) -> _test_hooks.CommandResult:
        """Answer the call, then act if it is the one named.

        Args:
            argv: The command.
            timeout_seconds: The deadline the caller chose.
            stdin_bytes: Its standard input, or None.
            unset_env: The variables the caller withheld from the child.
            set_env: The variables the caller set in the child.

        Returns:
            The scripted result.
        """
        index = len(self.runner.calls)
        answer = self.runner(
            argv,
            timeout_seconds=timeout_seconds,
            stdin_bytes=stdin_bytes,
            unset_env=unset_env,
            set_env=set_env,
        )
        if index == self._after:
            self._then()
        return answer


__all__ = [
    "ENDED",
    "STILL_RUNNING",
    "AnsweringThen",
    "collected",
    "lavender_watch",
    "never_settles",
    "poll_every_second",
]
