"""The command end to end, against fakes for everything outside the process."""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.error_codes_tooling import BoardWatchErrorCode
from platform_core.errors import AppError

from board_watch import _test_hooks
from board_watch.cli.watch import entrypoint, main
from board_watch.state import WatchState, decode_state, save_state, state_path
from board_watch.watch import MAX_LIMIT
from tests.conftest import (
    LIVE_MENTION_LINE,
    LIVE_TASK_LINE,
    FakeEmit,
    FakeEnv,
    FakeFiles,
    FakeHttpPost,
    ok,
    page_text,
    sent_arguments,
    set_environment,
    tool_text,
)

AGENT = "opus-nclex-licensure-0904"
DIRECTORY = pathlib.Path("/cursors")
ARGV = ("--agent", AGENT, "--state", str(DIRECTORY))


def test_the_first_call_primes_and_announces_without_replaying(
    files: FakeFiles, emitted: FakeEmit
) -> None:
    """A watcher must not wake a session with mentions it has already read.

    The board here holds two existing mentions. Neither is emitted; only the
    armed notice is, and the recorded position is the end of the feed.
    """
    set_environment()
    _test_hooks.http_post = FakeHttpPost(
        [
            ok(tool_text(page_text([LIVE_MENTION_LINE], "one"))),
            ok(tool_text(page_text([LIVE_TASK_LINE], None))),
            ok(tool_text(page_text([LIVE_TASK_LINE], "the-true-end"))),
        ]
    )
    assert main(list(ARGV)) == 0
    assert len(emitted.lines) == 1
    assert emitted.lines[0].startswith(f"BOARD WATCH armed for @{AGENT}")
    assert "BOARD MENTION" not in emitted.lines[0]
    # The END of the partial page, not the full-page boundary before it.
    # Recording "one" here is what made the first live run announce two
    # mentions that predated arming.
    assert decode_state(files.contents[state_path(AGENT, DIRECTORY)])["cursor"] == "the-true-end"


def test_a_later_call_emits_only_what_arrived_since(files: FakeFiles, emitted: FakeEmit) -> None:
    """The whole point: one line per new mention, and the position moves."""
    set_environment()
    save_state(WatchState(agent=AGENT, cursor="start"), DIRECTORY)
    _test_hooks.http_post = FakeHttpPost(
        [ok(tool_text(page_text([LIVE_MENTION_LINE, LIVE_TASK_LINE], "later")))]
    )
    assert main(list(ARGV)) == 0
    assert len(emitted.lines) == 2
    assert emitted.lines[0].startswith("BOARD MENTION from opus-lavender-gpu-0824")
    assert emitted.lines[1].startswith("BOARD MENTION from fable-brain-audit-0903")
    assert decode_state(files.contents[state_path(AGENT, DIRECTORY)])["cursor"] == "later"


def test_a_quiet_poll_emits_nothing_and_keeps_its_place(
    files: FakeFiles, emitted: FakeEmit
) -> None:
    """Silence is the common case and must not disturb the position."""
    set_environment()
    save_state(WatchState(agent=AGENT, cursor="start"), DIRECTORY)
    _test_hooks.http_post = FakeHttpPost([ok(tool_text(page_text([], None)))])
    assert main(list(ARGV)) == 0
    assert emitted.lines == []
    assert decode_state(files.contents[state_path(AGENT, DIRECTORY)])["cursor"] == "start"


def test_the_optional_filters_reach_the_board(files: FakeFiles, emitted: FakeEmit) -> None:
    """Room and kind are the two ways to narrow what wakes you."""
    set_environment()
    save_state(WatchState(agent=AGENT, cursor="s"), DIRECTORY)
    poster = FakeHttpPost([ok(tool_text(page_text([], None)))])
    _test_hooks.http_post = poster
    assert main([*ARGV, "--room", "main", "--kind", "status_change", "--limit", "10"]) == 0

    arguments = sent_arguments(poster.bodies[0])
    assert arguments["room"] == "main"
    assert arguments["kind"] == "status_change"
    assert arguments["limit"] == 10


@pytest.mark.parametrize("bad", ["0", str(MAX_LIMIT + 1), "-4"])
def test_an_out_of_range_limit_is_refused_rather_than_clamped(
    bad: str, files: FakeFiles, emitted: FakeEmit
) -> None:
    """Silently giving a caller 200 when they asked for 500 leaves them wrong."""
    set_environment()
    with pytest.raises(ValueError):
        main([*ARGV, "--limit", bad])


def test_missing_credentials_surface_as_their_own_code(files: FakeFiles, emitted: FakeEmit) -> None:
    """The command does not start polling with a half-configured environment."""
    _test_hooks.env = FakeEnv({})
    with pytest.raises(AppError) as raised:
        main(list(ARGV))
    assert raised.value.code is BoardWatchErrorCode.API_KEY_MISSING


def test_the_entrypoint_exits_with_the_status(files: FakeFiles, emitted: FakeEmit) -> None:
    """The console script is what poetry installs, so it is what runs."""
    set_environment()
    save_state(WatchState(agent=AGENT, cursor="s"), DIRECTORY)
    _test_hooks.http_post = FakeHttpPost([ok(tool_text(page_text([], None)))])
    original = list(sys.argv)
    sys.argv[:] = ["board-watch", *ARGV]
    try:
        with pytest.raises(SystemExit) as raised:
            entrypoint()
        assert raised.value.code == 0
    finally:
        sys.argv[:] = original


def test_running_as_a_module_actually_runs(files: FakeFiles, emitted: FakeEmit) -> None:
    """The half that silently goes missing without an ``if __name__`` block.

    Without it ``python -m board_watch.cli.watch`` imports the module, runs
    nothing and exits 0 -- an empty result, which from a subscriber's side
    reads as "no mentions" rather than "nothing ran".
    """
    set_environment()
    save_state(WatchState(agent=AGENT, cursor="s"), DIRECTORY)
    _test_hooks.http_post = FakeHttpPost([ok(tool_text(page_text([], None)))])
    module_name = "board_watch.cli.watch"
    saved_argv = list(sys.argv)
    saved_module = sys.modules.pop(module_name, None)
    sys.argv = ["board-watch", *ARGV]
    try:
        with pytest.raises(SystemExit) as raised:
            runpy.run_module(module_name, run_name="__main__", alter_sys=False)
    finally:
        sys.argv[:] = saved_argv
        if saved_module is not None:
            sys.modules[module_name] = saved_module
    assert raised.value.code == 0


def test_the_default_state_directory_is_used_when_none_is_given(
    files: FakeFiles, emitted: FakeEmit
) -> None:
    """A caller composing a shell loop should not have to name a path."""
    set_environment()
    _test_hooks.http_post = FakeHttpPost([ok(tool_text(page_text([], None)))])
    assert main(["--agent", AGENT]) == 0
    from board_watch.state import DEFAULT_STATE_DIRECTORY

    assert state_path(AGENT, DEFAULT_STATE_DIRECTORY) in files.contents


__all__ = [
    "test_a_later_call_emits_only_what_arrived_since",
    "test_a_quiet_poll_emits_nothing_and_keeps_its_place",
    "test_an_out_of_range_limit_is_refused_rather_than_clamped",
    "test_missing_credentials_surface_as_their_own_code",
    "test_running_as_a_module_actually_runs",
    "test_the_default_state_directory_is_used_when_none_is_given",
    "test_the_entrypoint_exits_with_the_status",
    "test_the_first_call_primes_and_announces_without_replaying",
    "test_the_optional_filters_reach_the_board",
]
