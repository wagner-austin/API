"""The cursor document: where it lives, what it holds, and round-tripping it."""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONTypeError

from board_watch import _test_hooks
from board_watch.state import (
    WatchState,
    decode_state,
    encode_state,
    load_state,
    save_state,
    state_path,
)
from tests.conftest import FakeFiles

DIRECTORY = pathlib.Path("/cursors")


def test_the_document_is_named_for_the_agent() -> None:
    """Two watchers on one board are at different positions.

    A shared filename would give whichever ran last the other's place in the
    feed, which reads as events going missing.
    """
    assert state_path("opus-a-0905", DIRECTORY) == DIRECTORY / "opus-a-0905.json"
    assert state_path("opus-b-0905", DIRECTORY) != state_path("opus-a-0905", DIRECTORY)


@pytest.mark.parametrize("cursor", ["abc123==", None])
def test_a_position_round_trips(cursor: str | None) -> None:
    """Including the primed-against-an-empty-board case, which holds None."""
    state = WatchState(agent="opus-a-0905", cursor=cursor)
    assert decode_state(encode_state(state)) == state


def test_a_document_that_cannot_be_read_raises_rather_than_restarting(
    files: FakeFiles,
) -> None:
    """Starting over means replaying the whole feed, so it is not a recovery.

    The validator's own error propagates: this package does not catch it to
    substitute a fresh position, because a watcher that silently rewinds is
    indistinguishable from one that works until it wakes a session with a
    hundred old mentions.
    """
    files.contents[state_path("opus-a-0905", DIRECTORY)] = '{"cursor": "x"}'
    with pytest.raises(JSONTypeError):
        load_state("opus-a-0905", DIRECTORY)


def test_no_document_means_no_position(files: FakeFiles) -> None:
    """Absence is the signal to prime, and is not an error."""
    assert load_state("opus-a-0905", DIRECTORY) is None
    assert files.contents == {}


def test_saving_then_loading_returns_the_same_position(files: FakeFiles) -> None:
    """The two halves have to agree on the path as well as the payload."""
    state = WatchState(agent="opus-a-0905", cursor="here")
    save_state(state, DIRECTORY)
    assert state_path("opus-a-0905", DIRECTORY) in files.contents
    assert load_state("opus-a-0905", DIRECTORY) == state


def test_the_default_directory_is_outside_the_working_directory() -> None:
    """A watcher polls from wherever its shell loop happens to be.

    A relative default would give the same agent a different position per
    directory, which is the same failure as sharing one file between agents.
    """
    from board_watch.state import DEFAULT_STATE_DIRECTORY

    assert DEFAULT_STATE_DIRECTORY.is_absolute()


def test_the_real_filesystem_hooks_create_the_parent_directory(
    tmp_path: pathlib.Path,
) -> None:
    """Exercises the production implementations, not the fake.

    The default state directory does not exist on a fresh machine, so a
    writer that assumed it did would fail on exactly the first run.
    """
    target = tmp_path / "made" / "up" / "opus-a-0905.json"
    assert _test_hooks.file_exists(target) is False
    _test_hooks.write_text(target, "contents")
    assert _test_hooks.file_exists(target) is True
    assert _test_hooks.read_text(target) == "contents"


__all__ = [
    "test_a_document_that_cannot_be_read_raises_rather_than_restarting",
    "test_a_position_round_trips",
    "test_no_document_means_no_position",
    "test_saving_then_loading_returns_the_same_position",
    "test_the_default_directory_is_outside_the_working_directory",
    "test_the_document_is_named_for_the_agent",
    "test_the_real_filesystem_hooks_create_the_parent_directory",
]
