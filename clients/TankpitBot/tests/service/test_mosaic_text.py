"""The label over each four-up tile, and how it reaches the encoder."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.service.mosaic_text import TileTextWriter, bot_label, waiting_label
from tankpit_bot.types.rooms import DEFAULT_LOBBY_ROOM


class TestLabels:
    """The label of a live tile and of a waiting one."""

    def test_a_live_tile_names_its_bot_and_room(self) -> None:
        """Instance, then the room it plays in."""
        assert bot_label("demo-2", "World") == "demo-2  World"

    def test_an_unnamed_room_is_the_default_room_by_name(self) -> None:
        """An empty room is the lobby default, written out rather than blank."""
        assert bot_label("artax", "") == f"artax  {DEFAULT_LOBBY_ROOM}"

    def test_a_waiting_tile_is_counted_from_one_and_says_it_waits(self) -> None:
        """Position 0 is the first slot a viewer sees."""
        assert [waiting_label(position) for position in range(4)] == [
            "Slot 1  Waiting for a bot",
            "Slot 2  Waiting for a bot",
            "Slot 3  Waiting for a bot",
            "Slot 4  Waiting for a bot",
        ]


@pytest.fixture()
def restore_replace_text() -> Generator[None, None, None]:
    """Put the real replace hook back after a test that records it.

    Yields:
        Nothing; the fixture exists for its teardown.
    """
    yield
    top_hooks.replace_text = top_hooks._real_replace_text


def test_the_writer_replaces_atomically_and_skips_repeats(
    tmp_path: Path, restore_replace_text: None
) -> None:
    """A second write of the same words never touches the file."""
    _ = restore_replace_text
    replaced: list[tuple[Path, str]] = []

    def recording_replace(path: Path, content: str) -> None:
        replaced.append((path, content))
        top_hooks._real_replace_text(path, content)

    top_hooks.replace_text = recording_replace
    writer = TileTextWriter()
    target = tmp_path / "text" / "tile0-label.txt"

    writer.write(target, waiting_label(0))
    writer.write(target, waiting_label(0))
    writer.write(target, bot_label("alpha", "World"))

    assert replaced == [(target, waiting_label(0)), (target, "alpha  World")]
    assert target.read_text(encoding="utf-8") == "alpha  World"
