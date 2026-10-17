"""The four-up director, driving a real registry and real child processes.

The registry is a real :class:`FleetManager` over the recording bot
spawner the other fleet suites use. The encoder is a REAL child process
started through the production spawner (:class:`SubstitutingSpawner`
runs a Python stand-in in ffmpeg's place and records the argv ffmpeg
would have had), so start, stop and exit are the operating system's.
The tile label files are written for real, under a directory given
relative to the working directory, which is how the fleet names its
own (``runs/fleet/mosaic``): an absolute Windows path would carry a
drive colon the filter graph refuses.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.runtime_artifacts import bot_run_dir
from tankpit_bot.service.fleet_manager import FleetManager
from tankpit_bot.service.mosaic_director import (
    MOSAIC_EXIT_LIMIT,
    MosaicDirector,
    MosaicStatusDict,
    decode_mosaic_status,
    direct_mosaic,
    encode_mosaic_status,
)
from tankpit_bot.stream import _test_hooks as stream_hooks
from tankpit_bot.stream.capture import HLS_PLAYLIST_FILENAME
from tankpit_bot.stream.helper_process import PROCESS_END_TIMEOUT_SECONDS
from tankpit_bot.stream.mosaic import tile_label_path
from tests.service._fleet_fixtures import (
    _FakeSpawner,
    _restore_account_hooks,
    _with_account_pool,
)
from tests.stream._capture_fixtures import SubstitutingSpawner, sleeper_argv

ACCOUNTS = ("alpha", "bravo", "charlie", "delta", "echo")


def _exiter_argv(code: int) -> list[str]:
    """A child that exits at once with ``code``.

    Args:
        code: The exit code.

    Returns:
        Its argv.
    """
    return [sys.executable, "-c", f"raise SystemExit({code})"]


class Rig:
    """One director over a real registry, with the bots' readiness set by hand.

    Attributes:
        manager: The registry.
        bots: The bot spawner, whose process doubles a test ends.
        encoders: The encoder spawner.
        director: The director under test.
        ready: Instances whose own encoder has written a playlist.
    """

    def __init__(self, bots: _FakeSpawner, encoders: SubstitutingSpawner, root: Path) -> None:
        """Build the registry and the director.

        Args:
            bots: The bot spawner.
            encoders: The encoder spawner.
            root: The director's directory, relative to the cwd.
        """
        self.manager = FleetManager()
        self.bots = bots
        self.encoders = encoders
        self.director = MosaicDirector(self.manager, root)
        self.ready: set[str] = set()
        accounts_exist = top_hooks.path_exists
        playlists = self.ready

        def exists(path: Path) -> bool:
            if path.name == HLS_PLAYLIST_FILENAME:
                return any(path == bot_run_dir(name) / "hls" / path.name for name in playlists)
            return accounts_exist(path)

        top_hooks.path_exists = exists

    def start(self, instance: str, *, room: str = "", ready: bool = True) -> int:
        """Spawn a bot and, unless told otherwise, mark its stream up.

        Args:
            instance: Instance and account name.
            room: Its room.
            ready: Whether its own encoder has written a playlist.

        Returns:
            Its display number (its service port).
        """
        row = self.manager.spawn(instance=instance, account=instance, kills=0, seconds=0, room=room)
        if ready:
            self.ready.add(instance)
        return row["service_port"]

    def end(self, instance: str) -> None:
        """Make a bot's process exit.

        Args:
            instance: Its name.
        """
        index = [row["instance"] for row in self.manager.report()].index(instance)
        self.bots.processes[index].returncode = 0

    def label(self, position: int) -> str:
        """Read a position's label file.

        Args:
            position: Grid position.

        Returns:
            The label the encoder draws.
        """
        return tile_label_path(self.director.text_dir, position).read_text(encoding="utf-8")

    def displays(self, call: int) -> list[str]:
        """Return the X displays one encoder command records.

        Args:
            call: Which encoder start, oldest first.

        Returns:
            Each ``:<display>`` input, in input order.
        """
        return [word for word in self.encoders.commands[call] if word.startswith(":")]


@pytest.fixture()
def rig(spawner: _FakeSpawner, tmp_path: Path) -> Generator[Rig, None, None]:
    """A five-account registry and a director rooted in the test's tree.

    Yields:
        The rig; its encoder children are ended after the test.
    """
    originals = _with_account_pool(*ACCOUNTS)
    encoders = SubstitutingSpawner([sleeper_argv() for _ in range(12)])
    stream_hooks.spawn_capture_process = encoders
    root = Path(os.path.relpath(tmp_path / "mosaic"))
    built = Rig(spawner, encoders, root)
    yield built
    built.director.stop()
    for process in encoders.processes:
        if process.poll() is None:
            process.kill()
            process.wait()
    _restore_account_hooks(originals)


class TestPlacement:
    """Which bot shows where."""

    def test_an_empty_fleet_still_streams_four_waiting_tiles(self, rig: Rig) -> None:
        """The first tick starts an encoder with no live tile."""
        rig.director.tick()

        assert rig.director.status() == {"tiles": ["", "", "", ""], "running": True, "failure": ""}
        assert "-i" not in rig.encoders.commands[0]
        assert rig.label(0) == "Slot 1  Waiting for a bot"
        assert rig.label(3) == "Slot 4  Waiting for a bot"
        assert rig.encoders.log_paths[0] == rig.director.log_path
        assert rig.director.hls_dir.is_dir()

    def test_ready_bots_fill_positions_in_start_order(self, rig: Rig) -> None:
        """Earliest started first; each records its own display."""
        alpha = rig.start("alpha")
        bravo = rig.start("bravo", room="World")
        rig.director.tick()

        assert rig.director.status()["tiles"] == ["alpha", "bravo", "", ""]
        assert rig.displays(0) == [f":{alpha}", f":{bravo}"]
        assert rig.label(0) == "alpha  Practice"
        assert rig.label(1) == "bravo  World"
        assert rig.label(2) == "Slot 3  Waiting for a bot"

    def test_a_bot_whose_stream_is_not_up_waits(self, rig: Rig) -> None:
        """No playlist yet means no display to record yet."""
        rig.start("alpha", ready=False)
        rig.director.tick()
        assert rig.director.status()["tiles"] == ["", "", "", ""]

        rig.ready.add("alpha")
        rig.director.tick()
        assert rig.director.status()["tiles"] == ["alpha", "", "", ""]

    def test_a_bot_keeps_its_tile_and_a_freed_tile_takes_the_next(self, rig: Rig) -> None:
        """Positions are sticky; the freed one goes to the newcomer."""
        rig.start("alpha")
        rig.start("bravo")
        rig.director.tick()
        rig.end("alpha")
        rig.director.tick()

        assert rig.director.status()["tiles"] == ["", "bravo", "", ""]
        assert rig.label(0) == "Slot 1  Waiting for a bot"

        rig.start("charlie")
        rig.director.tick()
        assert rig.director.status()["tiles"] == ["charlie", "bravo", "", ""]
        assert rig.label(0) == "charlie  Practice"

    def test_only_four_bots_are_shown(self, rig: Rig) -> None:
        """A fifth waits for a free position."""
        for name in ACCOUNTS:
            rig.start(name)
        rig.director.tick()

        assert rig.director.status()["tiles"] == ["alpha", "bravo", "charlie", "delta"]
        rig.end("bravo")
        rig.director.tick()
        assert rig.director.status()["tiles"] == ["alpha", "echo", "charlie", "delta"]


class TestEncoder:
    """When the encoder is restarted, and when it is not."""

    def test_nothing_changing_restarts_nothing(self, rig: Rig) -> None:
        """A tick with the same tiles and a running encoder leaves it be."""
        rig.start("alpha")
        rig.director.tick()
        rig.director.tick()
        rig.director.tick()
        assert len(rig.encoders.commands) == 1

    def test_a_change_ends_the_old_encoder_before_starting_the_new(self, rig: Rig) -> None:
        """The swap is stop-then-start, with the production stop bound."""
        rig.director.tick()
        alpha = rig.start("alpha")
        rig.director.tick()

        first, second = rig.encoders.processes
        assert first.wait_timeouts == [PROCESS_END_TIMEOUT_SECONDS]
        assert second.poll() is None
        assert rig.displays(0) == []
        assert rig.displays(1) == [f":{alpha}"]

    def test_an_encoder_that_exits_is_restarted_then_reported(self, rig: Rig) -> None:
        """Exits with nothing changed are retried up to the limit, then a fault."""
        rig.encoders = SubstitutingSpawner([_exiter_argv(3) for _ in range(MOSAIC_EXIT_LIMIT)])
        stream_hooks.spawn_capture_process = rig.encoders
        rig.start("alpha")
        for expected in range(1, MOSAIC_EXIT_LIMIT + 1):
            rig.director.tick()
            rig.encoders.processes[-1].wait(None)
            assert len(rig.encoders.commands) == expected

        rig.director.tick()
        status = rig.director.status()
        assert len(rig.encoders.commands) == MOSAIC_EXIT_LIMIT
        assert status["running"] is False
        assert status["failure"].startswith(
            f"mosaic ffmpeg exited 3 {MOSAIC_EXIT_LIMIT} times in a row"
            " recording the same 1 tile(s)"
        )

        rig.director.tick()
        assert len(rig.encoders.commands) == MOSAIC_EXIT_LIMIT

    def test_a_reported_fault_clears_when_the_bots_change(self, rig: Rig) -> None:
        """A new set of bots is a new attempt."""
        rig.encoders = SubstitutingSpawner(
            [_exiter_argv(1) for _ in range(MOSAIC_EXIT_LIMIT)] + [sleeper_argv()]
        )
        stream_hooks.spawn_capture_process = rig.encoders
        rig.start("alpha")
        for _ in range(MOSAIC_EXIT_LIMIT):
            rig.director.tick()
            rig.encoders.processes[-1].wait(None)
        rig.director.tick()
        assert rig.director.status()["failure"]

        rig.start("bravo")
        rig.director.tick()
        assert rig.director.status() == {
            "tiles": ["alpha", "bravo", "", ""],
            "running": True,
            "failure": "",
        }

    def test_stop_ends_the_encoder_and_is_safe_twice(self, rig: Rig) -> None:
        """Stopping twice is a no-op the second time."""
        rig.director.tick()
        rig.director.stop()
        rig.director.stop()

        assert rig.encoders.processes[0].wait_timeouts == [PROCESS_END_TIMEOUT_SECONDS]
        assert rig.director.status()["running"] is False


class TestLoop:
    """The task the fleet runs."""

    @pytest.mark.asyncio
    async def test_the_loop_ticks_until_stopped_then_ends_the_encoder(self, rig: Rig) -> None:
        """One tick, then the stop signal ends the loop and the encoder."""
        stop_event = asyncio.Event()

        async def stop_soon() -> None:
            while not rig.encoders.processes:
                await asyncio.sleep(0)
            stop_event.set()

        await asyncio.gather(direct_mosaic(rig.director, stop_event, tick_seconds=0.0), stop_soon())

        assert rig.encoders.processes[0].wait_timeouts == [PROCESS_END_TIMEOUT_SECONDS]
        assert rig.director.status()["running"] is False


class TestStatusCodec:
    """The status round-trips through JSON, and a bad one is refused."""

    def test_the_status_round_trips(self) -> None:
        """Encode then decode is the identity."""
        status = MosaicStatusDict(tiles=["alpha", "", "", ""], running=True, failure="")
        encoded = encode_mosaic_status(status)
        assert encoded == {"tiles": ["alpha", "", "", ""], "running": True, "failure": ""}
        assert decode_mosaic_status(encoded) == status

    def test_a_status_not_naming_every_position_is_refused(self) -> None:
        """One entry per grid position, no more and no fewer."""
        with pytest.raises(ValueError, match="tiles must name 4 positions, got 2"):
            decode_mosaic_status({"tiles": ["", ""], "running": False, "failure": ""})

    def test_a_status_with_a_mistyped_field_is_refused(self) -> None:
        """Each field is validated."""
        with pytest.raises(JSONTypeError):
            decode_mosaic_status({"tiles": ["", "", "", ""], "running": "yes", "failure": ""})
