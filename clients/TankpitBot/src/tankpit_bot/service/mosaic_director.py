"""Keeps the fleet's four-up stream showing the bots that are playing.

The fleet manager owns one :class:`MosaicDirector`, and the director
owns one encoder (:mod:`tankpit_bot.stream.mosaic`) recording up to
four bots' displays into ``runs/fleet/mosaic/hls``. Once a second it:

* places bots into the four grid positions. A bot keeps its position
  for as long as it runs, so a viewer's eye stays on the same tank; a
  freed position takes the earliest-started bot not yet shown. A bot
  is placed once its own encoder has written a playlist, which is the
  point at which its display and its sound server both exist;
* rewrites each position's label file (the encoder reloads it every
  frame) with the bot's name and room, or the waiting words for an
  empty position;
* restarts the encoder when the placed set changes, because ffmpeg
  cannot swap an input. The playlist continues across the restart
  (``append_list``), so a player that is already playing keeps
  playing.

An encoder that exits while the placed set has not changed is
restarted too: the usual cause is a bot whose display went away a tick
before its process did. Three such exits in a row with nothing changed
is a fault rather than a race, so the director stops restarting,
records why, and reports it through :meth:`MosaicDirector.status`
until the set of bots changes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_str,
    require_bool,
    require_list,
    require_str,
)
from platform_core.logging import get_logger
from typing_extensions import TypedDict

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.runtime_artifacts import bot_run_dir
from tankpit_bot.service.fleet_bot import FleetBotDict
from tankpit_bot.service.fleet_manager import FleetManager
from tankpit_bot.service.mosaic_text import TileTextWriter, bot_label, waiting_label
from tankpit_bot.stream import _test_hooks as stream_hooks
from tankpit_bot.stream.capture import FFMPEG_PROGRAM, HLS_PLAYLIST_FILENAME
from tankpit_bot.stream.helper_process import end_process
from tankpit_bot.stream.mosaic import (
    MOSAIC_POSITIONS,
    MosaicTileDict,
    mosaic_ffmpeg_command,
    tile_label_path,
)

log = get_logger(__name__)

MOSAIC_DIR = Path("runs") / "fleet" / "mosaic"
"""Where the four-up stream lives, beside the bots' own ``runs/bot``.
Relative, as every run directory is: the fleet runs from ``/app``,
where ``runs`` is the mount the bots write to."""

MOSAIC_TICK_SECONDS = 1.0
"""How often the director re-places bots and checks its encoder."""

MOSAIC_EXIT_LIMIT = 3
"""Consecutive encoder exits, with nothing placed changing, after which
the director stops restarting it and reports a fault."""


class MosaicStatusDict(TypedDict):
    """What the four-up stream is showing, as the operator reads it.

    Attributes:
        tiles: The instance shown at each position, in reading order;
            an empty string is a waiting tile.
        running: Whether the encoder is running.
        failure: Why the director stopped restarting the encoder;
            empty while it has not.
    """

    tiles: list[str]
    running: bool
    failure: str


def encode_mosaic_status(status: MosaicStatusDict) -> JSONObject:
    """Encode :class:`MosaicStatusDict` for the wire.

    Args:
        status: The status to encode.

    Returns:
        JSON-serializable object.
    """
    tiles: list[JSONValue] = list(status["tiles"])
    return {"tiles": tiles, "running": status["running"], "failure": status["failure"]}


def decode_mosaic_status(data: JSONObject) -> MosaicStatusDict:
    """Validate and decode a :class:`MosaicStatusDict` from JSON.

    Args:
        data: JSON object to validate.

    Returns:
        The status.

    Raises:
        JSONTypeError: A field is missing or has the wrong type.
        ValueError: The tile list is not one entry per grid position.
    """
    tiles = [narrow_json_to_str(tile) for tile in require_list(data, "tiles")]
    if len(tiles) != MOSAIC_POSITIONS:
        raise ValueError(f"tiles must name {MOSAIC_POSITIONS} positions, got {len(tiles)}")
    return MosaicStatusDict(
        tiles=tiles,
        running=require_bool(data, "running"),
        failure=require_str(data, "failure"),
    )


def _start_order(row: FleetBotDict) -> tuple[int, str]:
    """Order bots by when they started, then by name.

    Args:
        row: A bot's registry row.

    Returns:
        The sort key: earliest start first, ties by instance name.
    """
    return (row["started_ms"], row["instance"])


class MosaicDirector:
    """Places bots into the four-up stream and keeps its encoder running."""

    def __init__(self, manager: FleetManager, root: Path = MOSAIC_DIR) -> None:
        """Bind the director to the fleet; start nothing yet.

        Args:
            manager: The registry the bots are read from.
            root: Directory the stream, its text files and its log
                live under.
        """
        self._manager = manager
        self._root = root
        self._positions: list[str] = [""] * MOSAIC_POSITIONS
        self._text = TileTextWriter()
        self._encoder: stream_hooks.CaptureProcessProtocol | None = None
        self._encoded: list[MosaicTileDict] = []
        self._exits = 0
        self._failure = ""

    @property
    def hls_dir(self) -> Path:
        """The directory the stream's playlist and segments land in."""
        return self._root / "hls"

    @property
    def text_dir(self) -> Path:
        """The directory each tile's label file lives in."""
        return self._root / "text"

    @property
    def log_path(self) -> Path:
        """The file the encoder's console is captured to."""
        return self._root / "ffmpeg.log"

    def _ready(self, row: FleetBotDict) -> bool:
        """Return whether a bot can be recorded now.

        Args:
            row: The bot's registry row.

        Returns:
            True when it is running and its own encoder has written a
            playlist, which it does only once its display and its
            sound server are up.
        """
        playlist = bot_run_dir(row["instance"]) / "hls" / HLS_PLAYLIST_FILENAME
        return row["alive"] and top_hooks.path_exists(playlist)

    def _place(self) -> dict[str, FleetBotDict]:
        """Free the positions of bots that are gone and fill empty ones.

        Returns:
            Every bot that can be recorded now, by instance name.
        """
        ready = {row["instance"]: row for row in self._manager.report() if self._ready(row)}
        for position, instance in enumerate(self._positions):
            if instance and instance not in ready:
                self._positions[position] = ""
        shown = set(self._positions)
        waiting = sorted(
            (row for name, row in ready.items() if name not in shown),
            key=_start_order,
        )
        for position in range(MOSAIC_POSITIONS):
            if not self._positions[position] and waiting:
                self._positions[position] = waiting.pop(0)["instance"]
        return ready

    def _write_labels(self, ready: dict[str, FleetBotDict]) -> None:
        """Make every position's label say what it shows now.

        Args:
            ready: Every bot that can be recorded now, by instance.
        """
        for position, instance in enumerate(self._positions):
            if instance:
                label = bot_label(instance, ready[instance]["room"])
            else:
                label = waiting_label(position)
            self._text.write(tile_label_path(self.text_dir, position), label)

    def _start(self, tiles: list[MosaicTileDict]) -> None:
        """End any encoder, then start one recording ``tiles``.

        Args:
            tiles: The live tiles to record.

        Raises:
            OSError: ffmpeg is not installed.
            ValueError: The tiles cannot be laid out.
        """
        if self._encoder is not None:
            end_process(self._encoder, FFMPEG_PROGRAM)
            self._encoder = None
        self.hls_dir.mkdir(parents=True, exist_ok=True)
        command = mosaic_ffmpeg_command(tiles, self.text_dir, self.hls_dir)
        self._encoder = stream_hooks.spawn_capture_process(command, self.log_path)
        self._encoded = tiles
        log.info(
            "Mosaic: ffmpeg pid %d recording %d live tile(s): %s",
            self._encoder.pid,
            len(tiles),
            ", ".join(self._positions[tile["position"]] for tile in tiles) or "none",
        )

    def tick(self) -> None:
        """Re-place bots, refresh the text, and keep the encoder right.

        Raises:
            OSError: ffmpeg is not installed, or a text file cannot be
                written.
            ValueError: The tiles cannot be laid out.
        """
        ready = self._place()
        self._write_labels(ready)
        tiles = [
            MosaicTileDict(position=position, display=ready[instance]["service_port"])
            for position, instance in enumerate(self._positions)
            if instance
        ]
        if tiles != self._encoded or self._encoder is None:
            self._exits = 0
            self._failure = ""
            self._start(tiles)
            return
        code = self._encoder.poll()
        if code is None or self._failure:
            return
        self._exits += 1
        if self._exits >= MOSAIC_EXIT_LIMIT:
            self._failure = (
                f"mosaic ffmpeg exited {code} {self._exits} times in a row recording the same"
                f" {len(tiles)} tile(s); see {self.log_path.as_posix()}"
            )
            log.error("Mosaic: %s", self._failure)
            return
        log.warning(
            "Mosaic: ffmpeg exited %d with nothing placed changing (%d of %d); restarting",
            code,
            self._exits,
            MOSAIC_EXIT_LIMIT,
        )
        self._start(tiles)

    def status(self) -> MosaicStatusDict:
        """Return what the stream is showing.

        Returns:
            The tile at each position, whether the encoder runs, and
            any fault that stopped it being restarted.
        """
        running = self._encoder is not None and self._encoder.poll() is None
        return MosaicStatusDict(tiles=list(self._positions), running=running, failure=self._failure)

    def stop(self) -> None:
        """End the encoder. Safe to call twice."""
        if self._encoder is not None:
            end_process(self._encoder, FFMPEG_PROGRAM)
            self._encoder = None


async def direct_mosaic(
    director: MosaicDirector,
    stop_event: asyncio.Event,
    *,
    tick_seconds: float = MOSAIC_TICK_SECONDS,
) -> None:
    """Tick the director until the manager stops, then end its encoder.

    Args:
        director: The director to drive.
        stop_event: The serve loop's shutdown signal.
        tick_seconds: Cadence of the ticks.

    Returns:
        None, once ``stop_event`` is set.

    Raises:
        OSError: ffmpeg is not installed, or a text file cannot be
            written.
        ValueError: The tiles cannot be laid out.
    """
    try:
        while not stop_event.is_set():
            director.tick()
            await asyncio.sleep(tick_seconds)
    finally:
        director.stop()


__all__ = [
    "MOSAIC_DIR",
    "MOSAIC_EXIT_LIMIT",
    "MOSAIC_TICK_SECONDS",
    "MosaicDirector",
    "MosaicStatusDict",
    "decode_mosaic_status",
    "direct_mosaic",
    "encode_mosaic_status",
    "log",
]
