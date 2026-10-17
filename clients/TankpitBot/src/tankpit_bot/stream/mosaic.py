"""The fleet's four-up picture: up to four bots' screens in one stream.

Every streamed bot already records its own Xvfb display and sound sink
into its own HLS directory (:mod:`tankpit_bot.stream.capture`). A TV
plays one stream, so this module builds the command for ONE more
ffmpeg, run by the fleet manager, that records up to four of those
displays and sinks at once and lays them out as a 2x2 grid:

* each tile is one bot's display, scaled to fit and padded, with one
  label over its top margin naming the bot and its room, read from a
  text file the fleet rewrites when the tile changes (``drawtext``'s
  ``reload``);
* a position with no bot is a plain waiting tile whose label says so,
  so the grid is always four tiles and the stream never changes shape;
* the sound is every live tile's game audio mixed together, limited so
  four explosions at once do not clip; with no live tile it is
  silence, so the stream always carries an audio track.

What a bot is doing and why is NOT drawn here. The bot writes it into
the game's own preset-chat panel, in the game's font, inside its own
browser (board task c896a291), so it is already in the display this
records, as part of the picture. A second caption drawn over the tile
would say the same thing twice in a different hand.

The HLS flags differ from the per-bot encoder's in three ways, all for
the same reason. The set of bots changes while a viewer watches, and
ffmpeg cannot swap an input, so the fleet restarts this encoder on
every change. ``append_list`` continues the existing playlist and its
segment numbers, ``discont_start`` marks the seam so a player resets
its timestamps there, and ``omit_endlist`` keeps a stopped encoder from
telling every player the stream is over.
"""

from __future__ import annotations

import re
from pathlib import Path

from platform_core.json_utils import JSONObject, require_int
from typing_extensions import TypedDict

from tankpit_bot.stream.audio import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    AUDIO_SINK_NAME,
    pulse_server_address,
)
from tankpit_bot.stream.capture import (
    AUDIO_BITRATE_KBPS,
    FFMPEG_PROGRAM,
    HLS_LIST_SEGMENTS,
    HLS_PLAYLIST_FILENAME,
    HLS_SEGMENT_TEMPLATE,
    INPUT_QUEUE_PACKETS,
)

MOSAIC_POSITIONS = 4
"""Tiles in the grid. Four is what one TV shows legibly at once."""

MOSAIC_WIDTH = 1920
"""Output width in pixels: 1080p, which the TV's H.264 level 4.1
decodes; a 2x2 of the bots' native 1280x768 would exceed it."""

MOSAIC_HEIGHT = 1080
"""Output height in pixels."""

TILE_WIDTH = MOSAIC_WIDTH // 2
"""One tile's width in pixels."""

TILE_HEIGHT = MOSAIC_HEIGHT // 2
"""One tile's height in pixels."""

TILE_MARGIN = 30
"""Pixels kept clear on each side of a tile's picture, so the four
games read as four and not as one torn screen."""

MOSAIC_FPS = 30
"""Frames per second sampled from every display and written out."""

MOSAIC_SEGMENT_SECONDS = 2
"""HLS segment length, and the keyframe cadence, as the per-bot
encoder's: every segment opens decodable."""

MOSAIC_BITRATE_KBPS = 4000
"""Target video bitrate. Four tiles of mostly static map at 1080p sit
well under this; the headroom is for four battles at once."""

MOSAIC_LIMIT = 0.9
"""Peak level the mixed sound is limited to, as a linear fraction of
full scale. Four games summed without normalising keeps each as loud
as it is alone; the limiter is what stops their coincident peaks
clipping."""

MOSAIC_BACKGROUND = "0x0b0f14"
"""The colour of a waiting tile and of every tile's padding."""

LABEL_FONT = Path("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf")
"""A tile label's font, from the image's ``fonts-freefont-ttf`` package."""

LABEL_FONT_SIZE = 24
"""Point size of a tile's label."""

FILTER_UNSAFE = re.compile(r"[:,;'\[\]\\=]")
"""Characters with a meaning inside an ffmpeg filter graph. A path
carrying one would be parsed as graph syntax rather than as a file
name, so a path built into the graph is refused if it has one."""


class MosaicTileDict(TypedDict):
    """One live tile: which grid position shows which bot's display.

    Attributes:
        position: Grid position, 0 to 3, reading order (top left, top
            right, bottom left, bottom right).
        display: The bot's X display number, which is also the key of
            its sound server's socket
            (:func:`~tankpit_bot.stream.audio.pulse_server_address`).
    """

    position: int
    display: int


def encode_mosaic_tile(tile: MosaicTileDict) -> JSONObject:
    """Encode :class:`MosaicTileDict` to a JSON-serializable dict.

    Args:
        tile: The tile.

    Returns:
        JSON object carrying both fields.
    """
    return {"position": tile["position"], "display": tile["display"]}


def decode_mosaic_tile(data: JSONObject) -> MosaicTileDict:
    """Validate and decode a :class:`MosaicTileDict` from JSON.

    Args:
        data: JSON object to validate.

    Returns:
        The tile.

    Raises:
        JSONTypeError: A field is missing or has the wrong type.
        ValueError: The position is off the grid or the display is
            negative.
    """
    tile = MosaicTileDict(
        position=require_int(data, "position"), display=require_int(data, "display")
    )
    _validate_tiles([tile])
    return tile


def tile_label_path(text_dir: Path, position: int) -> Path:
    """Return the file a position's label is drawn from.

    Args:
        text_dir: Directory holding every tile's label file.
        position: Grid position.

    Returns:
        The label file path.
    """
    return text_dir / f"tile{position}-label.txt"


def _graph_path(path: Path) -> str:
    """Return a path as it may appear inside a filter graph.

    Args:
        path: An absolute file path.

    Returns:
        The path in POSIX form.

    Raises:
        ValueError: The path carries a character the filter graph
            would parse as syntax.
    """
    text = path.as_posix()
    if FILTER_UNSAFE.search(text):
        raise ValueError(f"path {text!r} carries a filter-graph character; it cannot be drawn")
    return text


def _label_filter(text_dir: Path, position: int) -> str:
    """Build a tile's label: one ``drawtext`` over its top margin.

    ``expansion=none`` because the label carries instance names, and a
    percent sign in one must print as itself rather than expand.

    Args:
        text_dir: Directory holding every tile's label file.
        position: Grid position.

    Returns:
        The filter.
    """
    label = _graph_path(tile_label_path(text_dir, position))
    font = _graph_path(LABEL_FONT)
    return (
        f"drawtext=fontfile={font}:textfile={label}:reload=1:expansion=none"
        f":fontcolor=white:fontsize={LABEL_FONT_SIZE}:box=1:boxcolor=black@0.55"
        f":boxborderw=8:x={TILE_MARGIN + 10}:y=10"
    )


def _validate_tiles(tiles: list[MosaicTileDict]) -> None:
    """Refuse a tile list the grid cannot lay out.

    Args:
        tiles: The live tiles.

    Raises:
        ValueError: A position is outside the grid or used twice, or a
            display number is negative.
    """
    seen: set[int] = set()
    for tile in tiles:
        if not 0 <= tile["position"] < MOSAIC_POSITIONS:
            raise ValueError(
                f"position {tile['position']} is outside the {MOSAIC_POSITIONS}-tile grid"
            )
        if tile["position"] in seen:
            raise ValueError(f"position {tile['position']} is given two displays")
        if tile["display"] < 0:
            raise ValueError(f"display must be non-negative, got {tile['display']}")
        seen.add(tile["position"])


def _input_args(tile: MosaicTileDict) -> list[str]:
    """Build the two live inputs one tile records: its display and its sound.

    Args:
        tile: The live tile.

    Returns:
        The input options, in the order ffmpeg numbers them.
    """
    return [
        "-thread_queue_size",
        str(INPUT_QUEUE_PACKETS),
        "-f",
        "x11grab",
        "-draw_mouse",
        "0",
        "-framerate",
        str(MOSAIC_FPS),
        "-i",
        f":{tile['display']}",
        "-thread_queue_size",
        str(INPUT_QUEUE_PACKETS),
        "-f",
        "pulse",
        "-server",
        pulse_server_address(tile["display"]),
        "-sample_rate",
        str(AUDIO_SAMPLE_RATE),
        "-channels",
        str(AUDIO_CHANNELS),
        "-i",
        f"{AUDIO_SINK_NAME}.monitor",
    ]


def mosaic_filter_graph(tiles: list[MosaicTileDict], text_dir: Path) -> str:
    """Build the filter graph laying the tiles out and mixing their sound.

    Live tiles are numbered by their order in ``tiles``: tile ``k``'s
    display is input ``2k`` and its sound input ``2k + 1``. Every live
    input is restarted at time zero (``setpts``/``asetpts``) because
    the live devices stamp wall-clock times while a waiting tile's
    generated picture starts at zero, and the grid joins frames by
    time. ``realtime`` paces the grid, which matters when every tile is
    waiting: generated pictures would otherwise encode as fast as the
    CPU allows.

    Args:
        tiles: The live tiles, at most one per position.
        text_dir: Directory holding every tile's label file.

    Returns:
        The graph, with outputs labelled ``[v]`` and ``[a]``.

    Raises:
        ValueError: The tiles cannot be laid out, or a text path cannot
            be drawn.
    """
    _validate_tiles(tiles)
    by_position = {tile["position"]: index for index, tile in enumerate(tiles)}
    picture_width = TILE_WIDTH - 2 * TILE_MARGIN
    chains: list[str] = []
    for position in range(MOSAIC_POSITIONS):
        text = _label_filter(text_dir, position)
        index = by_position.get(position)
        if index is None:
            source = f"color=c={MOSAIC_BACKGROUND}:s={TILE_WIDTH}x{TILE_HEIGHT}:r={MOSAIC_FPS}"
        else:
            source = (
                f"[{2 * index}:v]setpts=PTS-STARTPTS,"
                f"scale={picture_width}:{TILE_HEIGHT}:force_original_aspect_ratio=decrease,"
                f"pad={TILE_WIDTH}:{TILE_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color={MOSAIC_BACKGROUND}"
            )
        chains.append(f"{source},{text}[v{position}]")
    stacked = "".join(f"[v{position}]" for position in range(MOSAIC_POSITIONS))
    chains.append(
        f"{stacked}xstack=inputs={MOSAIC_POSITIONS}:layout=0_0|w0_0|0_h0|w0_h0,realtime[v]"
    )
    if not tiles:
        chains.append(f"anullsrc=r={AUDIO_SAMPLE_RATE}:cl=stereo[a]")
    else:
        for index in range(len(tiles)):
            chains.append(f"[{2 * index + 1}:a]asetpts=PTS-STARTPTS[a{index}]")
        mixed = "".join(f"[a{index}]" for index in range(len(tiles)))
        chains.append(
            f"{mixed}amix=inputs={len(tiles)}:normalize=0,alimiter=limit={MOSAIC_LIMIT}[a]"
        )
    return ";".join(chains)


def mosaic_ffmpeg_command(tiles: list[MosaicTileDict], text_dir: Path, hls_dir: Path) -> list[str]:
    """Build the ffmpeg argv recording the tiles into one HLS stream.

    Args:
        tiles: The live tiles, at most one per position; empty for an
            all-waiting grid.
        text_dir: Directory holding every tile's label file.
        hls_dir: Directory the playlist and segments land in.

    Returns:
        Full argv, program first.

    Raises:
        ValueError: The tiles cannot be laid out, or a text path cannot
            be drawn.
    """
    graph = mosaic_filter_graph(tiles, text_dir)
    inputs: list[str] = []
    for tile in tiles:
        inputs.extend(_input_args(tile))
    keyframe_interval = MOSAIC_FPS * MOSAIC_SEGMENT_SECONDS
    return [
        FFMPEG_PROGRAM,
        "-loglevel",
        "error",
        "-nostdin",
        *inputs,
        "-filter_complex",
        graph,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-profile:v",
        "high",
        "-level:v",
        "4.1",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(MOSAIC_FPS),
        "-g",
        str(keyframe_interval),
        "-keyint_min",
        str(keyframe_interval),
        "-sc_threshold",
        "0",
        "-b:v",
        f"{MOSAIC_BITRATE_KBPS}k",
        "-maxrate",
        f"{MOSAIC_BITRATE_KBPS * 3 // 2}k",
        "-bufsize",
        f"{MOSAIC_BITRATE_KBPS * 3}k",
        "-c:a",
        "aac",
        "-b:a",
        f"{AUDIO_BITRATE_KBPS}k",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-ac",
        str(AUDIO_CHANNELS),
        "-f",
        "hls",
        "-hls_time",
        str(MOSAIC_SEGMENT_SECONDS),
        "-hls_list_size",
        str(HLS_LIST_SEGMENTS),
        "-hls_flags",
        "delete_segments+independent_segments+temp_file+program_date_time"
        "+append_list+discont_start+omit_endlist",
        "-hls_segment_filename",
        str(hls_dir / HLS_SEGMENT_TEMPLATE),
        str(hls_dir / HLS_PLAYLIST_FILENAME),
    ]


__all__ = [
    "FILTER_UNSAFE",
    "LABEL_FONT",
    "LABEL_FONT_SIZE",
    "MOSAIC_BACKGROUND",
    "MOSAIC_BITRATE_KBPS",
    "MOSAIC_FPS",
    "MOSAIC_HEIGHT",
    "MOSAIC_LIMIT",
    "MOSAIC_POSITIONS",
    "MOSAIC_SEGMENT_SECONDS",
    "MOSAIC_WIDTH",
    "TILE_HEIGHT",
    "TILE_MARGIN",
    "TILE_WIDTH",
    "MosaicTileDict",
    "decode_mosaic_tile",
    "encode_mosaic_tile",
    "mosaic_ffmpeg_command",
    "mosaic_filter_graph",
    "tile_label_path",
]
