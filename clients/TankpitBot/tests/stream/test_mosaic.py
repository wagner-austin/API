"""The four-up encoder's argv and filter graph, pinned piece by piece.

ffmpeg is not installed on the test hosts, so what is pinned here is
the command the fleet asks for: which inputs it records, how the graph
places and mixes them, and the HLS flags that keep one playlist going
across restarts. The command itself was run against four live bots in
the fleet container on 2026-10-17 (board task 5067379e) before these
pins were written: 1920x1080 H.264 High 4.1 with AAC stereo at real
time, and a restart appending to the same playlist behind an
``EXT-X-DISCONTINUITY``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.stream.audio import pulse_server_address
from tankpit_bot.stream.capture import FFMPEG_PROGRAM, HLS_PLAYLIST_FILENAME, HLS_SEGMENT_TEMPLATE
from tankpit_bot.stream.mosaic import (
    LABEL_FONT,
    LABEL_FONT_SIZE,
    MOSAIC_BACKGROUND,
    MOSAIC_HEIGHT,
    MOSAIC_LIMIT,
    MOSAIC_POSITIONS,
    MOSAIC_WIDTH,
    TILE_HEIGHT,
    TILE_WIDTH,
    MosaicTileDict,
    decode_mosaic_tile,
    encode_mosaic_tile,
    mosaic_ffmpeg_command,
    mosaic_filter_graph,
    tile_label_path,
)

TEXT_DIR = Path("runs") / "fleet" / "mosaic" / "text"
HLS_DIR = Path("runs") / "fleet" / "mosaic" / "hls"


def _chains(graph: str) -> list[str]:
    """Split a filter graph into its chains.

    Args:
        graph: The graph.

    Returns:
        Each ``;``-separated chain, in order.
    """
    return graph.split(";")


def _option(argv: list[str], flag: str) -> str:
    """Return the value following the first occurrence of ``flag``.

    Args:
        argv: The command.
        flag: The option.

    Returns:
        Its value.
    """
    return argv[argv.index(flag) + 1]


class TestGrid:
    """The four positions, live or waiting, in reading order."""

    def test_the_grid_is_two_by_two_tiles_filling_1080p(self) -> None:
        """Four tiles of half the frame each make exactly the output frame."""
        assert MOSAIC_POSITIONS == 4
        assert (TILE_WIDTH * 2, TILE_HEIGHT * 2) == (MOSAIC_WIDTH, MOSAIC_HEIGHT) == (1920, 1080)

    def test_live_tiles_take_their_inputs_in_list_order_at_their_positions(self) -> None:
        """Tile k records input 2k; a free position is a generated waiting tile."""
        tiles = [
            MosaicTileDict(position=1, display=27101),
            MosaicTileDict(position=3, display=27102),
        ]
        chains = _chains(mosaic_filter_graph(tiles, TEXT_DIR))

        assert chains[0].startswith(f"color=c={MOSAIC_BACKGROUND}:s={TILE_WIDTH}x{TILE_HEIGHT}")
        assert chains[0].endswith("[v0]")
        assert chains[1].startswith("[0:v]setpts=PTS-STARTPTS,scale=")
        assert "force_original_aspect_ratio=decrease" in chains[1]
        assert chains[1].endswith("[v1]")
        assert chains[2].startswith("color=") and chains[2].endswith("[v2]")
        assert chains[3].startswith("[2:v]setpts=PTS-STARTPTS,") and chains[3].endswith("[v3]")
        assert chains[4] == (
            "[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0,realtime[v]"
        )

    def test_every_tile_draws_exactly_one_label_of_its_own(self) -> None:
        """One reloading label per position, and no caption over the picture.

        What a bot is doing is in the game's own panel inside the
        recorded display (board task c896a291); a second caption here
        would duplicate it.
        """
        chains = _chains(mosaic_filter_graph([], TEXT_DIR))
        for position in range(MOSAIC_POSITIONS):
            chain = chains[position]
            label = tile_label_path(TEXT_DIR, position).as_posix()
            assert chain.count("drawtext=") == 1
            assert (
                f"drawtext=fontfile={LABEL_FONT.as_posix()}:textfile={label}"
                f":reload=1:expansion=none:fontcolor=white:fontsize={LABEL_FONT_SIZE}"
            ) in chain

    def test_labels_are_named_by_position(self) -> None:
        """One file per position, distinct across positions."""
        labels = {tile_label_path(TEXT_DIR, position) for position in range(MOSAIC_POSITIONS)}
        assert len(labels) == MOSAIC_POSITIONS
        assert tile_label_path(TEXT_DIR, 2) == TEXT_DIR / "tile2-label.txt"


class TestSound:
    """The mix of every live tile, or silence."""

    def test_live_sound_is_mixed_unnormalised_then_limited(self) -> None:
        """Each live sound restarts at zero; all are summed and limited."""
        tiles = [
            MosaicTileDict(position=0, display=1),
            MosaicTileDict(position=1, display=2),
            MosaicTileDict(position=2, display=3),
        ]
        chains = _chains(mosaic_filter_graph(tiles, TEXT_DIR))

        assert chains[5:] == [
            "[1:a]asetpts=PTS-STARTPTS[a0]",
            "[3:a]asetpts=PTS-STARTPTS[a1]",
            "[5:a]asetpts=PTS-STARTPTS[a2]",
            f"[a0][a1][a2]amix=inputs=3:normalize=0,alimiter=limit={MOSAIC_LIMIT}[a]",
        ]

    def test_an_all_waiting_grid_still_carries_a_silent_track(self) -> None:
        """No live tile: the audio output is generated silence."""
        chains = _chains(mosaic_filter_graph([], TEXT_DIR))
        assert chains[5:] == ["anullsrc=r=48000:cl=stereo[a]"]


class TestRefusals:
    """Tile lists and paths the graph cannot express."""

    @pytest.mark.parametrize("position", [-1, 4])
    def test_a_position_off_the_grid_is_refused(self, position: int) -> None:
        """Positions run 0 to 3."""
        with pytest.raises(ValueError, match="outside the 4-tile grid"):
            mosaic_filter_graph([MosaicTileDict(position=position, display=1)], TEXT_DIR)

    def test_a_position_given_twice_is_refused(self) -> None:
        """Two displays cannot share a tile."""
        tiles = [MosaicTileDict(position=2, display=1), MosaicTileDict(position=2, display=3)]
        with pytest.raises(ValueError, match="position 2 is given two displays"):
            mosaic_filter_graph(tiles, TEXT_DIR)

    def test_a_negative_display_is_refused(self) -> None:
        """X display numbers are non-negative."""
        with pytest.raises(ValueError, match="display must be non-negative, got -5"):
            mosaic_filter_graph([MosaicTileDict(position=0, display=-5)], TEXT_DIR)

    @pytest.mark.parametrize("bad", ["C:/runs", "runs,x", "runs;x", "runs[x]", "runs'x", "a=b"])
    def test_a_text_path_carrying_graph_syntax_is_refused(self, bad: str) -> None:
        """A path the graph would parse as syntax never reaches ffmpeg."""
        with pytest.raises(ValueError, match="filter-graph character"):
            mosaic_filter_graph([], Path(bad))


class TestCommand:
    """The whole argv: inputs, graph, encoder and HLS output."""

    def test_each_live_tile_records_its_display_and_its_sound_server(self) -> None:
        """Two live inputs per tile, in tile order, pointed at the bot's own servers."""
        tiles = [MosaicTileDict(position=2, display=27104)]
        argv = mosaic_ffmpeg_command(tiles, TEXT_DIR, HLS_DIR)

        assert argv[0] == FFMPEG_PROGRAM
        inputs = [argv[index + 1] for index, word in enumerate(argv) if word == "-i"]
        assert inputs == [":27104", "game.monitor"]
        assert _option(argv, "-server") == pulse_server_address(27104)
        assert _option(argv, "-draw_mouse") == "0"

    def test_a_display_queues_one_second_of_frames_and_its_sound_1024_packets(self) -> None:
        """Each input's queue precedes it: 30 raw frames for the display, 1024 for the sound.

        A raw frame is megabytes, so the deep sound queue on a display let
        a lagging encoder hold gigabytes (board task 73d4789e).
        """
        tiles = [
            MosaicTileDict(position=0, display=27101),
            MosaicTileDict(position=1, display=27102),
        ]
        argv = mosaic_ffmpeg_command(tiles, TEXT_DIR, HLS_DIR)

        queues = [
            argv[index + 1] for index, word in enumerate(argv) if word == "-thread_queue_size"
        ]
        formats = [argv[index + 1] for index, word in enumerate(argv) if word == "-f"][:4]
        assert queues == ["30", "1024", "30", "1024"]
        assert formats == ["x11grab", "pulse", "x11grab", "pulse"]

    def test_an_all_waiting_grid_records_nothing(self) -> None:
        """With no live tile there are no device inputs at all."""
        argv = mosaic_ffmpeg_command([], TEXT_DIR, HLS_DIR)
        assert "-i" not in argv
        assert _option(argv, "-filter_complex") == mosaic_filter_graph([], TEXT_DIR)

    def test_the_output_is_one_appending_hls_playlist(self) -> None:
        """The playlist continues across restarts and never declares an end."""
        argv = mosaic_ffmpeg_command([], TEXT_DIR, HLS_DIR)

        flags = _option(argv, "-hls_flags").split("+")
        assert {"append_list", "discont_start", "omit_endlist", "temp_file"} <= set(flags)
        assert _option(argv, "-hls_segment_filename") == str(HLS_DIR / HLS_SEGMENT_TEMPLATE)
        assert argv[-1] == str(HLS_DIR / HLS_PLAYLIST_FILENAME)
        assert _option(argv, "-map") == "[v]"
        assert argv[argv.index("-map") + 2 : argv.index("-map") + 4] == ["-map", "[a]"]

    def test_the_video_is_what_the_tv_decodes(self) -> None:
        """H.264 High at level 4.1, 4:2:0, a keyframe every segment."""
        argv = mosaic_ffmpeg_command([], TEXT_DIR, HLS_DIR)

        assert _option(argv, "-c:v") == "libx264"
        assert _option(argv, "-profile:v") == "high"
        assert _option(argv, "-level:v") == "4.1"
        assert _option(argv, "-pix_fmt") == "yuv420p"
        assert _option(argv, "-g") == _option(argv, "-keyint_min") == "60"
        assert _option(argv, "-hls_time") == "2"


class TestTileCodec:
    """A tile round-trips through JSON, and a bad one is refused."""

    def test_a_tile_round_trips(self) -> None:
        """Encode then decode is the identity."""
        tile = MosaicTileDict(position=3, display=27104)
        assert encode_mosaic_tile(tile) == {"position": 3, "display": 27104}
        assert decode_mosaic_tile(encode_mosaic_tile(tile)) == tile

    def test_a_tile_off_the_grid_is_refused_on_decode(self) -> None:
        """Decoding validates as the graph does."""
        with pytest.raises(ValueError, match="outside the 4-tile grid"):
            decode_mosaic_tile({"position": 9, "display": 1})

    def test_a_tile_missing_a_field_is_refused_on_decode(self) -> None:
        """Both fields are required."""
        with pytest.raises(JSONTypeError):
            decode_mosaic_tile({"position": 1})
