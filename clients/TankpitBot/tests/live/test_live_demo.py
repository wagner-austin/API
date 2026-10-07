"""The public demo, live: a real bot that plays with sound and says what it is doing.

Board task 46934cd6. Its criteria are about what a visitor to
austinwagner.org/tankpit gets: game audio in the video, in sync with it (A1),
a caption from the bot's own state (A2), and both working end to end with a
real bot (A4). Its review asked for checks that "measure the live service
rather than file contents", because no bot was running when it looked. This
case measures what the service serves; ``test_live_page.py`` measures what
the page shows. Both are EXECUTION-ONLY (``tests/_host.py``): the fleet
project ``clients/TankpitBot-execution`` runs them, never ``make check``.

It does what a visitor does. If no demo bot is playing it presses the public
spawn button (``_demo.playing_slot``); then it waits for that bot's row in
``/demo/fleet`` to carry a caption, reads the bot's HLS playlist, and hands
the newest served segments to ``ffmpeg`` and ``ffprobe``. The case passes
only when a caption says what the tank is doing and why, the playlist stamps
its segments with wall-clock times (what the page aligns captions to), the
segments carry sound louder than the silent sink (:data:`SILENCE_DB`), and
the newest holds one AAC audio stream beside its video, starting within
:data:`SYNC_BOUND_SECONDS` of it.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Final, TypedDict

import pytest
from platform_core.json_utils import (
    JSONObject,
    load_json_str,
    narrow_json_to_dict,
    require_list,
    require_str,
)

from tests.live import _demo

pytestmark = pytest.mark.host_live_demo

#: How far apart a served segment's first audio and first video timestamps
#: may be. One ffmpeg encodes both from one clock, so they start at the
#: segment cut give or take an AAC frame and the video's frame grid: five
#: live segments read 50 to 66 ms on 2026-10-07 (seg00023-27 of demo-1). An
#: audio track recorded from its own clock, or muxed from a stale buffer,
#: starts hundreds of milliseconds to seconds away.
SYNC_BOUND_SECONDS: Final[float] = 0.1

#: How many of the newest served segments the case listens to (ten seconds).
LISTENED_SEGMENTS: Final[int] = 5

#: The loudness above which a segment carries sound. The bot's PulseAudio
#: sink reads -91.0 dB when the game is silent; the game's own effects read
#: -10.0 to -9.6 dB at their peak and -25.0 to -23.1 dB on average (demo-1,
#: seg00108-112, 2026-10-07). One two-second segment can fall silent between
#: effects (seg00027 read -91.0), so the case asks it of the loudest of
#: :data:`LISTENED_SEGMENTS`.
SILENCE_DB: Final[float] = -60.0

#: ffmpeg's volumedetect line for a file's peak sample.
MAX_VOLUME: Final[re.Pattern[str]] = re.compile(r"max_volume: (-?[0-9.]+) dB")


class SegmentStream(TypedDict):
    """One stream of a media file, as ffprobe reports it.

    Attributes:
        codec_type: ``audio`` or ``video``.
        codec_name: The codec, e.g. ``aac`` or ``h264``.
        start_time: The stream's first timestamp, in seconds.
    """

    codec_type: str
    codec_name: str
    start_time: float


def _decode_stream(stream: JSONObject) -> SegmentStream:
    """Decode one entry of ffprobe's JSON ``streams`` list.

    Args:
        stream: The entry. ffprobe writes ``start_time`` as a decimal string.

    Returns:
        The stream.
    """
    return SegmentStream(
        codec_type=require_str(stream, "codec_type"),
        codec_name=require_str(stream, "codec_name"),
        start_time=float(require_str(stream, "start_time")),
    )


def _streams(segment: Path) -> list[SegmentStream]:
    """Read every stream of a media file.

    Args:
        segment: The file to probe.

    Returns:
        Each stream, in stream order. Read from the JSON ``streams`` list:
        ffprobe's flat formats also print an MPEG-TS program's copy of each
        stream, so a two-stream segment reads as four.
    """
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,start_time",
            "-of",
            "json",
            str(segment),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    streams = require_list(narrow_json_to_dict(load_json_str(probe.stdout)), "streams")
    return [_decode_stream(narrow_json_to_dict(stream)) for stream in streams]


def _peak_db(segment: Path) -> float:
    """Measure the loudest audio sample in a media file.

    Args:
        segment: The file to measure.

    Returns:
        The peak, in dB below full scale, as ffmpeg's volumedetect filter
        reports it on stderr.

    Raises:
        AssertionError: When ffmpeg reports no peak, naming what it wrote.
    """
    measured = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(segment),
            "-vn",
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    peak = MAX_VOLUME.search(measured.stderr)
    if peak is None:
        raise AssertionError(f"ffmpeg reported no max_volume for {segment}: {measured.stderr}")
    return float(peak.group(1))


def _first_caption(slot: str) -> JSONObject:
    """Wait for a demo bot's first caption.

    Args:
        slot: The bot's slot.

    Returns:
        The oldest caption the bot's row carries.

    Raises:
        AssertionError: When the deadline passes with the bot captionless or
            gone, naming the last row read.
    """
    deadline = time.monotonic() + _demo.CAPTION_DEADLINE_SECONDS
    row = _demo.bot_row(slot)
    while time.monotonic() < deadline:
        captions = [] if row is None else require_list(row, "captions")
        if captions:
            return narrow_json_to_dict(captions[0])
        time.sleep(_demo.POLL_SECONDS)
        row = _demo.bot_row(slot)
    raise AssertionError(
        f"{slot} carried no caption within {_demo.CAPTION_DEADLINE_SECONDS:.0f} s; last row {row}"
    )


def test_a_live_demo_bot_streams_game_audio_and_captions_its_own_state(
    tmp_path: Path,
) -> None:
    """A real demo bot on austinwagner.org is captioned and its video has synced sound."""
    slot = _demo.playing_slot()

    caption = _first_caption(slot)
    assert require_str(caption, "doing").strip()
    assert require_str(caption, "why").strip()

    playlist = _demo.warm_request(f"/demo/video/{slot}/index.m3u8").decode("utf-8")
    assert "#EXT-X-PROGRAM-DATE-TIME:" in playlist
    names = [line for line in playlist.splitlines() if line and not line.startswith("#")]
    assert names, playlist
    segments = [tmp_path / name for name in names[-LISTENED_SEGMENTS:]]
    for segment in segments:
        segment.write_bytes(_demo.warm_request(f"/demo/video/{slot}/{segment.name}"))

    peaks = {segment.name: _peak_db(segment) for segment in segments}
    assert max(peaks.values()) > SILENCE_DB, peaks

    streams = _streams(segments[-1])
    audio = [stream for stream in streams if stream["codec_type"] == "audio"]
    video = [stream for stream in streams if stream["codec_type"] == "video"]
    assert [stream["codec_name"] for stream in audio] == ["aac"], streams
    assert len(video) == 1, streams
    gap = abs(audio[0]["start_time"] - video[0]["start_time"])
    assert gap < SYNC_BOUND_SECONDS, streams
