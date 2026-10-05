"""The public demo, live: a real bot that plays with sound and says what it is doing.

Board task 46934cd6. Its criteria are about what a visitor to
austinwagner.org/tankpit gets: game audio in the video (A1), a caption from
the bot's own state (A2), and both working end to end with a real bot (A4).
Its review asked for checks that "measure the live service rather than file
contents", because no bot was running when it looked. This case is that
measurement. It is EXECUTION-ONLY (``tests/_host.py``): the fleet project
``clients/TankpitBot-execution`` runs it, never ``make check``.

It does what a visitor does. If no demo bot is playing it presses the public
spawn button, which starts one bounded Practice bot that ends itself after
fifteen minutes (``DEMO_SESSION_SECONDS``); then it waits for that bot's row
in ``/demo/fleet`` to carry a caption, reads the bot's HLS playlist, and
hands one served segment to ``ffprobe``. The case passes only when a caption
says what the tank is doing and why, the playlist stamps its segments with
wall-clock times (what the page aligns captions to), and the segment holds
an AAC audio stream beside its video.
"""

from __future__ import annotations

import subprocess
import time
from http.client import HTTPSConnection
from pathlib import Path
from typing import Final

import pytest
from platform_core.json_utils import (
    JSONObject,
    load_json_bytes,
    load_json_str,
    narrow_json_to_dict,
    require_bool,
    require_list,
    require_str,
)

pytestmark = pytest.mark.host_live_demo

#: The demo service the public page calls (``Dashboards/tankpit/index.html``
#: sets ``API = "https://tankpit.austinwagner.org"``).
API_HOST: Final[str] = "tankpit.austinwagner.org"

#: What a video route answers while a new bot's encoder warms up
#: (``tankpit_bot.service.video_files``).
WARMING: Final[int] = 503

#: How long a freshly spawned bot may take to log in, play and decide.
CAPTION_DEADLINE_SECONDS: Final[float] = 300.0

#: Pause between reads of the fleet while waiting, about the page's own poll.
POLL_SECONDS: Final[float] = 5.0

#: Who is asking. The edge in front of the demo answers urllib's default,
#: ``Python-urllib/3.11``, with 403 Forbidden (measured 2026-10-05), so the
#: case names itself the way a reader of the access log would want it named.
USER_AGENT: Final[str] = "TankpitBot-execution/1 (live demo check; board task 46934cd6)"

#: Bound on any one HTTP request to the demo.
REQUEST_TIMEOUT_SECONDS: Final[float] = 30.0


def _fetch(path: str, *, method: str = "GET") -> tuple[int, bytes]:
    """Fetch one demo route, whatever status it answers with.

    :mod:`http.client` rather than ``urllib.request.urlopen``, for the reason
    ``tankpit_bot.service.probe`` gives (``urlopen``'s result is ``Any`` under
    strict mypy), and because a status is something this case reads: a
    playlist answers 503 while a new bot's encoder warms up.

    Args:
        path: The route, starting with ``/demo/``.
        method: ``GET``, or ``POST`` for the spawn button.

    Returns:
        The status and the body.
    """
    connection = HTTPSConnection(API_HOST, timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        connection.request(method, path, headers={"User-Agent": USER_AGENT})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def _request(path: str, *, method: str = "GET", expected: int = 200) -> bytes:
    """Fetch one demo route that must answer with one status.

    Args:
        path: The route, starting with ``/demo/``.
        method: ``GET``, or ``POST`` for the spawn button.
        expected: The status the route must answer with.

    Returns:
        The body.

    Raises:
        AssertionError: When the status differs, naming the route, the
            status and the body the demo gave.
    """
    status, body = _fetch(path, method=method)
    if status != expected:
        raise AssertionError(f"{method} {path} answered {status}, not {expected}: {body!r}")
    return body


def _warm_request(path: str) -> bytes:
    """Fetch a video file, waiting out the 503 a warming encoder answers.

    Args:
        path: A ``/demo/video/`` route.

    Returns:
        The body, once the route answers 200.

    Raises:
        AssertionError: When the route answers anything but 200 or 503, or
            still answers 503 at the deadline.
    """
    deadline = time.monotonic() + CAPTION_DEADLINE_SECONDS
    status, body = _fetch(path)
    while status == WARMING and time.monotonic() < deadline:
        time.sleep(POLL_SECONDS)
        status, body = _fetch(path)
    if status != 200:
        raise AssertionError(f"GET {path} answered {status}: {body!r}")
    return body


def _fleet() -> JSONObject:
    """Read the demo's whole public state.

    Returns:
        The ``/demo/fleet`` object.
    """
    return narrow_json_to_dict(load_json_bytes(_request("/demo/fleet")))


def _bots(fleet: JSONObject) -> list[JSONObject]:
    """List the demo bots a fleet read reports.

    Args:
        fleet: A ``/demo/fleet`` object.

    Returns:
        Each bot row.
    """
    return [narrow_json_to_dict(bot) for bot in require_list(fleet, "bots")]


def _playing_slot() -> str:
    """Name a demo bot that is playing, spawning one when none is.

    Returns:
        The slot of a live demo bot.
    """
    alive = [require_str(bot, "slot") for bot in _bots(_fleet()) if require_bool(bot, "alive")]
    if alive:
        return alive[0]
    spawned = narrow_json_to_dict(
        load_json_bytes(_request("/demo/spawn", method="POST", expected=201))
    )
    return require_str(spawned, "slot")


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
    deadline = time.monotonic() + CAPTION_DEADLINE_SECONDS
    row: JSONObject = {}
    while time.monotonic() < deadline:
        rows = [bot for bot in _bots(_fleet()) if require_str(bot, "slot") == slot]
        row = rows[0] if rows else {}
        captions = require_list(row, "captions") if rows else []
        if captions:
            return narrow_json_to_dict(captions[0])
        time.sleep(POLL_SECONDS)
    raise AssertionError(
        f"{slot} carried no caption within {CAPTION_DEADLINE_SECONDS:.0f} s; last row {row}"
    )


def _audio_codecs(segment: Path) -> list[str]:
    """Read the codec of every audio stream in a media file.

    Args:
        segment: The file to probe.

    Returns:
        Each audio stream's codec name, in stream order. Read from the JSON
        ``streams`` list: ffprobe's flat formats also print an MPEG-TS
        program's copy of each stream, so a one-stream segment reads as two.
    """
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_name",
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
    return [require_str(narrow_json_to_dict(stream), "codec_name") for stream in streams]


def test_a_live_demo_bot_streams_game_audio_and_captions_its_own_state(
    tmp_path: Path,
) -> None:
    """A real demo bot on austinwagner.org is captioned and its video has sound."""
    slot = _playing_slot()

    caption = _first_caption(slot)
    assert require_str(caption, "doing").strip()
    assert require_str(caption, "why").strip()

    playlist = _warm_request(f"/demo/video/{slot}/index.m3u8").decode("utf-8")
    assert "#EXT-X-PROGRAM-DATE-TIME:" in playlist
    segments = [line for line in playlist.splitlines() if line and not line.startswith("#")]
    assert segments, playlist
    segment = tmp_path / segments[-1]
    segment.write_bytes(_warm_request(f"/demo/video/{slot}/{segments[-1]}"))

    assert _audio_codecs(segment) == ["aac"]
