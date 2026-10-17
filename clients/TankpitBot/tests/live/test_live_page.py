"""The public page, live: what a visitor to austinwagner.org/tankpit sees and hears.

Board task 46934cd6, whose review asked for checks that "measure the live
service rather than file contents". ``test_live_demo.py`` measures what the
demo service serves; this case opens the page itself in the installed Edge
(``_visitor``) with a real demo bot playing, and holds the page to the
task's criteria as a visitor meets them:

- A3: the page explains TankPit and the bot, and says the bots play the
  practice room only.
- A1: the bot's tile starts with its sound off (the button reads "Sound off",
  the video is muted); one key press on the button turns it on, the video
  stays playing unmuted, and the browser goes on decoding the stream's audio.
- The game status is recorded into the video; there is no second caption
  under the tile. The real panel renderer is exercised in browser tests.
- A4: all of it on the published page, with a real bot.

The sound button is pressed with a real key press, not a scripted
``click()``: a key press is the user gesture a browser asks before it plays
sound, so the case meets the same autoplay rules a visitor does.

It is EXECUTION-ONLY (``tests/_host.py``): the fleet project
``clients/TankpitBot-execution`` runs it, never ``make check``.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Final, TypedDict

import pytest
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    narrow_json_to_dict,
    narrow_json_to_str,
    require_bool,
    require_int,
    require_str,
)

from tankpit_bot._test_hooks import PageProtocol
from tests.live import _demo, _visitor

pytestmark = pytest.mark.host_live_demo

#: The page a visitor opens.
PAGE_URL: Final[str] = "https://austinwagner.org/tankpit/"

#: The sentences of the page's "What you are watching" panel the case
#: requires, whitespace collapsed: what TankPit is, what the bot is, and the
#: practice-room-only promise (A3).
EXPLANATION: Final[tuple[str, ...]] = (
    "TankPit is a multiplayer tank game.",
    "This bot reads the game's messages and sends its own commands.",
    "These demos stay in the practice room and do not affect ranked play.",
)

#: How long the page may take to list the bot's tile after loading: one of
#: its fleet polls, with room for a slow first fetch.
TILE_DEADLINE_MS: Final[float] = 60_000.0

#: How long the case listens after turning the sound on. The stream's
#: segments are two seconds long, so this spans five of them.
LISTEN_MS: Final[float] = 10_000.0


class TileState(TypedDict):
    """What one bot's tile shows, read off the page.

    Attributes:
        sound: The sound button's label, ``Sound off`` or ``Sound on``.
        pressed: The button's ``aria-pressed``, ``false`` or ``true``.
        muted: Whether the video element is muted.
        paused: Whether the video element is paused.
        external_caption: Whether the obsolete caption below the video exists.
        audio_bytes: Audio the browser has decoded from the stream so far.
    """

    sound: str
    pressed: str
    muted: bool
    paused: bool
    external_caption: bool
    audio_bytes: int


def _decode_tile(raw: JSONObject) -> TileState:
    """Decode the object :func:`_tile_expression` evaluates to.

    Args:
        raw: The object.

    Returns:
        The tile's state.
    """
    return TileState(
        sound=require_str(raw, "sound"),
        pressed=require_str(raw, "pressed"),
        muted=require_bool(raw, "muted"),
        paused=require_bool(raw, "paused"),
        external_caption=require_bool(raw, "external_caption"),
        audio_bytes=require_int(raw, "audio_bytes"),
    )


def _tile_query(slot: str) -> str:
    """Write the script expression that finds a bot's tile.

    Args:
        slot: The bot's slot, which the tile's caption names in bold.

    Returns:
        An expression for the tile's ``li`` element, or undefined.
    """
    return (
        '[...document.querySelectorAll("li.tile")]'
        f'.find((tile) => tile.querySelector(".cap b").textContent === {dump_json_str(slot)})'
    )


def _tile_expression(slot: str) -> str:
    """Write the script expression that reads a bot's tile.

    Args:
        slot: The bot's slot.

    Returns:
        An expression for the tile's state as a plain object.
        ``webkitAudioDecodedByteCount`` is Chromium's count of the audio it
        has decoded for the element, which a muted or soundless stream
        leaves where it was.
    """
    return (
        "(() => {"
        f" const tile = {_tile_query(slot)};"
        ' const video = tile.querySelector("video");'
        ' const sound = tile.querySelector("button.sound");'
        " return {"
        " sound: sound.textContent,"
        ' pressed: sound.getAttribute("aria-pressed"),'
        " muted: video.muted,"
        " paused: video.paused,"
        ' external_caption: tile.querySelector(".say") !== null,'
        " audio_bytes: video.webkitAudioDecodedByteCount,"
        " };"
        "})()"
    )


def _read_tile(page: PageProtocol, slot: str) -> TileState:
    """Read a bot's tile off the page.

    Args:
        page: The page.
        slot: The bot's slot.

    Returns:
        The tile's state.
    """
    return _decode_tile(narrow_json_to_dict(page.evaluate(_tile_expression(slot))))


@pytest.fixture()
def visitor_page() -> Generator[PageProtocol, None, None]:
    """Open a fresh page in the installed Edge, headless.

    Yields:
        The page, its browser closed after the case.
    """
    with _visitor.visitor_playwright()() as playwright:
        browser = playwright.chromium.launch(headless=True, channel=_visitor.EDGE_CHANNEL)
        try:
            yield browser.new_context().new_page()
        finally:
            browser.close()


def test_the_live_page_explains_captions_and_plays_sound_on_request(
    visitor_page: PageProtocol,
) -> None:
    """A visitor's page explains the game, captions a real bot, and plays its sound."""
    slot = _demo.playing_slot()
    visitor_page.goto(PAGE_URL, wait_until="load")

    about = narrow_json_to_str(
        visitor_page.evaluate('document.getElementById("about").textContent.replace(/\\s+/g, " ")')
    )
    for sentence in EXPLANATION:
        assert sentence in about, about

    visitor_page.wait_for_function(f"{_tile_query(slot)} !== undefined", timeout=TILE_DEADLINE_MS)
    first = _read_tile(visitor_page, slot)
    assert (first["sound"], first["pressed"], first["muted"]) == ("Sound off", "false", True)
    # The visitor scrolls down to the tile, which the explanation puts below
    # the fold: the browser pauses a muted autoplaying video while it is out
    # of view, so a tile nobody scrolled to stops playing.
    visitor_page.evaluate(f"{_tile_query(slot)}.scrollIntoView()")

    visitor_page.wait_for_function(
        f'{_tile_query(slot)}.querySelector("video").currentTime > 0',
        timeout=TILE_DEADLINE_MS,
    )
    captioned = _read_tile(visitor_page, slot)
    assert not captioned["external_caption"], captioned
    assert not captioned["paused"], captioned
    assert captioned["muted"], captioned

    focused = visitor_page.evaluate(
        f'(() => {{ const button = {_tile_query(slot)}.querySelector("button.sound");'
        " button.focus(); return document.activeElement === button; })()"
    )
    assert focused is True
    visitor_page.keyboard.press("Enter")
    turned_on = _read_tile(visitor_page, slot)
    assert (turned_on["sound"], turned_on["pressed"], turned_on["muted"]) == (
        "Sound on",
        "true",
        False,
    )

    visitor_page.wait_for_timeout(LISTEN_MS)
    listened = _read_tile(visitor_page, slot)
    assert not listened["paused"], listened
    assert not listened["muted"], listened
    assert listened["audio_bytes"] > turned_on["audio_bytes"], (turned_on, listened)
