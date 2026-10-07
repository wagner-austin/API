"""The public demo's routes, as the live cases reach them over the internet.

Board task 46934cd6. Two execution-only cases watch the demo a visitor to
austinwagner.org/tankpit watches: ``test_live_demo.py`` reads what the
service serves (a caption, a playlist, a segment) and ``test_live_page.py``
reads what the page shows in a browser. Both need a demo bot playing and
both read ``/demo/fleet``, so the routes live here once.
"""

from __future__ import annotations

import time
from http.client import HTTPSConnection
from typing import Final

from platform_core.json_utils import (
    JSONObject,
    load_json_bytes,
    narrow_json_to_dict,
    require_bool,
    require_list,
    require_str,
)

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


def fetch(path: str, *, method: str = "GET") -> tuple[int, bytes]:
    """Fetch one demo route, whatever status it answers with.

    :mod:`http.client` rather than ``urllib.request.urlopen``, for the reason
    ``tankpit_bot.service.probe`` gives (``urlopen``'s result is ``Any`` under
    strict mypy), and because a status is something the cases read: a
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


def request(path: str, *, method: str = "GET", expected: int = 200) -> bytes:
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
    status, body = fetch(path, method=method)
    if status != expected:
        raise AssertionError(f"{method} {path} answered {status}, not {expected}: {body!r}")
    return body


def warm_request(path: str) -> bytes:
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
    status, body = fetch(path)
    while status == WARMING and time.monotonic() < deadline:
        time.sleep(POLL_SECONDS)
        status, body = fetch(path)
    if status != 200:
        raise AssertionError(f"GET {path} answered {status}: {body!r}")
    return body


def bot_rows() -> list[JSONObject]:
    """Read every demo bot row the public fleet route reports.

    Returns:
        Each row of ``/demo/fleet``'s ``bots``.
    """
    fleet = narrow_json_to_dict(load_json_bytes(request("/demo/fleet")))
    return [narrow_json_to_dict(bot) for bot in require_list(fleet, "bots")]


def bot_row(slot: str) -> JSONObject | None:
    """Read one demo bot's row.

    Args:
        slot: The bot's slot.

    Returns:
        The row, or None when the fleet no longer lists the slot.
    """
    rows = [bot for bot in bot_rows() if require_str(bot, "slot") == slot]
    return rows[0] if rows else None


def playing_slot() -> str:
    """Name a demo bot that is playing, spawning one when none is.

    A spawned bot is one bounded Practice bot that ends itself after fifteen
    minutes (``DEMO_SESSION_SECONDS``), the same one the page's button starts.

    Returns:
        The slot of a live demo bot.
    """
    alive = [require_str(bot, "slot") for bot in bot_rows() if require_bool(bot, "alive")]
    if alive:
        return alive[0]
    spawned = narrow_json_to_dict(
        load_json_bytes(request("/demo/spawn", method="POST", expected=201))
    )
    return require_str(spawned, "slot")
