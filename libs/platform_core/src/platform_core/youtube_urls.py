"""Reading a video id out of a YouTube URL, without deciding what a failure means.

clubbot and transcript-api each carried this whole module -- the same host
set, the same eleven-character id pattern, the same watch/shorts/live/youtu.be
handling. The fork had drifted in the way that matters: transcript-api names
FOUR distinct failures (empty, unparseable, not YouTube, bad id) and clubbot
collapsed all four into one generic INVALID_INPUT, so a Discord user who typed
a Vimeo link and one who typed a truncated id got the same message.

WHY THIS RETURNS AN OUTCOME RATHER THAN RAISING. The parsing is shared; the
ERROR POLICY is not, and should not be. A service answering HTTP wants its own
traceable codes and a chat client wants a sentence a person can act on. A
shared function that raised would force one package's vocabulary onto the
other, which is what made the fork look reasonable in the first place. So this
says WHICH of the four things went wrong and lets each caller map that to its
own error.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Literal

from typing_extensions import TypedDict

YOUTUBE_HOSTS: frozenset[str] = frozenset(
    {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}
)
"""Hosts a YouTube video URL may carry, lowercased."""

_WATCH_HOSTS = frozenset({"youtube.com", "www.youtube.com", "m.youtube.com"})
"""The hosts that use /watch?v= and /shorts/ and /live/ rather than a bare path."""

_PATH_ID_PREFIXES = frozenset({"shorts", "live"})
"""Path segments that are followed by the video id."""

VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
"""A YouTube video id: exactly eleven URL-safe characters."""


class YouTubeUrlOutcome(TypedDict):
    """What reading a URL produced.

    Attributes:
        kind: ``ok`` when ``video_id`` is set; otherwise which of the four
            failures occurred, so a caller can raise its own error for each
            rather than flattening them into one.
        video_id: The eleven-character id, or None on any failure.
    """

    kind: Literal["ok", "empty", "unparseable", "not_youtube", "bad_video_id"]
    video_id: str | None


def _watch_query_id(parsed: urllib.parse.SplitResult) -> str | None:
    """Read the ``v`` parameter from a watch URL's query.

    Args:
        parsed: The split URL.

    Returns:
        The first non-empty ``v`` value, or None when there is none.
    """
    values = urllib.parse.parse_qs(parsed.query).get("v")
    if not values:
        return None
    first = values[0]
    return first if first else None


def _candidate_id(parsed: urllib.parse.SplitResult, host: str) -> str | None:
    """Pull the id-shaped part out of a URL that is known to be YouTube's.

    Args:
        parsed: The split URL.
        host: Its lowercased netloc.

    Returns:
        The candidate id, unvalidated, or None when the path holds none.
    """
    path = parsed.path.strip("/")
    if host in _WATCH_HOSTS:
        if path == "watch":
            return _watch_query_id(parsed)
        parts = path.split("/")
        if len(parts) >= 2 and parts[0] in _PATH_ID_PREFIXES:
            return parts[1]
        return None
    parts = path.split("/")
    return parts[0] if parts and parts[0] else None


def read_video_id(url: str) -> YouTubeUrlOutcome:
    """Read a video id from a YouTube URL of any accepted shape.

    Accepts ``watch?v=``, ``shorts/``, ``live/`` and ``youtu.be/``, with or
    without a scheme.

    Args:
        url: The URL as a person or an API caller supplied it.

    Returns:
        An outcome naming the id, or which of the four failures occurred.
        Never raises: what a failure MEANS is the caller's to decide, and the
        two packages that share this parsing answer to different audiences.
    """
    raw = url.strip()
    if not raw:
        return {"kind": "empty", "video_id": None}
    try:
        parsed = urllib.parse.urlsplit(raw if "://" in raw else f"https://{raw}")
    except ValueError:
        return {"kind": "unparseable", "video_id": None}

    host = parsed.netloc.lower()
    if host not in YOUTUBE_HOSTS:
        return {"kind": "not_youtube", "video_id": None}

    candidate = _candidate_id(parsed, host)
    if candidate is None or not VIDEO_ID_PATTERN.match(candidate):
        return {"kind": "bad_video_id", "video_id": None}
    return {"kind": "ok", "video_id": candidate}


def canonical_watch_url(video_id: str) -> str:
    """Render the canonical watch URL for a video id.

    Args:
        video_id: An eleven-character id, as returned by :func:`read_video_id`.

    Returns:
        The ``https://www.youtube.com/watch?v=<id>`` form.
    """
    return f"https://www.youtube.com/watch?v={video_id}"


__all__ = [
    "VIDEO_ID_PATTERN",
    "YOUTUBE_HOSTS",
    "YouTubeUrlOutcome",
    "canonical_watch_url",
    "read_video_id",
]
