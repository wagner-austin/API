"""Fight report CLI: human episodes and the per-event play-by-play.

``tankpit-fight [capture] [start_hms end_hms]`` — defaults to
``runs/bot/latest.capture_session.json``. Without a window it prints
the session's human episodes; with ``HH:MM:SS HH:MM:SS`` (local time,
matching the run log's timestamps) it also renders the chronological
fight rows for that window. Wired so a death autopsy is one command
instead of the four hand-decoding passes the 2026-08-03 nope fight
took.
"""

from __future__ import annotations

import datetime
from pathlib import Path

from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.logging import get_logger
from platform_core.rich_logging import setup_rich_logging

from tankpit_bot import _test_hooks
from tankpit_bot.types import CaptureSession, decode_capture_session
from tankpit_bot.validate.fight_timeline import (
    FightRowDict,
    HumanEpisodeDict,
    extract_human_episodes,
    render_fight_rows,
)
from tankpit_bot.validate.shadow_timeline import extract_shadow_timeline

log = get_logger(__name__)

_DEFAULT_CAPTURE = Path("runs/bot/latest.capture_session.json")


def _hms(timestamp_ms: int) -> str:
    """Render a capture epoch timestamp as local wall-clock time.

    Args:
        timestamp_ms: Capture epoch milliseconds.

    Returns:
        ``HH:MM:SS`` in local time — the run log's clock.
    """
    return datetime.datetime.fromtimestamp(timestamp_ms / 1000).strftime("%H:%M:%S")


def window_bounds_ms(day_ms: int, start_hms: str, end_hms: str) -> tuple[int, int]:
    """Resolve an ``HH:MM:SS`` window against a session's local day.

    Args:
        day_ms: Any capture timestamp inside the session (anchors the
            local calendar day).
        start_hms: Inclusive window start, ``HH:MM:SS`` local.
        end_hms: Inclusive window end, ``HH:MM:SS`` local.

    Returns:
        The window as capture epoch milliseconds.
    """
    anchor = datetime.datetime.fromtimestamp(day_ms / 1000)

    def _at(hms: str) -> int:
        hour, minute, second = (int(part) for part in hms.split(":"))
        moment = datetime.datetime.combine(anchor.date(), datetime.time(hour, minute, second))
        # Naive same-day arithmetic against the anchor: no mktime
        # round-trip (Windows raises EINVAL on some local conversions).
        return day_ms + int((moment - anchor).total_seconds() * 1000)

    return _at(start_hms), _at(end_hms)


def _render_episode(episode: HumanEpisodeDict) -> str:
    """Render one human episode line.

    Args:
        episode: The episode record.

    Returns:
        One report line.
    """
    window = (
        f"{_hms(episode['first_shot_ms'])}-{_hms(episode['last_shot_ms'])}"
        if episode["shots_by_human"] > 0
        else "(no shots)"
    )
    return (
        f"  {episode['name']} (id {episode['tank_id']}) {window}: "
        f"{episode['shots_by_human']} shots taken, "
        f"{episode['our_shots_in_window']} returned, "
        f"kills {episode['kills_of_human']}, deaths {episode['deaths_to_human']}, "
        f"max stationary streak {episode['max_stationary_streak']}"
    )


def _render_row(row: FightRowDict) -> str:
    """Render one play-by-play row.

    Args:
        row: The fight row.

    Returns:
        One report line.
    """
    return f"  {_hms(row['timestamp_ms'])}  {row['actor']:>12}  {row['description']}"


def render_fight_report(capture: CaptureSession, capture_path: Path, args: list[str]) -> str:
    """Build the full fight report text for one capture.

    Args:
        capture: The decoded capture session.
        capture_path: The capture's path (report header).
        args: CLI arguments after the program name — optionally the
            capture path plus an ``HH:MM:SS HH:MM:SS`` window.

    Returns:
        The rendered report: the human episodes, plus the window's
        play-by-play rows when a window was given.
    """
    timeline = extract_shadow_timeline(capture)
    episodes = extract_human_episodes(timeline)
    lines = [
        f"FIGHT REPORT {capture_path}",
        f"human episodes: {len(episodes)}",
    ]
    lines.extend(_render_episode(episode) for episode in episodes)
    if len(args) == 3:
        start_ms, end_ms = window_bounds_ms(capture["start_timestamp_ms"], args[1], args[2])
        rows = render_fight_rows(timeline, start_ms, end_ms)
        lines.append(f"play-by-play {args[1]}..{args[2]}: {len(rows)} rows")
        lines.extend(_render_row(row) for row in rows)
    return "\n".join(lines)


def main() -> int:
    """Entry point for the ``tankpit-fight`` CLI.

    Returns:
        Process exit code (``0`` on success). Errors propagate as
        exceptions.
    """
    setup_rich_logging(level="INFO")
    args = _test_hooks.get_argv()[1:]
    capture_path = Path(args[0]) if args else _DEFAULT_CAPTURE
    capture = decode_capture_session(
        narrow_json_to_dict(load_json_str(_test_hooks.read_text(capture_path)))
    )
    log.info("%s", render_fight_report(capture, capture_path, args))
    return 0


__all__ = [
    "main",
    "render_fight_report",
    "window_bounds_ms",
]
