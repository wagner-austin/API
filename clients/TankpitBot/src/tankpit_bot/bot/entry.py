"""Bot CLI entry point and session argument parsing."""

from __future__ import annotations

import sys
from pathlib import Path

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.runtime_logging import configure_bot_runtime_logging

log = get_logger(__name__)

_USAGE = (
    "Usage: tankpit-bot [--seconds N]\n\n"
    "  --seconds N  Run for N seconds (0 or omit = run until\n"
    "               stopped). Defaults to TANKPIT_BOT_SESSION_SECONDS,\n"
    "               then 0.\n"
)


def resolve_session_seconds(argv: list[str], env_value: str | None) -> int:
    """Resolve the bounded session length from CLI args and environment.

    Args:
        argv: Process arguments without the program name.
        env_value: Raw ``TANKPIT_BOT_SESSION_SECONDS`` value, or None.

    Returns:
        Session length in seconds; zero means run until stopped.

    Raises:
        SystemExit: On ``--help``/``-h`` or unrecognized arguments.
        ValueError: If the seconds value is not an integer.
    """
    if argv and argv[0] in ("--help", "-h"):
        sys.stdout.write(_USAGE)
        raise SystemExit(0)
    if not argv:
        return int(env_value) if env_value is not None else 0
    if len(argv) == 2 and argv[0] == "--seconds":
        return int(argv[1])
    raise SystemExit(f"tankpit-bot: unrecognized arguments: {' '.join(argv)}\n\n{_USAGE}")


def resolve_session_kills(env_value: str | None) -> int:
    """Resolve the kill-target session bound from the environment.

    Args:
        env_value: Raw ``TANKPIT_BOT_SESSION_KILLS`` value, or None.

    Returns:
        Kill target; zero means no kill bound (time/stop-file only).

    Raises:
        ValueError: If the value is not an integer.
    """
    return int(env_value) if env_value is not None else 0


def main() -> None:
    """Entry point for tankpit-bot command."""
    from tankpit_bot.bot.base import Bot
    from tankpit_bot.bot.config import resolve_prefer_account, resolve_target_url
    from tankpit_bot.bot.tick_loop import (
        request_interrupt,
        reset_interrupt_flag,
    )
    from tankpit_bot.service import _test_hooks as service_hooks
    from tankpit_bot.sniffer.decoders import set_protocol_frame_logging

    service_hooks.load_dotenv()
    session_seconds = resolve_session_seconds(
        _test_hooks.get_argv()[1:],
        _test_hooks.get_env("TANKPIT_BOT_SESSION_SECONDS"),
    )
    session_kills = resolve_session_kills(_test_hooks.get_env("TANKPIT_BOT_SESSION_KILLS"))
    artifacts = configure_bot_runtime_logging()
    set_protocol_frame_logging(False)
    log.info("Bot latest log: %s", artifacts["latest_log_path"])
    log.info("Bot latest events: %s", artifacts["latest_events_path"])
    log.info("Bot latest capture: %s", artifacts["latest_capture_path"])
    stop_file_path = Path(artifacts["latest_log_path"]).parent / "STOP"
    _test_hooks.remove_file(stop_file_path)
    log.info(
        "Session bound: %s%s; stop file: %s",
        f"{session_seconds}s" if session_seconds > 0 else "until stopped",
        f" or {session_kills} kills" if session_kills > 0 else "",
        stop_file_path,
    )

    # Reset the interrupt flag so a re-run within the same process
    # (rare; test/probe paths) starts clean, then install SIGINT/SIGTERM
    # handlers. Ctrl+C and ``kill PID`` now request a graceful exit at
    # the next tick boundary, writing the scorecard + index row before
    # process exit. Without this, an interrupted run leaves
    # ``runs/bot/_index.tsv`` silent on that session, which made
    # "find runs that were Ctrl+C'd" impossible.
    reset_interrupt_flag()
    _test_hooks.install_signal_handlers(request_interrupt)

    if _test_hooks.sync_playwright is None:
        _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()

    bot = Bot(
        resolve_target_url(),
        headless=False,
        prefer_account=resolve_prefer_account(),
    )
    bot.run(
        session_seconds=session_seconds,
        session_kills=session_kills,
        stop_file_path=stop_file_path,
    )


__all__ = [
    "main",
    "resolve_session_seconds",
]
