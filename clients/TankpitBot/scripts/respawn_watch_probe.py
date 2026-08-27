"""Run the live respawn-watch probe harness."""

from __future__ import annotations

from platform_core.logging import get_logger

from scripts import _test_hooks as script_hooks
from tankpit_bot import _test_hooks
from tankpit_bot.action_lab import format_enemy_teleport_probe_summary
from tankpit_bot.action_lab.respawn_watch import run_respawn_watch_probe
from tankpit_bot.runtime_artifacts import make_run_stamp
from tankpit_bot.runtime_logging import configure_probe_runtime_logging

log = get_logger(__name__)


def _parse_bool_env(value: str | None) -> bool:
    """Parse a boolean-like environment variable."""
    return value is not None and value.lower() in ("true", "1", "yes")


def _format_saved_path(output_path: str) -> str:
    """Return the saved-path log line payload."""
    return f"Saved to: {output_path}"


def main() -> int:
    """Run the respawn-watch probe entrypoint."""
    from dotenv import load_dotenv

    load_dotenv()
    script_hooks.setup_rich_logging(level="INFO")
    stamp = make_run_stamp()
    configure_probe_runtime_logging("respawn_watch", stamp)

    if _test_hooks.sync_playwright is None:
        _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()

    target_url = _test_hooks.get_env("TANKPIT_URL") or "https://tankpit.com/play"
    output_path = (
        _test_hooks.get_env("TANKPIT_RESPAWN_WATCH_PROBE_OUTPUT")
        or f"runs/probe/respawn-watch-{stamp}.json"
    )
    headless = _parse_bool_env(_test_hooks.get_env("TANKPIT_HEADLESS"))
    prefer_account = _parse_bool_env(_test_hooks.get_env("TANKPIT_PREFER_ACCOUNT"))
    max_attempts = int(_test_hooks.get_env("TANKPIT_RESPAWN_WATCH_MAX_ATTEMPTS") or "4")
    initial_sync_timeout_ms = int(
        _test_hooks.get_env("TANKPIT_RESPAWN_WATCH_INITIAL_SYNC_TIMEOUT_MS") or "10000"
    )
    acquisition_timeout_ms = int(
        _test_hooks.get_env("TANKPIT_RESPAWN_WATCH_ACQUISITION_TIMEOUT_MS") or "3000"
    )
    teleport_timeout_ms = int(
        _test_hooks.get_env("TANKPIT_RESPAWN_WATCH_TELEPORT_TIMEOUT_MS") or "10000"
    )
    engage_ms = int(_test_hooks.get_env("TANKPIT_RESPAWN_WATCH_ENGAGE_MS") or "30000")
    shot_interval_ms = int(_test_hooks.get_env("TANKPIT_RESPAWN_WATCH_SHOT_INTERVAL_MS") or "2000")
    poll_ms = int(_test_hooks.get_env("TANKPIT_RESPAWN_WATCH_POLL_MS") or "60000")
    poll_interval_ms = int(_test_hooks.get_env("TANKPIT_RESPAWN_WATCH_POLL_INTERVAL_MS") or "2000")

    session = run_respawn_watch_probe(
        target_url,
        output_path,
        headless=headless,
        prefer_account=prefer_account,
        max_attempts=max_attempts,
        initial_sync_timeout_ms=initial_sync_timeout_ms,
        acquisition_timeout_ms=acquisition_timeout_ms,
        teleport_timeout_ms=teleport_timeout_ms,
        engage_ms=engage_ms,
        shot_interval_ms=shot_interval_ms,
        poll_ms=poll_ms,
        poll_interval_ms=poll_interval_ms,
    )
    log.info(format_enemy_teleport_probe_summary(session))
    log.info(_format_saved_path(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "run_respawn_watch_probe",
]
