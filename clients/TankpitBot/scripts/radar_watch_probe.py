"""Run the live radar-watch probe harness."""

from __future__ import annotations

from platform_core.logging import get_logger

from scripts import _test_hooks as script_hooks
from tankpit_bot import _test_hooks
from tankpit_bot.action_lab.radar_watch import (
    format_radar_watch_summary,
    run_radar_watch_probe,
)
from tankpit_bot.runtime_logging import configure_probe_runtime_logging

log = get_logger(__name__)


def _parse_bool_env(value: str | None) -> bool:
    """Parse a boolean-like environment variable."""
    return value is not None and value.lower() in ("true", "1", "yes")


def _format_saved_path(output_path: str) -> str:
    """Return the saved-path log line payload."""
    return f"Saved to: {output_path}"


def main() -> int:
    """Run the radar-watch probe entrypoint."""
    from dotenv import load_dotenv

    load_dotenv()
    script_hooks.setup_rich_logging(level="INFO")
    configure_probe_runtime_logging("radar_watch")

    if _test_hooks.sync_playwright is None:
        _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()

    target_url = _test_hooks.get_env("TANKPIT_URL") or "https://tankpit.com/play"
    output_path = _test_hooks.get_env("TANKPIT_RADAR_WATCH_OUTPUT") or "radar_watch_probe.json"
    headless = _parse_bool_env(_test_hooks.get_env("TANKPIT_HEADLESS"))
    duration_ms = int(_test_hooks.get_env("TANKPIT_RADAR_WATCH_DURATION_MS") or "1800000")
    scan_interval_ms = int(_test_hooks.get_env("TANKPIT_RADAR_WATCH_SCAN_INTERVAL_MS") or "15000")
    map_poll_interval_ms = int(
        _test_hooks.get_env("TANKPIT_RADAR_WATCH_MAP_POLL_INTERVAL_MS") or "30000"
    )
    initial_sync_timeout_ms = int(
        _test_hooks.get_env("TANKPIT_RADAR_WATCH_INITIAL_SYNC_TIMEOUT_MS") or "10000"
    )

    session = run_radar_watch_probe(
        target_url,
        output_path,
        headless=headless,
        duration_ms=duration_ms,
        scan_interval_ms=scan_interval_ms,
        map_poll_interval_ms=map_poll_interval_ms,
        initial_sync_timeout_ms=initial_sync_timeout_ms,
    )
    log.info(format_radar_watch_summary(session))
    log.info(_format_saved_path(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "run_radar_watch_probe",
]
