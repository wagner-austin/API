"""Run the live enemy-tracking probe harness."""

from __future__ import annotations

from platform_core.logging import get_logger

from scripts import _test_hooks as script_hooks
from tankpit_bot import _test_hooks
from tankpit_bot.action_lab.enemy_tracking import run_enemy_tracking_probe
from tankpit_bot.action_lab.enemy_tracking_records import format_enemy_tracking_probe_summary
from tankpit_bot.action_lab.enemy_tracking_types import EnemyTrackingProbeSessionDict
from tankpit_bot.runtime_logging import configure_probe_runtime_logging

log = get_logger(__name__)


def _parse_bool_env(value: str | None) -> bool:
    """Parse a boolean-like environment variable."""
    return value is not None and value.lower() in ("true", "1", "yes")


def _parse_optional_int_arg(argv: list[str], flag: str) -> int | None:
    """Parse an optional integer CLI flag value."""
    if flag not in argv:
        return None
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise ValueError(f"{flag} requires an integer value")
    return int(argv[index + 1])


def main() -> int:
    """Run the enemy-tracking probe entrypoint."""
    from dotenv import load_dotenv

    load_dotenv()
    script_hooks.setup_rich_logging(level="INFO")
    configure_probe_runtime_logging("enemy_tracking")

    if _test_hooks.sync_playwright is None:
        _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()

    argv = _test_hooks.get_argv()
    sample_duration_ms = _parse_optional_int_arg(argv, "--sample-duration-ms")
    sample_interval_ms = _parse_optional_int_arg(argv, "--sample-interval-ms")

    target_url = _test_hooks.get_env("TANKPIT_URL") or "https://tankpit.com/play"
    output_path = (
        _test_hooks.get_env("TANKPIT_ENEMY_TRACKING_PROBE_OUTPUT") or "enemy_tracking_probe.json"
    )
    headless = _parse_bool_env(_test_hooks.get_env("TANKPIT_HEADLESS"))
    prefer_account = _parse_bool_env(_test_hooks.get_env("TANKPIT_PREFER_ACCOUNT"))

    env_duration_ms = int(
        _test_hooks.get_env("TANKPIT_ENEMY_TRACKING_SAMPLE_DURATION_MS") or "120000"
    )
    env_interval_ms = int(
        _test_hooks.get_env("TANKPIT_ENEMY_TRACKING_SAMPLE_INTERVAL_MS") or "1000"
    )

    session: EnemyTrackingProbeSessionDict = run_enemy_tracking_probe(
        target_url,
        output_path,
        headless=headless,
        prefer_account=prefer_account,
        sample_duration_ms=(
            sample_duration_ms if sample_duration_ms is not None else env_duration_ms
        ),
        sample_interval_ms=(
            sample_interval_ms if sample_interval_ms is not None else env_interval_ms
        ),
    )
    log.info(format_enemy_tracking_probe_summary(session))
    log.info("Saved to: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
