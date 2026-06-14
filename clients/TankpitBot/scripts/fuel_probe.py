"""Run the live fuel action probe harness."""

from __future__ import annotations

from platform_core.logging import get_logger

from scripts import _test_hooks as script_hooks
from tankpit_bot import _test_hooks
from tankpit_bot.action_lab import FuelProbeSessionDict, format_fuel_probe_summary, run_fuel_probe
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


def _format_saved_path(output_path: str) -> str:
    """Return the saved-path log line payload."""
    return f"Saved to: {output_path}"


def main() -> int:
    """Run the fuel probe entrypoint."""
    from dotenv import load_dotenv

    load_dotenv()
    script_hooks.setup_rich_logging(level="INFO")
    configure_probe_runtime_logging("fuel")

    if _test_hooks.sync_playwright is None:
        _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()

    argv = _test_hooks.get_argv()
    target_pickups = _parse_optional_int_arg(argv, "--target-pickups")
    max_attempts = _parse_optional_int_arg(argv, "--max-attempts")
    initial_sync_timeout_ms = _parse_optional_int_arg(argv, "--initial-sync-timeout-ms")

    target_url = _test_hooks.get_env("TANKPIT_URL") or "https://tankpit.com/play"
    output_path = _test_hooks.get_env("TANKPIT_FUEL_PROBE_OUTPUT") or "fuel_probe.json"
    headless = _parse_bool_env(_test_hooks.get_env("TANKPIT_HEADLESS"))
    prefer_account = _parse_bool_env(_test_hooks.get_env("TANKPIT_PREFER_ACCOUNT"))
    env_target_pickups = _test_hooks.get_env("TANKPIT_FUEL_PROBE_TARGET_PICKUPS")
    env_max_attempts = _test_hooks.get_env("TANKPIT_FUEL_PROBE_MAX_ATTEMPTS")
    env_initial_sync_timeout_ms = int(
        _test_hooks.get_env("TANKPIT_FUEL_PROBE_INITIAL_SYNC_TIMEOUT_MS") or "10000"
    )
    map_sync_timeout_ms = int(
        _test_hooks.get_env("TANKPIT_FUEL_PROBE_MAP_SYNC_TIMEOUT_MS") or "3000"
    )
    teleport_timeout_ms = int(
        _test_hooks.get_env("TANKPIT_FUEL_PROBE_TELEPORT_TIMEOUT_MS") or "10000"
    )
    radar_timeout_ms = int(_test_hooks.get_env("TANKPIT_FUEL_PROBE_RADAR_TIMEOUT_MS") or "3000")
    pickup_timeout_ms = int(_test_hooks.get_env("TANKPIT_FUEL_PROBE_PICKUP_TIMEOUT_MS") or "3000")
    settle_delay_ms = int(_test_hooks.get_env("TANKPIT_FUEL_PROBE_SETTLE_MS") or "500")

    session: FuelProbeSessionDict = run_fuel_probe(
        target_url,
        output_path,
        headless=headless,
        prefer_account=prefer_account,
        target_pickups=(
            target_pickups if target_pickups is not None else int(env_target_pickups or "3")
        ),
        max_attempts=max_attempts if max_attempts is not None else int(env_max_attempts or "9"),
        initial_sync_timeout_ms=(
            initial_sync_timeout_ms
            if initial_sync_timeout_ms is not None
            else env_initial_sync_timeout_ms
        ),
        map_sync_timeout_ms=map_sync_timeout_ms,
        teleport_timeout_ms=teleport_timeout_ms,
        radar_timeout_ms=radar_timeout_ms,
        pickup_timeout_ms=pickup_timeout_ms,
        settle_delay_ms=settle_delay_ms,
    )
    log.info(format_fuel_probe_summary(session))
    log.info(_format_saved_path(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_fuel_probe"]
