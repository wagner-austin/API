"""Run the live enemy-directed teleport probe harness."""

from __future__ import annotations

from typing import Literal

from platform_core.logging import get_logger

from scripts import _test_hooks as script_hooks
from tankpit_bot import _test_hooks
from tankpit_bot.action_lab import (
    EnemyTeleportProbeSessionDict,
    format_enemy_teleport_probe_summary,
    run_enemy_teleport_probe,
)
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


def _parse_optional_strategy_arg(argv: list[str]) -> str | None:
    """Parse an optional enemy acquisition strategy CLI flag value."""
    if "--acquisition-strategy" not in argv:
        return None
    index = argv.index("--acquisition-strategy")
    if index + 1 >= len(argv):
        raise ValueError("--acquisition-strategy requires a value")
    return argv[index + 1]


def _parse_acquisition_strategy(value: str) -> Literal["map_open", "nearest_enemy"]:
    """Validate an enemy acquisition strategy value."""
    if value == "map_open":
        return "map_open"
    if value == "nearest_enemy":
        return "nearest_enemy"
    raise ValueError(f"unsupported acquisition strategy: {value}")


def _format_saved_path(output_path: str) -> str:
    """Return the saved-path log line payload."""
    return f"Saved to: {output_path}"


def main() -> int:
    """Run the enemy teleport probe entrypoint."""
    from dotenv import load_dotenv

    load_dotenv()
    script_hooks.setup_rich_logging(level="INFO")
    configure_probe_runtime_logging("enemy_teleport")
    configure_probe_runtime_logging("enemy_teleport")

    if _test_hooks.sync_playwright is None:
        _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()

    argv = _test_hooks.get_argv()
    max_attempts = _parse_optional_int_arg(argv, "--max-attempts")
    initial_sync_timeout_ms = _parse_optional_int_arg(argv, "--initial-sync-timeout-ms")
    acquisition_strategy_arg = _parse_optional_strategy_arg(argv)

    target_url = _test_hooks.get_env("TANKPIT_URL") or "https://tankpit.com/play"
    output_path = (
        _test_hooks.get_env("TANKPIT_ENEMY_TELEPORT_PROBE_OUTPUT") or "enemy_teleport_probe.json"
    )
    headless = _parse_bool_env(_test_hooks.get_env("TANKPIT_HEADLESS"))
    prefer_account = _parse_bool_env(_test_hooks.get_env("TANKPIT_PREFER_ACCOUNT"))
    acquisition_strategy = _parse_acquisition_strategy(
        acquisition_strategy_arg
        or _test_hooks.get_env("TANKPIT_ENEMY_TELEPORT_ACQUISITION_STRATEGY")
        or "map_open"
    )
    env_initial_sync_timeout_ms = int(
        _test_hooks.get_env("TANKPIT_ENEMY_TELEPORT_INITIAL_SYNC_TIMEOUT_MS") or "10000"
    )
    env_max_attempts = _test_hooks.get_env("TANKPIT_ENEMY_TELEPORT_MAX_ATTEMPTS")
    acquisition_timeout_ms = int(
        _test_hooks.get_env("TANKPIT_ENEMY_TELEPORT_ACQUISITION_TIMEOUT_MS") or "3000"
    )
    teleport_timeout_ms = int(_test_hooks.get_env("TANKPIT_ENEMY_TELEPORT_TIMEOUT_MS") or "10000")
    settle_delay_ms = int(_test_hooks.get_env("TANKPIT_ENEMY_TELEPORT_SETTLE_MS") or "500")

    session: EnemyTeleportProbeSessionDict = run_enemy_teleport_probe(
        target_url,
        output_path,
        headless=headless,
        prefer_account=prefer_account,
        acquisition_strategy=acquisition_strategy,
        max_attempts=max_attempts if max_attempts is not None else int(env_max_attempts or "3"),
        initial_sync_timeout_ms=(
            initial_sync_timeout_ms
            if initial_sync_timeout_ms is not None
            else env_initial_sync_timeout_ms
        ),
        acquisition_timeout_ms=acquisition_timeout_ms,
        teleport_timeout_ms=teleport_timeout_ms,
        settle_delay_ms=settle_delay_ms,
    )
    log.info(format_enemy_teleport_probe_summary(session))
    log.info(_format_saved_path(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "run_enemy_teleport_probe",
]
