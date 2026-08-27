"""Run the live movement probe harness."""

from __future__ import annotations

from platform_core.logging import get_logger

from scripts import _test_hooks as script_hooks
from tankpit_bot import _test_hooks
from tankpit_bot.action_lab import (
    MovementProbeSessionDict,
    TeleportTargetDict,
    format_movement_probe_summary,
    parse_targets_arg,
    run_movement_probe,
)
from tankpit_bot.runtime_artifacts import make_run_stamp
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


def _parse_targets_cli(argv: list[str]) -> list[TeleportTargetDict] | None:
    """Parse optional explicit movement targets from CLI arguments."""
    if "--targets" not in argv:
        return None
    index = argv.index("--targets")
    if index + 1 >= len(argv):
        raise ValueError("--targets requires a value")
    return parse_targets_arg(argv[index + 1])


def _has_flag(argv: list[str], flag: str) -> bool:
    """Return whether the CLI flag is present."""
    return flag in argv


def _format_saved_path(output_path: str) -> str:
    """Return the saved-path log line payload."""
    return f"Saved to: {output_path}"


def main() -> int:
    """Run the movement probe entrypoint."""
    from dotenv import load_dotenv

    load_dotenv()
    script_hooks.setup_rich_logging(level="INFO")
    stamp = make_run_stamp()
    configure_probe_runtime_logging("movement", stamp)

    if _test_hooks.sync_playwright is None:
        _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()

    argv = _test_hooks.get_argv()
    explicit_targets = _parse_targets_cli(argv)
    max_targets = _parse_optional_int_arg(argv, "--max-targets")
    initial_sync_timeout_ms = _parse_optional_int_arg(argv, "--initial-sync-timeout-ms")
    move_timeout_ms = _parse_optional_int_arg(argv, "--move-timeout-ms")
    map_open_delay_ms = _parse_optional_int_arg(argv, "--map-open-delay-ms")
    queue_map_open_during_move = _has_flag(argv, "--queue-map-open")

    target_url = _test_hooks.get_env("TANKPIT_URL") or "https://tankpit.com/play"
    output_path = _test_hooks.get_env("TANKPIT_MOVEMENT_PROBE_OUTPUT") or (
        f"runs/probe/movement-{stamp}.json"
    )
    headless = _parse_bool_env(_test_hooks.get_env("TANKPIT_HEADLESS"))
    prefer_account = _parse_bool_env(_test_hooks.get_env("TANKPIT_PREFER_ACCOUNT"))
    env_max_targets = int(_test_hooks.get_env("TANKPIT_MOVEMENT_MAX_TARGETS") or "3")
    env_initial_sync_timeout_ms = int(
        _test_hooks.get_env("TANKPIT_MOVEMENT_INITIAL_SYNC_TIMEOUT_MS") or "10000"
    )
    env_move_timeout_ms = int(_test_hooks.get_env("TANKPIT_MOVEMENT_TIMEOUT_MS") or "5000")
    env_queue_map_open = _parse_bool_env(_test_hooks.get_env("TANKPIT_MOVEMENT_QUEUE_MAP_OPEN"))
    env_map_open_delay_ms = int(_test_hooks.get_env("TANKPIT_MOVEMENT_MAP_OPEN_DELAY_MS") or "0")
    settle_delay_ms = int(_test_hooks.get_env("TANKPIT_MOVEMENT_SETTLE_MS") or "500")

    session: MovementProbeSessionDict = run_movement_probe(
        target_url,
        output_path,
        headless=headless,
        prefer_account=prefer_account,
        explicit_targets=explicit_targets,
        max_targets=max_targets if max_targets is not None else env_max_targets,
        initial_sync_timeout_ms=(
            initial_sync_timeout_ms
            if initial_sync_timeout_ms is not None
            else env_initial_sync_timeout_ms
        ),
        move_timeout_ms=move_timeout_ms if move_timeout_ms is not None else env_move_timeout_ms,
        queue_map_open_during_move=queue_map_open_during_move or env_queue_map_open,
        map_open_delay_ms=(
            map_open_delay_ms if map_open_delay_ms is not None else env_map_open_delay_ms
        ),
        settle_delay_ms=settle_delay_ms,
    )
    log.info(format_movement_probe_summary(session))
    log.info(_format_saved_path(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_movement_probe"]
