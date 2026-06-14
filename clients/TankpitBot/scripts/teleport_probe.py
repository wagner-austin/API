"""Run the live teleport probe harness."""

from __future__ import annotations

from typing import Literal

from platform_core.logging import get_logger

from scripts import _test_hooks as script_hooks
from tankpit_bot import _test_hooks
from tankpit_bot.action_lab import (
    DEFAULT_TELEPORT_STRATEGY,
    TeleportProbeSessionDict,
    TeleportTargetDict,
    format_teleport_probe_summary,
    parse_targets_arg,
    run_teleport_probe,
)
from tankpit_bot.runtime_logging import configure_probe_runtime_logging

log = get_logger(__name__)


def _parse_bool_env(value: str | None) -> bool:
    """Parse a boolean-like environment variable.

    Args:
        value: Raw environment variable value.

    Returns:
        True for ``true``, ``1``, and ``yes``. False otherwise.
    """
    return value is not None and value.lower() in ("true", "1", "yes")


def _parse_optional_int_arg(argv: list[str], flag: str) -> int | None:
    """Parse an optional integer CLI flag value.

    Args:
        argv: Raw CLI argument vector.
        flag: Flag name to inspect.

    Returns:
        Parsed integer value, or None when the flag is absent.

    Raises:
        ValueError: If the flag is present without a value.
    """
    if flag not in argv:
        return None
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise ValueError(f"{flag} requires an integer value")
    return int(argv[index + 1])


def _parse_targets_cli(argv: list[str]) -> list[TeleportTargetDict] | None:
    """Parse optional explicit teleport targets from CLI arguments.

    Args:
        argv: Raw CLI argument vector.

    Returns:
        Parsed targets, or None when ``--targets`` is absent.

    Raises:
        ValueError: If the flag is present without a value or is malformed.
    """
    if "--targets" not in argv:
        return None
    index = argv.index("--targets")
    if index + 1 >= len(argv):
        raise ValueError("--targets requires a value")
    return parse_targets_arg(argv[index + 1])


def _format_saved_path(output_path: str) -> str:
    """Return the saved-path log line payload.

    Args:
        output_path: Output file path.

    Returns:
        Formatted saved-path string.
    """
    return f"Saved to: {output_path}"


def _parse_optional_strategy_arg(
    argv: list[str],
) -> str | None:
    """Parse an optional teleport strategy CLI flag value.

    Args:
        argv: Raw CLI argument vector.

    Returns:
        Strategy value, or None when the flag is absent.

    Raises:
        ValueError: If the flag is present without a value.
    """
    if "--teleport-strategy" not in argv:
        return None
    index = argv.index("--teleport-strategy")
    if index + 1 >= len(argv):
        raise ValueError("--teleport-strategy requires a value")
    return argv[index + 1]


def _parse_teleport_strategy(
    value: str,
) -> Literal["sync_before_teleport", "immediate_after_map_open"]:
    """Validate a teleport strategy value.

    Args:
        value: Raw strategy string.

    Returns:
        Validated teleport strategy literal.

    Raises:
        ValueError: If the strategy is unsupported.
    """
    if value == "sync_before_teleport":
        return "sync_before_teleport"
    if value == "immediate_after_map_open":
        return "immediate_after_map_open"
    raise ValueError(f"unsupported teleport strategy: {value}")


def main() -> int:
    """Run the teleport probe entrypoint.

    Returns:
        Process exit code.
    """
    from dotenv import load_dotenv

    load_dotenv()
    script_hooks.setup_rich_logging(level="INFO")
    configure_probe_runtime_logging("teleport")

    if _test_hooks.sync_playwright is None:
        _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()

    argv = _test_hooks.get_argv()
    explicit_targets = _parse_targets_cli(argv)
    box_step_x = _parse_optional_int_arg(argv, "--step-x")
    box_step_y = _parse_optional_int_arg(argv, "--step-y")
    max_targets = _parse_optional_int_arg(argv, "--max-targets")
    initial_sync_timeout_ms = _parse_optional_int_arg(argv, "--initial-sync-timeout-ms")
    teleport_strategy_arg = _parse_optional_strategy_arg(argv)

    target_url = _test_hooks.get_env("TANKPIT_URL") or "https://tankpit.com/play"
    output_path = _test_hooks.get_env("TANKPIT_TELEPORT_PROBE_OUTPUT") or "teleport_probe.json"
    headless = _parse_bool_env(_test_hooks.get_env("TANKPIT_HEADLESS"))
    prefer_account = _parse_bool_env(_test_hooks.get_env("TANKPIT_PREFER_ACCOUNT"))
    teleport_strategy = _parse_teleport_strategy(
        teleport_strategy_arg
        or _test_hooks.get_env("TANKPIT_TELEPORT_STRATEGY")
        or DEFAULT_TELEPORT_STRATEGY
    )
    env_initial_sync_timeout_ms = int(
        _test_hooks.get_env("TANKPIT_TELEPORT_INITIAL_SYNC_TIMEOUT_MS") or "10000"
    )
    env_max_targets = _test_hooks.get_env("TANKPIT_TELEPORT_MAX_TARGETS")
    map_sync_timeout_ms = int(_test_hooks.get_env("TANKPIT_TELEPORT_MAP_SYNC_TIMEOUT_MS") or "3000")
    teleport_timeout_ms = int(_test_hooks.get_env("TANKPIT_TELEPORT_TIMEOUT_MS") or "10000")
    settle_delay_ms = int(_test_hooks.get_env("TANKPIT_TELEPORT_SETTLE_MS") or "500")

    session: TeleportProbeSessionDict = run_teleport_probe(
        target_url,
        output_path,
        headless=headless,
        prefer_account=prefer_account,
        explicit_targets=explicit_targets,
        box_step_x=box_step_x if box_step_x is not None else 8,
        box_step_y=box_step_y if box_step_y is not None else 8,
        max_targets=max_targets if max_targets is not None else int(env_max_targets or "0") or None,
        teleport_strategy=teleport_strategy,
        initial_sync_timeout_ms=(
            initial_sync_timeout_ms
            if initial_sync_timeout_ms is not None
            else env_initial_sync_timeout_ms
        ),
        map_sync_timeout_ms=map_sync_timeout_ms,
        teleport_timeout_ms=teleport_timeout_ms,
        settle_delay_ms=settle_delay_ms,
    )
    log.info(format_teleport_probe_summary(session))
    log.info(_format_saved_path(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "run_teleport_probe",
]
