"""Run the live command queue probe harness."""

from __future__ import annotations

from platform_core.logging import get_logger

from scripts import _test_hooks as script_hooks
from tankpit_bot import _test_hooks
from tankpit_bot.action_lab.queue_probe import (
    format_queue_probe_summary,
    run_queue_probe,
)
from tankpit_bot.action_lab.queue_probe_types import QueueProbeSessionDict
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
    """Run the queue probe entrypoint.

    Returns:
        Exit code (0 for success).
    """
    from dotenv import load_dotenv

    load_dotenv()
    script_hooks.setup_rich_logging(level="INFO")
    configure_probe_runtime_logging("queue")

    if _test_hooks.sync_playwright is None:
        _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()

    argv = _test_hooks.get_argv()
    experiment_timeout_ms = _parse_optional_int_arg(argv, "--experiment-timeout-ms")
    initial_sync_timeout_ms = _parse_optional_int_arg(argv, "--initial-sync-timeout-ms")

    target_url = _test_hooks.get_env("TANKPIT_URL") or "https://tankpit.com/play"
    output_path = _test_hooks.get_env("TANKPIT_QUEUE_PROBE_OUTPUT") or "queue_probe.json"
    headless = _parse_bool_env(_test_hooks.get_env("TANKPIT_HEADLESS"))
    prefer_account = _parse_bool_env(_test_hooks.get_env("TANKPIT_PREFER_ACCOUNT"))
    env_initial_sync_timeout_ms = int(
        _test_hooks.get_env("TANKPIT_QUEUE_INITIAL_SYNC_TIMEOUT_MS") or "10000"
    )
    env_experiment_timeout_ms = int(
        _test_hooks.get_env("TANKPIT_QUEUE_EXPERIMENT_TIMEOUT_MS") or "5000"
    )

    session: QueueProbeSessionDict = run_queue_probe(
        target_url,
        output_path,
        headless=headless,
        prefer_account=prefer_account,
        initial_sync_timeout_ms=(
            initial_sync_timeout_ms
            if initial_sync_timeout_ms is not None
            else env_initial_sync_timeout_ms
        ),
        experiment_timeout_ms=(
            experiment_timeout_ms
            if experiment_timeout_ms is not None
            else env_experiment_timeout_ms
        ),
    )
    log.info(format_queue_probe_summary(session))
    log.info(_format_saved_path(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_queue_probe"]
