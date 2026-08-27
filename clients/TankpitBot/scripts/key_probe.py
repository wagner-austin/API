"""Run the live keyboard probe harness."""

from __future__ import annotations

from platform_core.logging import get_logger

from scripts import _test_hooks as script_hooks
from tankpit_bot import _test_hooks
from tankpit_bot.action_lab.key_probe import (
    DEFAULT_KEYS,
    format_key_probe_summary,
    run_key_probe,
)
from tankpit_bot.runtime_artifacts import make_run_stamp
from tankpit_bot.runtime_logging import configure_probe_runtime_logging

log = get_logger(__name__)


def _parse_bool_env(value: str | None) -> bool:
    """Parse a boolean-like environment variable."""
    return value is not None and value.lower() in ("true", "1", "yes")


def _parse_keys_env(value: str | None) -> tuple[str, ...]:
    """Parse a comma-separated key list, defaulting to the safe set."""
    if value is None or not value.strip():
        return DEFAULT_KEYS
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _format_saved_path(output_path: str) -> str:
    """Return the saved-path log line payload."""
    return f"Saved to: {output_path}"


def main() -> int:
    """Run the keyboard probe entrypoint."""
    from dotenv import load_dotenv

    load_dotenv()
    script_hooks.setup_rich_logging(level="INFO")
    stamp = make_run_stamp()
    configure_probe_runtime_logging("key_probe", stamp)

    if _test_hooks.sync_playwright is None:
        _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()

    target_url = _test_hooks.get_env("TANKPIT_URL") or "https://tankpit.com/play"
    output_path = _test_hooks.get_env("TANKPIT_KEY_PROBE_OUTPUT") or f"runs/probe/key-{stamp}.json"
    headless = _parse_bool_env(_test_hooks.get_env("TANKPIT_HEADLESS"))
    prefer_account = _parse_bool_env(_test_hooks.get_env("TANKPIT_PREFER_ACCOUNT"))
    keys = _parse_keys_env(_test_hooks.get_env("TANKPIT_KEY_PROBE_KEYS"))
    initial_sync_timeout_ms = int(
        _test_hooks.get_env("TANKPIT_KEY_PROBE_INITIAL_SYNC_TIMEOUT_MS") or "10000"
    )
    inter_key_delay_ms = int(_test_hooks.get_env("TANKPIT_KEY_PROBE_DELAY_MS") or "1500")

    session = run_key_probe(
        target_url,
        output_path,
        headless=headless,
        prefer_account=prefer_account,
        keys=keys,
        initial_sync_timeout_ms=initial_sync_timeout_ms,
        inter_key_delay_ms=inter_key_delay_ms,
    )
    log.info(format_key_probe_summary(session))
    log.info(_format_saved_path(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "run_key_probe",
]
