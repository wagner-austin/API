"""Run the live container-density probe harness."""

from __future__ import annotations

from platform_core.logging import get_logger

from scripts import _test_hooks as script_hooks
from tankpit_bot import _test_hooks
from tankpit_bot.action_lab.density_probe import (
    format_density_probe_summary,
    run_density_probe,
)
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
    """Run the density-probe entrypoint."""
    from dotenv import load_dotenv

    load_dotenv()
    script_hooks.setup_rich_logging(level="INFO")
    stamp = make_run_stamp()
    configure_probe_runtime_logging("density", stamp)

    if _test_hooks.sync_playwright is None:
        _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()

    target_url = _test_hooks.get_env("TANKPIT_URL") or "https://tankpit.com/play"
    # Per-run archive (2026-07-25 incident rule): a fixed output path
    # let earlier runs overwrite their own capture evidence.
    output_path = (
        _test_hooks.get_env("TANKPIT_DENSITY_OUTPUT") or f"runs/probe/density-{stamp}.json"
    )
    headless = _parse_bool_env(_test_hooks.get_env("TANKPIT_HEADLESS"))
    max_extras = int(_test_hooks.get_env("TANKPIT_DENSITY_MAX_EXTRAS") or "12")
    initial_sync_timeout_ms = int(
        _test_hooks.get_env("TANKPIT_DENSITY_INITIAL_SYNC_TIMEOUT_MS") or "10000"
    )

    session = run_density_probe(
        target_url,
        output_path,
        headless=headless,
        max_extras=max_extras,
        initial_sync_timeout_ms=initial_sync_timeout_ms,
    )
    log.info(format_density_probe_summary(session))
    log.info(_format_saved_path(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "run_density_probe",
]
