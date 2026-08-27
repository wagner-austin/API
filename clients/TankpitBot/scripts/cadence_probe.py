"""Run the live fire-cadence probe."""

from __future__ import annotations

from platform_core.logging import get_logger

from scripts import _test_hooks as script_hooks
from tankpit_bot import _test_hooks
from tankpit_bot.action_lab.cadence_probe import (
    format_cadence_probe_summary,
    run_cadence_probe,
)
from tankpit_bot.action_lab.cadence_probe_types import CadenceProbeSessionDict
from tankpit_bot.runtime_artifacts import make_run_stamp
from tankpit_bot.runtime_logging import configure_probe_runtime_logging

log = get_logger(__name__)


def _parse_bool_env(value: str | None) -> bool:
    """Parse a boolean-like environment variable.

    Args:
        value: Raw environment value or ``None``.

    Returns:
        Parsed boolean.
    """
    return value is not None and value.lower() in ("true", "1", "yes")


def _parse_spacings_arg(argv: list[str]) -> tuple[int, ...] | None:
    """Parse the optional ``--spacings`` CLI flag.

    Args:
        argv: Process argument vector.

    Returns:
        Spacings tuple, or ``None`` when the flag is absent.

    Raises:
        ValueError: When the flag has no value.
    """
    if "--spacings" not in argv:
        return None
    index = argv.index("--spacings")
    if index + 1 >= len(argv):
        raise ValueError("--spacings requires a comma-separated ms list")
    return tuple(int(part) for part in argv[index + 1].split(","))


def _parse_shots_arg(argv: list[str]) -> int | None:
    """Parse the optional ``--shots`` CLI flag.

    Args:
        argv: Process argument vector.

    Returns:
        Shots per burst, or ``None`` when the flag is absent.

    Raises:
        ValueError: When the flag has no value.
    """
    if "--shots" not in argv:
        return None
    index = argv.index("--shots")
    if index + 1 >= len(argv):
        raise ValueError("--shots requires an integer value")
    return int(argv[index + 1])


def main() -> int:
    """Run the cadence probe entrypoint.

    Returns:
        Process exit code.
    """
    from dotenv import load_dotenv

    load_dotenv()
    script_hooks.setup_rich_logging(level="INFO")
    stamp = make_run_stamp()
    configure_probe_runtime_logging("cadence", stamp)

    if _test_hooks.sync_playwright is None:
        _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()

    argv = _test_hooks.get_argv()
    spacings = _parse_spacings_arg(argv)
    shots = _parse_shots_arg(argv)

    target_url = _test_hooks.get_env("TANKPIT_URL") or "https://tankpit.com/play"
    output_path = _test_hooks.get_env("TANKPIT_CADENCE_PROBE_OUTPUT") or (
        f"runs/probe/cadence-{stamp}.json"
    )
    headless = _parse_bool_env(_test_hooks.get_env("TANKPIT_HEADLESS"))
    prefer_account = _parse_bool_env(_test_hooks.get_env("TANKPIT_PREFER_ACCOUNT"))

    session: CadenceProbeSessionDict = run_cadence_probe(
        target_url,
        output_path,
        headless=headless,
        prefer_account=prefer_account,
        spacings_ms=spacings or (2000, 1000, 500, 250),
        shots_per_burst=shots or 6,
    )
    log.info(format_cadence_probe_summary(session))
    log.info("Saved to: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "run_cadence_probe",
]
