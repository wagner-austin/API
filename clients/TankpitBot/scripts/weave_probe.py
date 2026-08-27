"""Run the live shoot+move weave probe."""

from __future__ import annotations

from platform_core.logging import get_logger

from scripts import _test_hooks as script_hooks
from tankpit_bot import _test_hooks
from tankpit_bot.action_lab.weave_probe import (
    format_weave_probe_summary,
    run_weave_probe,
)
from tankpit_bot.action_lab.weave_probe_types import WeaveProbeSessionDict
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


def _parse_int_arg(argv: list[str], flag: str) -> int | None:
    """Parse an optional integer CLI flag value.

    Args:
        argv: Process argument vector.
        flag: Flag name to look for.

    Returns:
        Integer value, or ``None`` when the flag is absent.

    Raises:
        ValueError: When the flag has no value.
    """
    if flag not in argv:
        return None
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise ValueError(f"{flag} requires an integer value")
    return int(argv[index + 1])


def main() -> int:
    """Run the weave probe entrypoint.

    Returns:
        Process exit code.
    """
    from dotenv import load_dotenv

    load_dotenv()
    script_hooks.setup_rich_logging(level="INFO")
    stamp = make_run_stamp()
    configure_probe_runtime_logging("weave", stamp)

    if _test_hooks.sync_playwright is None:
        _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()

    argv = _test_hooks.get_argv()
    beats = _parse_int_arg(argv, "--beats")
    bursts = _parse_int_arg(argv, "--bursts")

    target_url = _test_hooks.get_env("TANKPIT_URL") or "https://tankpit.com/play"
    output_path = _test_hooks.get_env("TANKPIT_WEAVE_PROBE_OUTPUT") or (
        f"runs/probe/weave-{stamp}.json"
    )
    headless = _parse_bool_env(_test_hooks.get_env("TANKPIT_HEADLESS"))
    prefer_account = _parse_bool_env(_test_hooks.get_env("TANKPIT_PREFER_ACCOUNT"))

    session: WeaveProbeSessionDict = run_weave_probe(
        target_url,
        output_path,
        headless=headless,
        prefer_account=prefer_account,
        beats_per_burst=beats or 8,
        burst_count=bursts or 1,
    )
    log.info(format_weave_probe_summary(session))
    log.info("Saved to: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "run_weave_probe",
]
