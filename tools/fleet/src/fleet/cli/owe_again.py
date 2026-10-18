"""CLI: owe a failed retire again, once the fault that failed it is fixed.

Usage:
    python -m fleet.cli.owe_again --config registry \\
        --run MCPs-fleet-mcp-sedona-1792355094 \\
        --because "8e655f640 skips a transcript holder that exits under the retire"

A retire the node answered and failed is recorded ``failed`` and no pass
sends it again (:mod:`fleet.core.retire_owed`), so its run directory stays
on the node. This records it ``owed`` once more, with what fixed the fault,
and the node's next collect pass retires it as it would any owed run. Run it
on the hub whose records hold the run, after the fix is rolled: a run owed
again under the code that failed it fails the same way.

IT OWES EXACTLY ONE RUN, BY NAME, and only one whose last line is
``failed``: an owed run is already sent, a retired one left nothing, and a
run the record does not hold was never leased by this hub's runners, so each
is refused with ``RUN_UNKNOWN`` rather than recorded.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Final

from platform_core import cli_args
from platform_core.logging import LogFormat, LogLevel, get_logger, setup_logging

from fleet.cli import _config
from fleet.core import retire_owed

_log = get_logger(__name__)

RUN_FLAG: Final = "--run"
BECAUSE_FLAG: Final = "--because"

_FLAGS: Final = (_config.CONFIG_FLAG, _config.RECORDS_FLAG, RUN_FLAG, BECAUSE_FLAG)


def main(argv: Sequence[str] | None = None) -> int:
    """Owe one failed retire again.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 once the run is recorded owed.

    Raises:
        ValueError: When a flag is unknown, repeated, or missing its value,
            or ``--because`` says nothing.
        AppError: ``RUN_UNKNOWN`` when the run's last line is not ``failed``.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    loaded = _config.load_workspace(parsed)
    run_id = cli_args.require_flag(parsed, RUN_FLAG)
    because = cli_args.require_flag(parsed, BECAUSE_FLAG).strip()
    if not because:
        raise ValueError(f"{BECAUSE_FLAG} must say what fixed the retire's fault")

    alias = retire_owed.owe_again(loaded.retires, run_id=run_id, because=because)

    _log.info("%s on %s is owed its retire again: %s", run_id, alias, because)
    return 0


def entrypoint() -> None:
    """Module entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level=LogLevel.INFO,
        format_mode=LogFormat.TEXT,
        service_name="fleet-owe-again",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = ["BECAUSE_FLAG", "RUN_FLAG", "entrypoint", "main"]


# Without this, `python -m fleet.cli.owe_again` imports the module, runs
# nothing and exits 0, which reads as a run owed that nothing will retire.
if __name__ == "__main__":
    entrypoint()
