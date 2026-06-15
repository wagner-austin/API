"""Probe the game server's command queue behavior.

Connects to the game, enters a room, and sends command pairs to measure
how the server processes queued commands. Reports timing between sent
commands and server responses.

Experiments:
1. Send shoot + pickup back-to-back — does the server process both?
2. Send two moves rapidly — does the server queue both?
3. Measure server-side shot cooldown from response timing.
"""

from __future__ import annotations

import sys

from platform_core.logging import get_logger

from scripts import _test_hooks as script_hooks
from tankpit_bot.runtime_logging import configure_probe_runtime_logging

log = get_logger(__name__)


def main() -> int:
    """Run the queue probe entrypoint.

    Returns:
        Exit code (0 for success).
    """
    from dotenv import load_dotenv

    load_dotenv()
    script_hooks.setup_rich_logging(level="INFO")
    configure_probe_runtime_logging("queue")

    sys.stdout.write("Queue probe: not yet implemented\n")
    sys.stdout.write("Use 'make analyze-timing' to analyze existing captures\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
