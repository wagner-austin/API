"""A reused fleet slot starts clean: nothing of its predecessor reaches the next session.

An instance name is a SLOT: ``runs/bot/<instance>/`` outlives the
process that ran in it, and the next spawn under the same name reuses
it. Three things a predecessor can leave there would reach the new
session, so :func:`clear_reused_slot` removes them before the child
starts:

* the ``hls/`` stream files, which the video route would serve as live
  (operator observation 2026-09-05: "it opened the previous session …
  then it restarted the session"). Clearing makes the warmup honest:
  the playlist reads 503 warming until THIS session's stream exists;
* a ``STOP`` the predecessor never took, which would end the new
  session at its first tick;
* a ``CONTROL`` verb the predecessor never took, which would steer the
  new session and, while pending, refuse every verb sent to it.

Clearing happens at spawn, BEFORE the child exists, so it can never
remove a sentinel written for the new session.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.bot.control import control_file_path
from tankpit_bot.runtime_artifacts import bot_run_dir, bot_stop_file

log = get_logger(__name__)


def clear_reused_slot(instance: str) -> None:
    """Delete a predecessor session's stream files and untaken sentinels.

    Args:
        instance: The instance namespace about to be spawned.
    """
    run_dir = bot_run_dir(instance)
    hls = run_dir / "hls"
    stale = top_hooks.glob_paths(hls, "*")
    for path in stale:
        top_hooks.remove_file(path)
    if stale:
        log.info("Fleet: cleared %d stale stream file(s) from %s", len(stale), hls)
    for sentinel in (bot_stop_file(instance), control_file_path(run_dir)):
        if top_hooks.path_exists(sentinel):
            top_hooks.remove_file(sentinel)
            log.info("Fleet: cleared the untaken %s left in %s", sentinel.name, run_dir)


__all__ = ["clear_reused_slot"]
