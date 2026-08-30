"""Building a batch's configuration, shared by every submission door.

The CLI and the HTTP surface accept the same knobs a sweep takes; this is
the one place they become a validated :class:`SweepConfig`, so a batch
submitted over the wire and a batch submitted from a shell are the same
batch ([[harness-match-service]]).
"""

from __future__ import annotations

from rw_bot.harness.clone import CLONE_PREFIX
from rw_bot.harness.match import MatchConfig, decode_match_config
from rw_bot.harness.results_layout import PINNED_GAME_DIR, SWEEP_ROOT, TRACE_ROOT
from rw_bot.harness.runner import TREE_DIR, SweepConfig, decode_sweep_config

#: One opponent: the duel, whose count the map caps anyway.
DUEL_OPPONENTS = 1


def batch_config(
    name: str,
    lockstep: int,
    map_path: str,
    difficulty: int,
    pin_delta: int,
    fast_forward: int,
) -> SweepConfig:
    """Build the configuration a queued batch stores.

    Args:
        name: The sweep name artifacts will file under.
        lockstep: Frames per planner step.
        map_path: The match map, or an empty string for the engine's own
            sandbox.
        difficulty: AI difficulty for the match, ignored when no map.
        pin_delta: The determinism pin in milliseconds, zero for wall clock.
        fast_forward: The speed multiplier, zero for realtime.

    Returns:
        The validated configuration, workers pinned to 1 because how many
        workers play a queued batch is decided by who polls.

    Raises:
        DecodeError: When a knob is out of range.
    """
    match: MatchConfig | None = None
    if map_path != "":
        match = decode_match_config(
            {"map_path": map_path, "opponents": DUEL_OPPONENTS, "difficulty": difficulty}
        )
    # Forward-slashed rather than through `Path`, whose str() is backslashed
    # on Windows: this goes into the batch's config and out again into every
    # path composed from it, one of which a child process is handed.
    out_dir = f"{SWEEP_ROOT}/{name}"
    return decode_sweep_config(
        {
            "out_dir": out_dir,
            # The repository-relative root: the fleet service runs where the
            # repository is, unlike a cluster member.
            "traces": TRACE_ROOT,
            "workers": 1,
            "lockstep": lockstep,
            "clone_prefix": CLONE_PREFIX,
            "source_game_dir": PINNED_GAME_DIR,
            "tree": f"{out_dir}/{TREE_DIR}",
            "pin_delta": pin_delta,
            "fast_forward": fast_forward,
        },
        match,
    )
