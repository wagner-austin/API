"""Building a batch's configuration, shared by every submission door.

The CLI and the HTTP surface accept the same knobs a sweep takes; this is
the one place they become a validated :class:`SweepConfig`, so a batch
submitted over the wire and a batch submitted from a shell are the same
batch ([[harness-match-service]]).
"""

from __future__ import annotations

from pathlib import Path

from rw_bot.harness.match import MatchConfig, decode_match_config
from rw_bot.harness.runner import TREE_DIR, SweepConfig, decode_sweep_config

#: Where batches file, identical to the sweep CLI's constant.
SWEEP_ROOT = "runs/sweeps"

#: The clone prefix every worker shares with the sweep harness.
CLONE_PREFIX = ".game-w"

#: The pinned game dir clones copy from.
SOURCE_GAME_DIR = ".game"

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
    out_dir = Path(SWEEP_ROOT) / name
    return decode_sweep_config(
        {
            "out_dir": str(out_dir),
            "workers": 1,
            "lockstep": lockstep,
            "clone_prefix": CLONE_PREFIX,
            "source_game_dir": SOURCE_GAME_DIR,
            "tree": str(out_dir / TREE_DIR),
            "pin_delta": pin_delta,
            "fast_forward": fast_forward,
        },
        match,
    )
