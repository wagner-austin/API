"""What match to play, and the agent settings that ask for it.

Separated from :mod:`rw_bot.harness.launch`, which knows how to start *a* game,
because this knows *which* game. The distinction matters now that the match is
an experiment variable: the map, the opponent count and the difficulty are arms
in a sweep exactly as the seed is, rather than whatever the engine defaults to.

**The engine defaults are not a choice anyone made.** ``-sandbox`` queues a
script naming a ten-player map, and the setup is read out of a GUI document
that has no values headless -- so every figure falls through to a Java default:
four opponents, at Medium, on *Crossing Large (10p)*. Nobody picked that, and it
is a five-way free-for-all in which no opponent is ever eliminated
([[policy-determinism]]).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, TypedDict

from rw_bot import RwBotError
from rw_bot.validation import require_int, require_non_empty_str, require_positive_int

#: Lowest difficulty the engine defines: Very Easy.
MIN_DIFFICULTY: Final = -2

#: Highest difficulty the engine defines: Impossible.
MAX_DIFFICULTY: Final = 3

#: AI income multiplier by difficulty, from the engine's own arithmetic.
#:
#: ``n.E()`` returns ``1 + n*0.4`` above zero and ``1 + n*0.3`` at or below it,
#: with a further ``+1.5`` at Impossible, and it is applied to **AI players
#: only**. So at the default of 0 an opponent earns exactly what the bot does,
#: and the difficulty dial is an economy handicap rather than a change of play.
INCOME_MULTIPLIER: Final[Mapping[int, float]] = {
    -2: 0.4,
    -1: 0.7,
    0: 1.0,
    1: 1.4,
    2: 1.8,
    3: 3.7,
}

_BAD_DIFFICULTY = "RW-MATCH-001"


class MatchError(RwBotError):
    """A match was asked for that the engine cannot set up.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of the offending setting.
    """


class MatchConfig(TypedDict):
    """Which match to play.

    Attributes:
        map_path: Map as the engine names it, e.g.
            ``maps/skirmish/[p2]Lake (2p).tmx``. **Carries spaces and brackets**,
            which is why nothing here is ever rendered into a shell string.
        opponents: How many AI players to face. The engine caps this by the
            map's own team count, so a two-player map is a duel regardless.
        difficulty: AI difficulty, :data:`MIN_DIFFICULTY` to
            :data:`MAX_DIFFICULTY`. An income multiplier on the opponent alone;
            see :data:`INCOME_MULTIPLIER`.
    """

    map_path: str
    opponents: int
    difficulty: int


def decode_match_config(payload: Mapping[str, str | int]) -> MatchConfig:
    """Decode an untyped mapping into a :class:`MatchConfig`.

    Args:
        payload: Untyped mapping carrying every field.

    Returns:
        The validated configuration.

    Raises:
        DecodeError: When a field is absent, mistyped, blank or non-positive.
        MatchError: ``RW-MATCH-001`` when the difficulty is outside the scale
            the engine defines. Out of range is refused rather than clamped: a
            silently clamped arm reports a difficulty it did not play.
    """
    difficulty = require_int(payload, "difficulty")
    if difficulty < MIN_DIFFICULTY or difficulty > MAX_DIFFICULTY:
        raise MatchError(
            _BAD_DIFFICULTY,
            f"difficulty {difficulty} is outside the engine's scale of "
            f"{MIN_DIFFICULTY} (Very Easy) to {MAX_DIFFICULTY} (Impossible)",
        )
    return MatchConfig(
        map_path=require_non_empty_str(payload, "map_path"),
        opponents=require_positive_int(payload, "opponents"),
        difficulty=difficulty,
    )


def encode_match_config(config: MatchConfig) -> dict[str, str | int]:
    """Encode a :class:`MatchConfig` back to a plain mapping.

    Round-trips with :func:`decode_match_config`.

    Args:
        config: The configuration to encode.

    Returns:
        A plain mapping suitable for recording beside a run's artifacts.
    """
    return {
        "map_path": config["map_path"],
        "opponents": config["opponents"],
        "difficulty": config["difficulty"],
    }


def agent_options(
    *,
    port: int,
    sample_ms: int,
    seed: int,
    lockstep: int,
    match: MatchConfig | None,
) -> str:
    """Render the agent's ``;``-separated settings.

    Zero means "not asked for" for the seed and the lockstep, matching what the
    agent's own parser refuses: it rejects a zero seed rather than treating it
    as one, because zero is how every other option spells absence.

    Args:
        port: Port the command channel listens on.
        sample_ms: Milliseconds between world samples.
        seed: Engine random seed, or zero to leave it unpinned.
        lockstep: Engine frames between samples, or zero to run free.
        match: Which match to play, or None to leave the engine's own default.

    Returns:
        The option string, to be handed to
        :func:`~rw_bot.harness.launch.build_argv` as one argv element.
    """
    settings = [f"channelPort={port}", f"sampleIntervalMs={sample_ms}"]
    if seed:
        settings.append(f"randomSeed={seed}")
    if lockstep:
        settings.append(f"lockstepFrames={lockstep}")
    if match is not None:
        settings.append(f"matchMap={match['map_path']}")
        settings.append(f"matchOpponents={match['opponents']}")
        settings.append(f"matchDifficulty={match['difficulty']}")
    return ";".join(settings)


def describe(match: MatchConfig) -> str:
    """Render a match as one human-readable line, for the run log.

    Args:
        match: The match.

    Returns:
        A description naming the handicap, because "difficulty -2" says nothing
        about what it does and "0.4x income" says all of it.
    """
    multiplier = INCOME_MULTIPLIER[match["difficulty"]]
    return (
        f"{match['opponents']} opponent(s) at difficulty {match['difficulty']} "
        f"({multiplier:g}x AI income) on {match['map_path']}"
    )


__all__ = [
    "INCOME_MULTIPLIER",
    "MAX_DIFFICULTY",
    "MIN_DIFFICULTY",
    "MatchConfig",
    "MatchError",
    "agent_options",
    "decode_match_config",
    "describe",
    "encode_match_config",
]
