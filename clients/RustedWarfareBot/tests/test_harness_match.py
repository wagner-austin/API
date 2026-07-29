"""Which match an experiment asks the engine for.

The map, the opponent count and the difficulty are arm variables now, so they
are validated like every other one: an arm that reports a difficulty it did not
play is worse than an arm that refuses to start.
"""

from __future__ import annotations

import pytest

from rw_bot.harness.match import (
    INCOME_MULTIPLIER,
    MAX_DIFFICULTY,
    MIN_DIFFICULTY,
    MatchConfig,
    MatchError,
    agent_options,
    decode_match_config,
    describe,
    encode_match_config,
)
from rw_bot.validation import DecodeError

#: A real map path, brackets and all. The engine reads the player count out of
#: the ``[pN]`` prefix, so the shape of the name is load-bearing rather than
#: decorative -- an alias without it reported ``Max teams = -1``.
_MAP = "maps/skirmish/[p2]duel_lake.tmx"


def _config(difficulty: int = -2, opponents: int = 1) -> MatchConfig:
    return MatchConfig(map_path=_MAP, opponents=opponents, difficulty=difficulty)


def test_a_match_round_trips() -> None:
    config = _config()
    assert decode_match_config(encode_match_config(config)) == config


def test_every_difficulty_the_engine_defines_is_accepted() -> None:
    """The scale is -2 to 3, and each value is a different opponent economy."""
    for difficulty in range(MIN_DIFFICULTY, MAX_DIFFICULTY + 1):
        decoded = decode_match_config({"map_path": _MAP, "opponents": 1, "difficulty": difficulty})
        assert decoded["difficulty"] == difficulty


def test_a_difficulty_off_the_scale_is_refused_rather_than_clamped() -> None:
    """Clamping would let an arm report a difficulty it did not play."""
    for difficulty in (MIN_DIFFICULTY - 1, MAX_DIFFICULTY + 1):
        with pytest.raises(MatchError) as caught:
            decode_match_config({"map_path": _MAP, "opponents": 1, "difficulty": difficulty})
        assert caught.value.code == "RW-MATCH-001"
        assert str(difficulty) in caught.value.message


def test_a_blank_map_is_refused() -> None:
    with pytest.raises(DecodeError):
        decode_match_config({"map_path": "", "opponents": 1, "difficulty": 0})


def test_a_non_positive_opponent_count_is_refused() -> None:
    with pytest.raises(DecodeError):
        decode_match_config({"map_path": _MAP, "opponents": 0, "difficulty": 0})


def test_the_income_table_matches_the_engines_own_arithmetic() -> None:
    """``n.E()`` is ``1 + n*0.4`` above zero and ``1 + n*0.3`` at or below it,
    with a further ``+1.5`` at Impossible, applied to AI players only.

    Asserted because the whole point of the difficulty dial is that it is an
    economy handicap: at the default of 0 an opponent earns exactly what the
    bot does, which is the game every measurement before this one was taken in.
    """
    assert INCOME_MULTIPLIER[0] == pytest.approx(1.0)
    assert INCOME_MULTIPLIER[-2] == pytest.approx(1.0 + -2 * 0.3)
    assert INCOME_MULTIPLIER[-1] == pytest.approx(1.0 + -1 * 0.3)
    assert INCOME_MULTIPLIER[1] == pytest.approx(1.0 + 1 * 0.4)
    assert INCOME_MULTIPLIER[2] == pytest.approx(1.0 + 2 * 0.4)
    assert INCOME_MULTIPLIER[3] == pytest.approx(1.0 + 3 * 0.4 + 1.5)


def test_the_options_carry_every_match_field() -> None:
    rendered = agent_options(port=27800, sample_ms=250, seed=12345, lockstep=75, match=_config())
    assert rendered == (
        "channelPort=27800;sampleIntervalMs=250;randomSeed=12345;lockstepFrames=75;"
        f"matchMap={_MAP};matchOpponents=1;matchDifficulty=-2"
    )


def test_the_map_rides_in_one_option_with_its_brackets_intact() -> None:
    """The whole string becomes one argv element, so nothing needs quoting.

    Assembled as a shell command line instead, a map path carrying a space
    split the ``-javaagent`` flag in two and the JVM aborted with ``processing
    of -javaagent failed`` before the agent loaded ([[policy-determinism]]).
    """
    spaced = "maps/skirmish/[p2]Lake (2p).tmx"
    rendered = agent_options(
        port=1,
        sample_ms=250,
        seed=0,
        lockstep=0,
        match=MatchConfig(map_path=spaced, opponents=1, difficulty=0),
    )
    assert f"matchMap={spaced}" in rendered


def test_no_match_leaves_the_engines_own_default() -> None:
    """Absent means absent: the probes that predate a chosen match pass none."""
    rendered = agent_options(port=27800, sample_ms=250, seed=0, lockstep=0, match=None)
    assert rendered == "channelPort=27800;sampleIntervalMs=250"


def test_a_zero_seed_and_lockstep_are_omitted_rather_than_sent() -> None:
    """Zero is how every option spells absence, and the agent rejects a zero
    seed rather than treating it as one.
    """
    rendered = agent_options(port=27800, sample_ms=250, seed=0, lockstep=0, match=_config())
    assert "randomSeed" not in rendered
    assert "lockstepFrames" not in rendered


def test_a_description_names_the_handicap_rather_than_the_number() -> None:
    """ "difficulty -2" says nothing about what it does; "0.4x income" says it."""
    rendered = describe(_config())
    assert "1 opponent(s)" in rendered
    assert "0.4x AI income" in rendered
    assert _MAP in rendered
