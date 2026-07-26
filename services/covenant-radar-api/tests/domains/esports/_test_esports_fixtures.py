"""Shared match snapshot builder for the esports domain tests.

A match snapshot carries fifteen fields, but any one test varies two or
three of them. Building the whole event inline in each test buries the one
value under test among a dozen that are only there to satisfy the type, so
the default here is an even scoreline ten minutes in and every test states
only what it changes.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from covenant_radar_api.domains.esports.schemas import (
    MatchEventV1,
    encode_match_event,
    make_match_event,
)

DEFAULT_MATCH_ID = "match-1"
DEFAULT_TIMESTAMP = "2026-07-25T18:00:00Z"

# Ten minutes in, so a per-minute rate is well defined and the zero-elapsed
# case is something a test opts into rather than the default.
DEFAULT_GAME_TIME_SECONDS = 600


def make_snapshot(
    *,
    event_id: str = "evt-1",
    match_id: str = DEFAULT_MATCH_ID,
    game_number: int = 1,
    game_time_seconds: int = DEFAULT_GAME_TIME_SECONDS,
    blue_kills: int = 0,
    red_kills: int = 0,
    blue_gold: int = 0,
    red_gold: int = 0,
    blue_towers: int = 0,
    red_towers: int = 0,
    blue_dragons: int = 0,
    red_dragons: int = 0,
    blue_barons: int = 0,
    red_barons: int = 0,
    timestamp: str = DEFAULT_TIMESTAMP,
) -> MatchEventV1:
    """Build a match state snapshot, defaulting to an even scoreline.

    Args:
        event_id: UUID for deduplication.
        match_id: Match identifier.
        game_number: Which game of the series, starting at 1.
        game_time_seconds: Elapsed game time in seconds.
        blue_kills: Kills by the blue side.
        red_kills: Kills by the red side.
        blue_gold: Total gold earned by the blue side.
        red_gold: Total gold earned by the red side.
        blue_towers: Towers destroyed by the blue side.
        red_towers: Towers destroyed by the red side.
        blue_dragons: Dragons taken by the blue side.
        red_dragons: Dragons taken by the red side.
        blue_barons: Barons taken by the blue side.
        red_barons: Barons taken by the red side.
        timestamp: ISO datetime when the snapshot was taken.

    Returns:
        MatchEventV1 with the given values over an even-scoreline default.
    """
    return make_match_event(
        event_id=event_id,
        match_id=match_id,
        game_number=game_number,
        game_time_seconds=game_time_seconds,
        blue_kills=blue_kills,
        red_kills=red_kills,
        blue_gold=blue_gold,
        red_gold=red_gold,
        blue_towers=blue_towers,
        red_towers=red_towers,
        blue_dragons=blue_dragons,
        red_dragons=red_dragons,
        blue_barons=blue_barons,
        red_barons=red_barons,
        timestamp=timestamp,
    )


def make_payload(
    *,
    event_id: str = "evt-1",
    match_id: str = DEFAULT_MATCH_ID,
    game_time_seconds: int = DEFAULT_GAME_TIME_SECONDS,
    blue_kills: int = 0,
    red_kills: int = 0,
    blue_gold: int = 0,
    red_gold: int = 0,
    blue_towers: int = 0,
    red_towers: int = 0,
    blue_dragons: int = 0,
    red_dragons: int = 0,
    blue_barons: int = 0,
    red_barons: int = 0,
) -> str:
    """Build a snapshot and encode it as it would arrive from Kafka.

    Args:
        event_id: UUID for deduplication.
        match_id: Match identifier.
        game_time_seconds: Elapsed game time in seconds.
        blue_kills: Kills by the blue side.
        red_kills: Kills by the red side.
        blue_gold: Total gold earned by the blue side.
        red_gold: Total gold earned by the red side.
        blue_towers: Towers destroyed by the blue side.
        red_towers: Towers destroyed by the red side.
        blue_dragons: Dragons taken by the blue side.
        red_dragons: Dragons taken by the red side.
        blue_barons: Barons taken by the blue side.
        red_barons: Barons taken by the red side.

    Returns:
        Compact JSON string of the snapshot.
    """
    return encode_match_event(
        make_snapshot(
            event_id=event_id,
            match_id=match_id,
            game_time_seconds=game_time_seconds,
            blue_kills=blue_kills,
            red_kills=red_kills,
            blue_gold=blue_gold,
            red_gold=red_gold,
            blue_towers=blue_towers,
            red_towers=red_towers,
            blue_dragons=blue_dragons,
            red_dragons=red_dragons,
            blue_barons=blue_barons,
            red_barons=red_barons,
        )
    )


__all__ = [
    "DEFAULT_GAME_TIME_SECONDS",
    "DEFAULT_MATCH_ID",
    "DEFAULT_TIMESTAMP",
    "make_payload",
    "make_snapshot",
]
