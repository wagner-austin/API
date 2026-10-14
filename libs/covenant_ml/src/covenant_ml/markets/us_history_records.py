"""Strict native identity records for the declared historical sports feed."""

from __future__ import annotations

import re

from platform_core.json_utils import (
    JSONObject,
    narrow_json_to_dict,
    require_bool,
    require_dict,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import ReadOnly, TypedDict

from .validation import parse_timestamp, require_day, require_nonempty


class TeamLink(TypedDict):
    """Reviewed one-to-one link between the venues' native team identities."""

    kalshi_id: ReadOnly[str]
    us_id: ReadOnly[int]
    name: ReadOnly[str]


def encode_team_link(record: TeamLink) -> JSONObject:
    """Encode a reviewed identity link.

    Args:
        record: Validated team link.

    Returns:
        JSON object for persistence.
    """
    return {"kalshi_id": record["kalshi_id"], "us_id": record["us_id"], "name": record["name"]}


def decode_team_link(document: JSONObject) -> TeamLink:
    """Validate a native team identity link.

    Args:
        document: Decoded mapping entry.

    Returns:
        Immutable team link.

    Raises:
        ValueError: CROSS-001 for a nonpositive native ID.
        TypeError: Missing or incorrectly typed fields.
    """
    native_id = require_int(document, "us_id")
    if native_id <= 0:
        raise ValueError("[CROSS-001] native team ID must be positive")
    return TeamLink(
        kalshi_id=require_nonempty(document, "kalshi_id"),
        us_id=native_id,
        name=require_nonempty(document, "name"),
    )


class USGame(TypedDict):
    """Historical game identity and timing, without prices or outcomes."""

    slug: ReadOnly[str]
    day: ReadOnly[str]
    long_team: ReadOnly[int]
    short_team: ReadOnly[int]
    start_ms: ReadOnly[int]
    game_ms: ReadOnly[int]
    created_ms: ReadOnly[int]


def encode_us_game(record: USGame) -> JSONObject:
    """Encode a normalized historical game.

    Args:
        record: Validated game metadata.

    Returns:
        JSON object without terminal prices or outcomes.
    """
    return {
        "slug": record["slug"],
        "day": record["day"],
        "long_team": record["long_team"],
        "short_team": record["short_team"],
        "start_ms": record["start_ms"],
        "game_ms": record["game_ms"],
        "created_ms": record["created_ms"],
    }


def decode_us_game(document: JSONObject) -> USGame:
    """Validate normalized identity and metadata timestamps.

    Args:
        document: Decoded normalized game.

    Returns:
        Immutable game record.

    Raises:
        ValueError: CROSS-002 for inconsistent identity or chronology.
        TypeError: Missing or incorrectly typed fields.
    """
    slug, day = require_nonempty(document, "slug"), require_day(document, "day")
    long_team, short_team = require_int(document, "long_team"), require_int(document, "short_team")
    start, game, created = (
        require_int(document, key) for key in ("start_ms", "game_ms", "created_ms")
    )
    match = re.fullmatch(r"aec-nhl-[a-z]+-[a-z]+-(\d{4}-\d{2}-\d{2})", slug)
    if (
        match is None
        or not slug.endswith("-" + day)
        or min(long_team, short_team) <= 0
        or long_team == short_team
        or not 0 <= start < game
        or created < 0
    ):
        raise ValueError("[CROSS-002] invalid historical game identity or chronology")
    return USGame(
        slug=slug,
        day=day,
        long_team=long_team,
        short_team=short_team,
        start_ms=start,
        game_ms=game,
        created_ms=created,
    )


def decode_native_us_game(document: JSONObject) -> USGame:
    """Decode the historical moneyline schema with explicit opposing teams.

    Args:
        document: Native market-list row.

    Returns:
        Validated normalized game.

    Raises:
        ValueError: CROSS-003 for wrong market type or inconsistent native sides.
        TypeError: Required native metadata is malformed.
    """
    slug = require_nonempty(document, "slug")
    sides = tuple(narrow_json_to_dict(row) for row in require_list(document, "marketSides"))
    if (
        require_str(document, "sportsMarketType") != "moneyline"
        or len(sides) != 2
        or sum(require_bool(side, "long") for side in sides) != 1
    ):
        raise ValueError("[CROSS-003] historical market needs two opposing moneyline sides")
    teams: dict[bool, int] = {}
    for side in sides:
        team = require_dict(side, "team")
        identity = require_int(side, "teamId")
        if identity != require_int(team, "id") or require_str(team, "league") != "nhl":
            raise ValueError("[CROSS-003] native side and NHL team identity disagree")
        teams[require_bool(side, "long")] = identity
    return decode_us_game(
        {
            "slug": slug,
            "day": slug[-10:],
            "long_team": teams[True],
            "short_team": teams[False],
            "start_ms": parse_timestamp(require_str(document, "startDate")),
            "game_ms": parse_timestamp(require_str(document, "gameStartTime")),
            "created_ms": parse_timestamp(require_str(document, "createdAt")),
        }
    )
