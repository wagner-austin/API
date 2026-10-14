"""Frozen sports identities and exact Polymarket US book normalization."""

from __future__ import annotations

import re

from platform_core.json_utils import (
    JSONObject,
    narrow_json_to_dict,
    require_dict,
    require_list,
    require_str,
)
from typing_extensions import ReadOnly, TypedDict

from .paper_records import BookTop, decode_book_top
from .validation import parse_fixed, require_bounded_int, require_nonempty


class SportsPair(TypedDict):
    """Declared same-team contracts with independently frozen native identities."""

    kalshi_ticker: ReadOnly[str]
    us_slug: ReadOnly[str]
    kalshi_team: ReadOnly[str]
    us_team: ReadOnly[int]
    rules_sha256: ReadOnly[str]


def encode_sports_pair(pair: SportsPair) -> JSONObject:
    """Encode the immutable selection without deriving settlement equivalence.

    Args:
        pair: Validated selection.

    Returns:
        JSON identity and rule digest.
    """
    return {
        "kalshi_ticker": pair["kalshi_ticker"],
        "us_slug": pair["us_slug"],
        "kalshi_team": pair["kalshi_team"],
        "us_team": pair["us_team"],
        "rules_sha256": pair["rules_sha256"],
    }


def decode_sports_pair(document: JSONObject) -> SportsPair:
    """Require full-game NHL identities and a canonical frozen rule digest.

    Args:
        document: Untrusted selection JSON.

    Returns:
        Immutable selection.

    Raises:
        ValueError: SPORTS-001 for unsupported identity or digest.
        TypeError: Required fields have incorrect types.
    """
    ticker = require_nonempty(document, "kalshi_ticker")
    slug = require_nonempty(document, "us_slug")
    digest = require_nonempty(document, "rules_sha256")
    if (
        re.fullmatch(r"KXNHLGAME-[A-Z0-9]+-[A-Z]+", ticker) is None
        or re.fullmatch(r"aec-nhl-[a-z]+-[a-z]+-\d{4}-\d{2}-\d{2}", slug) is None
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise ValueError("[SPORTS-001] invalid full-game identity or rules digest")
    return SportsPair(
        kalshi_ticker=ticker,
        us_slug=slug,
        kalshi_team=require_nonempty(document, "kalshi_team"),
        us_team=require_bounded_int(document, "us_team", 1, 2**63 - 1),
        rules_sha256=digest,
    )


def _us_levels(document: JSONObject, key: str) -> dict[int, int]:
    """Decode every US level into dollar ten-thousandths and centi-contracts.

    Args:
        document: Native marketData object.
        key: Required bids or offers field.

    Returns:
        Unique price levels retaining exact size.

    Raises:
        ValueError: SPORTS-002 for currency, precision, size or duplicate levels.
        TypeError: Required native fields are malformed.
    """
    levels: dict[int, int] = {}
    for value in require_list(document, key):
        level = narrow_json_to_dict(value)
        quote = require_dict(level, "px")
        price = parse_fixed(require_str(quote, "value"), 4)
        size = parse_fixed(require_str(level, "qty"), 4)
        if (
            require_str(quote, "currency") != "USD"
            or price > 10000
            or not 0 < size < 2**63
            or size % 100 != 0
            or price in levels
        ):
            raise ValueError("[SPORTS-002] invalid US book currency, price or exact size")
        levels[price] = size // 100
    return levels


def decode_us_book(document: JSONObject, slug: str) -> BookTop | None:
    """Validate the requested open book and select its actual best prices.

    Args:
        document: Native book response.
        slug: Expected exact full-game market identity.

    Returns:
        Existing shared book record, or explicit missing-side state.

    Raises:
        ValueError: SPORTS-003 for identity or state; SPORTS-002 for levels.
        TypeError: Native response is malformed.
    """
    book = require_dict(document, "marketData")
    if require_str(book, "marketSlug") != slug or require_str(book, "state") != "MARKET_STATE_OPEN":
        raise ValueError("[SPORTS-003] US book must identify the requested open market")
    bids, offers = _us_levels(book, "bids"), _us_levels(book, "offers")
    if not bids or not offers:
        return None
    bid, ask = max(bids), min(offers)
    return decode_book_top({"bid": bid, "ask": ask, "bid_size": bids[bid], "ask_size": offers[ask]})


__all__ = ["SportsPair", "decode_sports_pair", "decode_us_book", "encode_sports_pair"]
