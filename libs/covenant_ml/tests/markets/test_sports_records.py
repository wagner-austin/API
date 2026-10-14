"""Exact book normalization, frozen identities and malformed native evidence."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONValue, require_dict, require_list

from covenant_ml.markets.paper_records import BookTop
from covenant_ml.markets.sports_records import (
    decode_sports_pair,
    decode_us_book,
    encode_sports_pair,
)

from ._sports_fixtures import SLUG, sports_pair, us_book


def test_selection_roundtrip_and_unsorted_exact_books() -> None:
    pair = sports_pair()
    assert decode_sports_pair(encode_sports_pair(pair)) == pair
    document = us_book()
    native = require_dict(document, "marketData")
    require_list(native, "bids").append(
        {"px": {"value": "0.5500", "currency": "USD"}, "qty": "1.2300"}
    )
    require_list(native, "offers").append(
        {"px": {"value": "0.5700", "currency": "USD"}, "qty": "99.0000"}
    )
    assert decode_us_book(document, SLUG) == BookTop(bid=5500, ask=5500, bid_size=123, ask_size=900)


@pytest.mark.parametrize(
    "key,value", [("kalshi_ticker", "OTHER"), ("us_slug", "../bad"), ("rules_sha256", "a")]
)
def test_invalid_selection(key: str, value: str) -> None:
    document = encode_sports_pair(sports_pair())
    document[key] = value
    with pytest.raises(ValueError, match="SPORTS-001"):
        decode_sports_pair(document)


@pytest.mark.parametrize("key,value", [("marketSlug", "wrong"), ("state", "MARKET_STATE_CLOSED")])
def test_invalid_book_identity(key: str, value: str) -> None:
    document = us_book()
    require_dict(document, "marketData")[key] = value
    with pytest.raises(ValueError, match="SPORTS-003"):
        decode_us_book(document, SLUG)


@pytest.mark.parametrize("side", ["bids", "offers"])
def test_missing_book_side(side: str) -> None:
    document = us_book()
    require_dict(document, "marketData")[side] = []
    assert decode_us_book(document, SLUG) is None


@pytest.mark.parametrize(
    "price,size,currency",
    [
        ("1.0001", "1.0000", "USD"),
        ("0.5000", "0.0000", "USD"),
        ("0.5000", "1.0001", "USD"),
        ("0.5000", "1.0000", "EUR"),
        ("0.5000", "9223372036854775808", "USD"),
    ],
)
def test_invalid_level(price: str, size: str, currency: str) -> None:
    document = us_book()
    require_dict(document, "marketData")["bids"] = [
        {"px": {"value": price, "currency": currency}, "qty": size}
    ]
    with pytest.raises(ValueError, match="SPORTS-002"):
        decode_us_book(document, SLUG)


def test_duplicate_crossed_and_malformed_books() -> None:
    document = us_book()
    bids = require_list(require_dict(document, "marketData"), "bids")
    bids.append(bids[0])
    with pytest.raises(ValueError, match="SPORTS-002"):
        decode_us_book(document, SLUG)
    bids[:] = [{"px": {"value": "0.5600", "currency": "USD"}, "qty": "1"}]
    with pytest.raises(ValueError, match="PAPER-002"):
        decode_us_book(document, SLUG)
    for malformed in (None, True, 4):
        bad: JSONValue = malformed
        bids[:] = [bad]
        with pytest.raises(TypeError):
            decode_us_book(document, SLUG)
