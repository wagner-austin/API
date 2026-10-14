"""Read-only paired sports snapshots with pinned rules and bounded receipt times."""

from __future__ import annotations

from hashlib import sha256

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    narrow_json_to_dict,
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_list,
    require_str,
)

from .archive import ResponseArchive
from .paper_records import BookTop, decode_native_book, encode_book_top
from .sports_records import SportsPair, decode_us_book, encode_sports_pair
from .validation import parse_timestamp, require_nonempty


def sports_rules(kalshi: JSONObject, us: JSONObject) -> JSONObject:
    """Extract both complete text fields without treating similar text as equivalence.

    Args:
        kalshi: Native market metadata.
        us: Native market metadata.

    Returns:
        Canonical rule material for the frozen selection digest.

    Raises:
        TypeError: Required fields are absent or malformed.
        ValueError: A primary rule or US description is blank.
    """
    return {
        "kalshi_primary": require_nonempty(kalshi, "rules_primary"),
        "kalshi_secondary": require_str(kalshi, "rules_secondary"),
        "us_description": require_nonempty(us, "description"),
    }


def _validate_markets(pair: SportsPair, kalshi: JSONObject, us: JSONObject) -> JSONObject:
    """Require unchanged open contracts, team orientation and frozen rule material.

    Args:
        pair: Frozen selection.
        kalshi: Current native contract.
        us: Current native contract.

    Returns:
        Verified rule text.

    Raises:
        ValueError: SPORTS-004 for identity/state, SPORTS-005 for rule drift.
        TypeError: Native metadata fields are malformed.
    """
    sides = tuple(narrow_json_to_dict(row) for row in require_list(us, "marketSides"))
    long_sides = tuple(side for side in sides if require_bool(side, "long"))
    if (
        require_str(kalshi, "ticker") != pair["kalshi_ticker"]
        or require_str(kalshi, "status") != "active"
        or require_str(kalshi, "result") != ""
        or require_str(require_dict(kalshi, "custom_strike"), "hockey_team") != pair["kalshi_team"]
        or require_str(us, "slug") != pair["us_slug"]
        or require_str(us, "sportsMarketType") != "hockey_team_full_game_winner"
        or not require_bool(us, "active")
        or require_bool(us, "closed")
        or len(sides) != 2
        or len(long_sides) != 1
        or require_int(long_sides[0], "teamId") != pair["us_team"]
    ):
        raise ValueError("[SPORTS-004] contract state or declared same-team identity changed")
    rules = sports_rules(kalshi, us)
    if sha256(dump_json_str(rules).encode("utf-8")).hexdigest() != pair["rules_sha256"]:
        raise ValueError("[SPORTS-005] contract rules differ from the frozen selection")
    return rules


def _directions(kalshi: BookTop, us: BookTop) -> list[JSONValue]:
    """Compute both opposing-position costs and visible capacity without fee assumptions.

    Args:
        kalshi: YES book for the declared team.
        us: Long-side book for that same declared team.

    Returns:
        Gross screens in ten-thousandth dollars, with size in centi-contracts.
    """
    rows: list[JSONValue] = []
    for name, first, second in (("kalshi_yes_us_no", kalshi, us), ("us_yes_kalshi_no", us, kalshi)):
        price = first["ask"] + 10000 - second["bid"]
        rows.append(
            {
                "direction": name,
                "pair_price": price,
                "ordinary_completion_gross_bound": 10000 - price,
                "visible_size": min(first["ask_size"], second["bid_size"]),
                "size_checks": [
                    {
                        "contracts": quantity,
                        "gross_cost": price * quantity,
                        "sufficient_top_depth": min(first["ask_size"], second["bid_size"])
                        >= quantity * 100,
                        "within_gross_20_dollar_budget": price * quantity <= 200000,
                    }
                    for quantity in (1, 5, 20)
                ],
            }
        )
    return rows


def collect_sports_pair(
    pair: SportsPair,
    kalshi: ResponseArchive,
    us: ResponseArchive,
    capture: bool,
    start_ms: int,
    end_ms: int,
    max_skew_ms: int,
) -> JSONObject:
    """Capture or replay one fully declared pair without submitting an order.

    Args:
        pair: Validated frozen identities and rule digest.
        kalshi: Existing archive for the Kalshi API prefix.
        us: Existing archive for the Polymarket US origin.
        capture: Fresh acquisition when true, strictly offline replay otherwise.
        start_ms: Inclusive receipt-window start.
        end_ms: Inclusive receipt-window end.
        max_skew_ms: Maximum accepted book receipt separation.

    Returns:
        Deterministic evidence, gross screens and explicit exclusions. Fee metadata
        is preserved but not asserted to be a complete fee schedule.

    Raises:
        ValueError: SPORTS-006 for window/clock failure; native codec failures.
        HTTPStatusError: A required request failed; no recovery is attempted.
        OSError: Required evidence cannot be written or read.
    """
    if not 0 <= start_ms < end_ms or max_skew_ms <= 0:
        raise ValueError("[SPORTS-006] invalid acquisition window or receipt tolerance")
    requests = (
        (kalshi, "/markets/" + pair["kalshi_ticker"]),
        (us, "/v1/market/slug/" + pair["us_slug"]),
        (kalshi, "/markets/" + pair["kalshi_ticker"] + "/orderbook"),
        (us, "/v1/markets/" + pair["us_slug"] + "/book"),
    )
    documents: list[JSONObject] = []
    receipts: list[int] = []
    previous = start_ms
    for archive, path in requests:
        document = archive.capture(path, {}) if capture else archive.replay(path, {})
        envelope = archive.load_capture(path, {})
        received = parse_timestamp(envelope["captured_at"])
        if not previous <= received <= end_ms:
            raise ValueError("[SPORTS-006] response outside window or receipt clock regressed")
        documents.append(document)
        receipts.append(received)
        previous = received
    native_k, native_u = require_dict(documents[0], "market"), require_dict(documents[1], "market")
    rules = _validate_markets(pair, native_k, native_u)
    fee = require_float(native_u, "feeCoefficient")
    if not 0 <= fee <= 1:
        raise ValueError("[SPORTS-008] US fee coefficient must be finite and between zero and one")
    k_book, u_book = decode_native_book(documents[2]), decode_us_book(documents[3], pair["us_slug"])
    skew = receipts[3] - receipts[2]
    exclusion = ""
    directions: list[JSONValue] = []
    if k_book is None or u_book is None:
        exclusion = "missing_book_side"
    elif skew > max_skew_ms:
        exclusion = "book_receipt_skew"
    else:
        directions = _directions(k_book, u_book)
    return {
        "pair": encode_sports_pair(pair),
        "rules": rules,
        "receipts_ms": list(receipts),
        "book_receipt_skew_ms": skew,
        "last_received_ms": receipts[-1],
        "kalshi_book": None if k_book is None else encode_book_top(k_book),
        "us_book": None if u_book is None else encode_book_top(u_book),
        "us_fee_coefficient": fee,
        "directions": directions,
        "exclusion": exclusion,
        "settlement_equivalence_verified": False,
        "fees_verified": False,
        "orders_submitted": 0,
    }


__all__ = ["collect_sports_pair", "sports_rules"]
