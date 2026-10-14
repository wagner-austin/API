"""Complete historical inventory, explicit team joins and cutoff-safe forecasts."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_dict,
    require_float,
    require_int,
    require_list,
    require_str,
)

from .kalshi import MarketSpec
from .types import DOLLAR
from .us_history_records import USGame, decode_native_us_game, decode_team_link
from .validation import parse_fixed, require_nonempty


def team_links(document: JSONObject) -> dict[str, int]:
    """Require a nonempty bijection between independently reviewed team IDs.

    Args:
        document: Mapping file with a teams array.

    Returns:
        Kalshi IDs mapped to US IDs.

    Raises:
        ValueError: CROSS-004 for empty or non-bijective mappings.
        TypeError: Invalid mapping fields.
    """
    links = tuple(
        decode_team_link(narrow_json_to_dict(row)) for row in require_list(document, "teams")
    )
    result = {link["kalshi_id"]: link["us_id"] for link in links}
    if not links or len(result) != len(links) or len(set(result.values())) != len(links):
        raise ValueError("[CROSS-004] team mapping must be a nonempty bijection")
    return result


def us_inventory(
    read: Callable[[str, dict[str, str | int]], JSONObject],
) -> tuple[tuple[USGame, ...], dict[str, int]]:
    """Page the declared inventory through an empty page, rejecting duplicates.

    Args:
        read: Explicit archive capture or replay method.

    Returns:
        Native NHL games and inventory counts.

    Raises:
        ValueError: CROSS-005 for repeated native identities or an oversized page.
        HTTPStatusError: The public acquisition failed.
    """
    offset, pages = 0, 0
    ids: set[str] = set()
    slugs: set[str] = set()
    games: list[USGame] = []
    while True:
        document = read(
            "/v1/markets",
            {
                "limit": 500,
                "offset": offset,
                "categories": "sports",
                "endDateMin": "2026-01-01T00:00:00Z",
                "endDateMax": "2026-04-17T00:00:00Z",
            },
        )
        rows = require_list(document, "markets")
        pages += 1
        if len(rows) > 500:
            raise ValueError("[CROSS-005] inventory page exceeds requested limit")
        for value in rows:
            row = narrow_json_to_dict(value)
            identity, slug = require_nonempty(row, "id"), require_nonempty(row, "slug")
            if identity in ids or slug in slugs:
                raise ValueError("[CROSS-005] historical inventory repeats a native identity")
            ids.add(identity)
            slugs.add(slug)
            if slug.startswith("aec-nhl-"):
                games.append(decode_native_us_game(row))
        if not rows:
            break
        # Offsets index the server's requested window, including filtered records.
        offset += 500
    return tuple(games), {"pages": pages, "native_markets": len(ids), "nhl_games": len(games)}


def history_probability(
    game: USGame,
    cutoff_ms: int,
    selected_team: int,
    read: Callable[[str, dict[str, str | int]], JSONObject],
) -> tuple[float | None, str]:
    """Validate every stored observation before selecting a fresh display midpoint.

    Args:
        game: Validated native market metadata.
        cutoff_ms: Declared decision time in milliseconds.
        selected_team: Native US ID of the selected Kalshi side.
        read: Explicit history capture or replay method.

    Returns:
        Oriented probability and eligibility reason; missing data stays excluded.

    Raises:
        ValueError: CROSS-006 for invalid query identity or time;
            CROSS-007 for malformed, crossed or non-monotonic history.
        HTTPStatusError: Acquisition failed.
    """
    if (
        cutoff_ms < 3600000
        or cutoff_ms % 1000
        or selected_team not in (game["long_team"], game["short_team"])
    ):
        raise ValueError("[CROSS-006] invalid cutoff or selected team")
    if not game["start_ms"] <= cutoff_ms < game["game_ms"]:
        return None, "outside_pregame_window"
    cutoff = cutoff_ms // 1000
    document = read(
        "/v1/price-history",
        {
            "symbol": game["slug"],
            "timestamp.startTimestamp": cutoff - 3600,
            "timestamp.endTimestamp": cutoff,
            "fidelity": 1,
        },
    )
    previous = cutoff - 3601
    prices: list[tuple[int, int, int]] = []
    for value in require_list(document, "history"):
        row = narrow_json_to_dict(value)
        stamp = require_int(row, "timestamp")
        yes, no = (require_float(row, key) for key in ("longPrice", "shortPrice"))
        if not previous < stamp <= cutoff or not 0 <= yes <= 1 or not 0 <= no <= 1:
            raise ValueError("[CROSS-007] invalid historical timestamp or display price")
        ask, bid = parse_fixed(str(yes), 4), DOLLAR - parse_fixed(str(no), 4)
        if bid > ask:
            raise ValueError("[CROSS-007] crossed historical display prices")
        previous = stamp
        prices.append((stamp, bid, ask))
    if not prices:
        return None, "missing_history"
    stamp, bid, ask = prices[-1]
    if cutoff - stamp > 60:
        return None, "stale_history"
    if bid <= 0 or ask >= DOLLAR:
        return None, "boundary_display"
    if ask - bid > 500:
        return None, "wide_display"
    probability = (bid + ask) / (2 * DOLLAR)
    return (probability if selected_team == game["long_team"] else 1 - probability), "eligible"


def _pair_identities(
    pair: Sequence[JSONObject],
    event: str,
    links: dict[str, int],
) -> dict[str, int]:
    """Resolve both opposing contracts through the reviewed identity map.

    Args:
        pair: Exactly two native contracts, checked by the caller.
        event: Expected event identity.
        links: Reviewed native team bijection.

    Returns:
        Native ticker to US team ID mapping.

    Raises:
        ValueError: CROSS-008 for unmapped teams or inconsistent tickers.
    """
    identities: dict[str, int] = {}
    for row in pair:
        team = require_nonempty(require_dict(row, "custom_strike"), "hockey_team")
        ticker = require_nonempty(row, "ticker")
        if team not in links or ticker in identities or not ticker.startswith(event + "-"):
            raise ValueError("[CROSS-008] missing team link or inconsistent native ticker")
        identities[ticker] = links[team]
    return identities


def cross_predictions(
    records: Sequence[JSONObject],
    specs: Sequence[MarketSpec],
    games: Sequence[USGame],
    links: dict[str, int],
    read: Callable[[str, dict[str, str | int]], JSONObject],
) -> tuple[dict[str, float], list[JSONValue]]:
    """Join both native teams and report one forecast or exclusion per decision.

    Args:
        records: Complete Kalshi metadata with native team identities.
        specs: Original selected decision specifications.
        games: Validated historical US inventory.
        links: Reviewed bijection between native team IDs.
        read: Explicit history acquisition or replay method.

    Returns:
        Eligible oriented forecasts and complete per-decision coverage records.

    Raises:
        ValueError: CROSS-008 for ambiguous games, incomplete native pairs or unmapped teams.
        HTTPStatusError: Required acquisition failed.
    """
    index: dict[tuple[str, frozenset[int]], USGame] = {}
    for game in games:
        key = (game["day"], frozenset((game["long_team"], game["short_team"])))
        if key in index:
            raise ValueError("[CROSS-008] multiple US markets match one game")
        index[key] = game
    grouped: dict[str, list[JSONObject]] = {}
    for row in records:
        grouped.setdefault(require_str(row, "event_ticker"), []).append(row)
    forecasts: dict[str, float] = {}
    coverage: list[JSONValue] = []
    seen: set[str] = set()
    for spec in specs:
        event = spec["ticker"].rsplit("-", 1)[0]
        pair = grouped.get(event, [])
        if event in seen or len(pair) != 2:
            raise ValueError("[CROSS-008] duplicate decision or incomplete Kalshi pair")
        seen.add(event)
        identities = _pair_identities(pair, event, links)
        if len(set(identities.values())) != 2 or spec["ticker"] not in identities:
            raise ValueError("[CROSS-008] selected team or opposing pair is inconsistent")
        key = (spec["day"], frozenset(identities.values()))
        record: JSONObject = {"ticker": spec["ticker"], "status": "unmatched_game"}
        if key in index:
            game = index[key]
            probability, status = history_probability(
                game, spec["decision_ms"], identities[spec["ticker"]], read
            )
            record.update(
                {
                    "slug": game["slug"],
                    "status": status,
                    "late_created_metadata": game["created_ms"] > spec["decision_ms"],
                }
            )
            if probability is not None:
                forecasts[spec["ticker"]] = probability
        coverage.append(record)
    return forecasts, coverage
