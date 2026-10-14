"""Sequential team ratings from outcomes known before each quoted decision."""

from __future__ import annotations

from collections.abc import Sequence
from math import fsum, pow

from platform_core.json_utils import JSONObject, require_dict, require_str

from .kalshi import MarketSpec
from .sports_history import TEST_END, event_day
from .sports_strategy import score_sports_predictions, split_sports
from .types import DOLLAR, Quote
from .validation import parse_fixed, parse_timestamp, require_nonempty

BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _native_key(native: JSONObject) -> str:
    """Order native team contracts before reading their outcomes.

    Args:
        native: Required native metadata.

    Returns:
        Nonempty ticker string.
    """
    return require_nonempty(native, "ticker")


def _games(
    records: Sequence[JSONObject],
) -> tuple[list[tuple[int, str, str, str, str, int]], dict[str, int]]:
    """Reconcile complementary team contracts into single available game outcomes.

    Args:
        records: Native archived market inventory.

    Returns:
        Settlement-ordered games and explicit coverage counts.

    Raises:
        ValueError: RATING-001 for duplicated tickers or team identities;
            RATING-002 for a settlement preceding market close.
        TypeError: Required native fields are absent or malformed.
    """
    grouped: dict[str, list[JSONObject]] = {}
    seen: set[str] = set()
    for native in records:
        event, ticker = require_str(native, "event_ticker"), _native_key(native)
        if ticker in seen or not ticker.startswith(event + "-"):
            raise ValueError("[RATING-001] duplicate or mismatched native identity")
        seen.add(ticker)
        if "2025-10-01" <= event_day(event) <= TEST_END:
            grouped.setdefault(event, []).append(native)
    games: list[tuple[int, str, str, str, str, int]] = []
    counts = {"inventory_games": len(grouped), "incomplete_pairs": 0, "nonbinary_games": 0}
    for event, pair in grouped.items():
        if len(pair) != 2:
            counts["incomplete_pairs"] += 1
            continue
        first, second = sorted(pair, key=_native_key)
        left, right = (
            require_nonempty(require_dict(native, "custom_strike"), "hockey_team")
            for native in (first, second)
        )
        if left == right:
            raise ValueError("[RATING-001] opposing contracts name the same team")
        payouts = tuple(
            parse_fixed(require_str(native, "settlement_value_dollars"), 4)
            for native in (first, second)
        )
        if set(payouts) != {0, DOLLAR} or any(
            require_str(native, "status") != "finalized" for native in (first, second)
        ):
            counts["nonbinary_games"] += 1
            continue
        settled = tuple(
            parse_timestamp(require_str(native, "settlement_ts")) for native in (first, second)
        )
        if any(
            instant < parse_timestamp(require_str(native, "close_time"))
            for instant, native in zip(settled, (first, second), strict=True)
        ):
            raise ValueError("[RATING-002] native settlement precedes close")
        games.append((max(settled), event, _native_key(first), left, right, payouts[0]))
    return sorted(games), counts


def _decision_key(spec: MarketSpec) -> tuple[int, str]:
    """Order predictions by their information cutoff.

    Args:
        spec: Selected observation's market specification.

    Returns:
        Decision timestamp and ticker.
    """
    return spec["decision_ms"], spec["ticker"]


def rating_predictions(
    records: Sequence[JSONObject], specs: Sequence[MarketSpec]
) -> tuple[dict[str, float], dict[str, int]]:
    """Compute frozen K=20 team forecasts using only already settled games.

    Args:
        records: Native inventory, including earlier rating warmup games.
        specs: Selected games at their declared quote decision times.

    Returns:
        Forecasts with ten prior games per team and coverage/exclusion counts.

    Raises:
        ValueError: RATING-003 for duplicate or inconsistent targets; invalid
            native pairs propagate their own validation failures.
    """
    games, counts = _games(records)
    teams = {team for _, _, _, left, right, _ in games for team in (left, right)}
    ratings = dict.fromkeys(teams, 1500.0)
    completed = dict.fromkeys(teams, 0)
    by_event = {
        event: (ticker, left, right, settled) for settled, event, ticker, left, right, _ in games
    }
    predictions: dict[str, float] = {}
    seen: set[str] = set()
    counts.update({"unavailable_target": 0, "insufficient_history": 0, "eligible_forecasts": 0})
    index = 0
    for spec in sorted(specs, key=_decision_key):
        event = spec["ticker"].rsplit("-", 1)[0]
        if event in seen:
            raise ValueError("[RATING-003] duplicate target game")
        seen.add(event)
        while index < len(games) and games[index][0] < spec["decision_ms"]:
            _, _, _, left, right, payout = games[index]
            expected = 1 / (1 + pow(10.0, (ratings[right] - ratings[left]) / 400))
            change = 20 * (payout / DOLLAR - expected)
            ratings[left] += change
            ratings[right] -= change
            completed[left] += 1
            completed[right] += 1
            index += 1
        if event not in by_event:
            counts["unavailable_target"] += 1
            continue
        ticker, left, right, settled = by_event[event]
        if ticker != spec["ticker"] or settled <= spec["decision_ms"]:
            raise ValueError("[RATING-003] target identity or settlement chronology mismatch")
        if min(completed[left], completed[right]) < 10:
            counts["insufficient_history"] += 1
            continue
        predictions[ticker] = 1 / (1 + pow(10.0, (ratings[right] - ratings[left]) / 400))
        counts["eligible_forecasts"] += 1
    return predictions, counts


def _blend(quote: Quote, rating: float, weight: float) -> float:
    """Combine an independent team forecast with a validated market midpoint.

    Args:
        quote: Validated non-null bid and ask from the shared splitter.
        rating: Team probability in the unit interval.
        weight: Declared mixture weight.

    Returns:
        Convexly blended probability.

    Raises:
        ValueError: RATING-004 for missing quote sides or invalid probabilities.
    """
    bid, ask = quote["bid"], quote["ask"]
    if bid is None or ask is None or not 0 <= rating <= 1:
        raise ValueError("[RATING-004] blend needs quoted sides and a valid team probability")
    return (1 - weight) * (bid + ask) / (2 * DOLLAR) + weight * rating


def evaluate_ratings(
    rows: Sequence[tuple[MarketSpec, Quote]], forecasts: dict[str, float]
) -> JSONObject:
    """Fit the declared blend on earlier games and reuse later-game accounting.

    Args:
        rows: Eligible quotes and resolved outcomes.
        forecasts: Earlier-outcome-only team predictions keyed by ticker.

    Returns:
        Primary blended strategy, pure-rating diagnostic and training selection.

    Raises:
        ValueError: RATING-005 for unknown forecast tickers; insufficient
            chronological coverage or invalid probabilities propagate.
    """
    if not set(forecasts) <= {spec["ticker"] for spec, _ in rows}:
        raise ValueError("[RATING-005] forecast names a ticker outside the selected quotes")
    eligible = tuple((spec, quote) for spec, quote in rows if spec["ticker"] in forecasts)
    train, test = split_sports(eligible)
    scores = {
        weight: fsum(
            (_blend(quote, forecasts[spec["ticker"]], weight) - spec["settlement"] / DOLLAR) ** 2
            for spec, quote in train
        )
        / len(train)
        for weight in BLEND_WEIGHTS
    }
    _, weight = min((score, weight) for weight, score in scores.items())
    primary = {
        spec["ticker"]: _blend(quote, forecasts[spec["ticker"]], weight) for spec, quote in test
    }
    pure = {spec["ticker"]: forecasts[spec["ticker"]] for spec, _ in test}
    report = score_sports_predictions(eligible, primary)
    report["weight"] = weight
    report["training_brier"] = {str(key): value for key, value in scores.items()}
    report["pure_rating_diagnostic"] = score_sports_predictions(eligible, pure)
    report["study_design"] = (
        "Exploratory reuse of previously examined dates; not untouched confirmation."
    )
    return report
