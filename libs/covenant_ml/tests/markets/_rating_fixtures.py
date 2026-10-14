"""Native two-contract games with stable team identities for rating validation."""

from __future__ import annotations

from platform_core.json_utils import JSONObject

from ._strategy_fixtures import native_market


def game(stamp: str, winner: str = "CHI") -> tuple[JSONObject, JSONObject]:
    left, right = native_market(stamp, "CHI"), native_market(stamp, "DAL")
    left["custom_strike"] = {"hockey_team": "chicago-id"}
    right["custom_strike"] = {"hockey_team": "dallas-id"}
    left["settlement_value_dollars"] = "1.0000" if winner == "CHI" else "0.0000"
    right["settlement_value_dollars"] = "0.0000" if winner == "CHI" else "1.0000"
    return left, right


def warmup() -> list[JSONObject]:
    return [native for day in range(1, 11) for native in game(f"25OCT{day:02d}")]
