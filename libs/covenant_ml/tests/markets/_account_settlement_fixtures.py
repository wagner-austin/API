"""Native settlement documents consumed by actual codecs and socket services."""

from __future__ import annotations

from platform_core.json_utils import JSONObject


def native_credit(ticker: str = "MARKET-1", payout: int = 100) -> JSONObject:
    return {
        "ticker": ticker,
        "event_ticker": "EVENT-1",
        "market_result": "yes" if payout == 100 else "no" if payout == 0 else "scalar",
        "yes_count_fp": "1.00",
        "yes_total_cost_dollars": "0.9800",
        "no_count_fp": "0.00",
        "no_total_cost_dollars": "0.0000",
        "revenue": payout,
        "fee_cost": "0.0200",
        "settled_time": "1970-01-01T00:02:00Z",
    }


def final_market(ticker: str = "MARKET-1", payout: int = 100) -> JSONObject:
    return {
        "ticker": ticker,
        "event_ticker": "EVENT-1",
        "status": "finalized",
        "result": "yes" if payout == 100 else "no" if payout == 0 else "scalar",
        "settlement_value_dollars": f"{payout // 100}.{payout % 100:02d}00",
        "settlement_ts": "1970-01-01T00:02:00Z",
    }
