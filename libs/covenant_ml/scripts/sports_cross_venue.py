"""Compare a frozen second-venue forecast with archived Kalshi decision quotes."""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import narrow_json_to_dict, require_float
from platform_core.run_record import Observation, run_record_sidecar


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Run explicit history capture or entirely offline reproduction.

    Args:
        argv: Command arguments, or process arguments when omitted.
        pin: Arithmetic environment initializer called before numerical imports.

    Returns:
        Zero after writing a complete report and RunRecord.

    Raises:
        FileExistsError: CROSS-009 when either result artifact already exists.
        ValueError: Invalid identities, timestamps or insufficient paired evidence.
        HTTPStatusError: Public acquisition failed.
        OSError: Required evidence could not be read or persisted.
    """
    determinism = pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.archive import ResponseArchive
    from covenant_ml.markets.kalshi import MarketSpec
    from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
    from covenant_ml.markets.reporting import write_market_report
    from covenant_ml.markets.sports_history import (
        TEST_START,
        historical_inventory,
        historical_quote,
        historical_selection,
    )
    from covenant_ml.markets.sports_strategy import score_sports_predictions
    from covenant_ml.markets.types import Quote
    from covenant_ml.markets.us_history import cross_predictions, team_links, us_inventory
    from covenant_ml.markets.validation import read_document

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("kalshi-archive", "us-archive", "team-map", "protocol", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("kalshi-url", "us-url"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--inventory-mode", choices=("capture", "replay"), required=True)
    parser.add_argument("--history-mode", choices=("capture", "replay"), required=True)
    args = parser.parse_args(argv)
    k_root: Path = args.kalshi_archive
    u_root: Path = args.us_archive
    team_map: Path = args.team_map
    protocol: Path = args.protocol
    output: Path = args.out
    k_url: str = args.kalshi_url
    u_url: str = args.us_url
    inventory_mode: str = args.inventory_mode
    history_mode: str = args.history_mode
    if output.exists() or run_record_sidecar(output).exists():
        raise FileExistsError("[CROSS-009] report or RunRecord already exists")
    links = team_links(read_document(team_map))
    rows: list[tuple[MarketSpec, Quote]] = []
    predictions: dict[str, float] = {}
    with closing(_test_hooks.client_factory(60.0, None)) as client:
        k_inventory, k_quotes = (
            ResponseArchive(k_root / name, k_url, client, _test_hooks.clock, _test_hooks.pause)
            for name in ("inventory", "candles")
        )
        u_inventory, u_history = (
            ResponseArchive(u_root / name, u_url, client, _test_hooks.clock, _test_hooks.pause)
            for name in ("inventory", "history")
        )
        specs, selection = historical_selection(k_inventory.replay)
        games, inventory_counts = us_inventory(
            u_inventory.capture if inventory_mode == "capture" else u_inventory.replay
        )
        forecasts, coverage = cross_predictions(
            historical_inventory(k_inventory.replay),
            specs,
            games,
            links,
            u_history.capture if history_mode == "capture" else u_history.replay,
        )
        for spec, coverage_value in zip(specs, coverage, strict=True):
            record = narrow_json_to_dict(coverage_value)
            quote, reason = historical_quote(spec, k_quotes.replay)
            record["kalshi_quote_status"] = reason
            eligible = quote is not None and spec["ticker"] in forecasts
            record["paired_eligible"] = eligible
            if quote is not None and eligible:
                rows.append((spec, quote))
                if spec["day"] >= TEST_START:
                    predictions[spec["ticker"]] = forecasts[spec["ticker"]]
    report = score_sports_predictions(rows, predictions)
    report.update(
        {
            "model": "oriented-us-display-midpoint",
            "exploratory": True,
            "coverage": coverage,
            "selection": dict(selection),
            "us_inventory": dict(inventory_counts),
            "implementation_sha256": implementation_digest(),
        }
    )
    evidence = [team_map, protocol]
    for root, names in ((k_root, ("inventory", "candles")), (u_root, ("inventory", "history"))):
        for name in names:
            evidence.extend(sorted((root / name).glob("*.json")))
    write_market_report(
        report,
        market_fingerprint(determinism, config_test_hooks.get_env),
        output,
        evidence,
        "sports-cross-venue-v1",
        "fixed-forecast_exploratory_2026",
        (Observation(name="adjusted_brier", value=require_float(report, "adjusted_brier")),),
    )
    sys.stdout.write(f"Report: {output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
