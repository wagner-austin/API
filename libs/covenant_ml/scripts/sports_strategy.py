"""Run the frozen sports calibration screen with archived, replayable evidence."""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import require_float
from platform_core.run_record import Observation, run_record_sidecar


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Capture or replay historical evidence and persist a provenance-bound result.

    Args:
        argv: Explicit command arguments or process arguments.
        pin: Arithmetic initializer, called before numerical imports.

    Returns:
        Zero when the full report and RunRecord are persisted.

    Raises:
        FileExistsError: STRATEGY-009 if either output already exists.
        ValueError: Invalid native evidence or insufficient evaluation dates.
        HTTPStatusError: Acquisition failed; no retries or substitute evidence.
        OSError: Required evidence cannot be read or persisted.
    """
    determinism = pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.archive import ResponseArchive
    from covenant_ml.markets.decision_quotes import decision_quote
    from covenant_ml.markets.kalshi import MarketSpec
    from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
    from covenant_ml.markets.reporting import write_market_report
    from covenant_ml.markets.sports_history import (
        historical_inventory,
        historical_selection,
    )
    from covenant_ml.markets.sports_ratings import evaluate_ratings, rating_predictions
    from covenant_ml.markets.sports_strategy import evaluate_sports
    from covenant_ml.markets.types import Quote

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--market-url", required=True)
    parser.add_argument("--mode", choices=("capture", "replay"), required=True)
    parser.add_argument("--model", choices=("temperature", "ratings"), required=True)
    arguments = parser.parse_args(argv)
    root: Path = arguments.archive
    output: Path = arguments.out
    origin: str = arguments.market_url
    mode: str = arguments.mode
    model: str = arguments.model
    if output.exists() or run_record_sidecar(output).exists():
        raise FileExistsError("[STRATEGY-009] report or RunRecord exists; choose a new output")
    rows: list[tuple[MarketSpec, Quote]] = []
    with closing(_test_hooks.client_factory(60.0, None)) as client:
        inventory, quotes = (
            ResponseArchive(root / name, origin, client, _test_hooks.clock, _test_hooks.pause)
            for name in ("inventory", "candles")
        )
        specs, counts = historical_selection(
            inventory.capture if mode == "capture" else inventory.replay
        )
        read = quotes.capture if mode == "capture" else quotes.replay
        for spec in specs:
            quote, state = decision_quote(spec, read, historical=True)
            counts[state] = counts.get(state, 0) + 1
            if quote is not None:
                rows.append((spec, quote))
        if model == "ratings":
            forecasts, rating_counts = rating_predictions(
                historical_inventory(inventory.replay), [spec for spec, _ in rows]
            )
            report = evaluate_ratings(rows, forecasts)
            report["rating_coverage"] = dict(rating_counts)
        else:
            report = evaluate_sports(rows)
    report["coverage"] = dict(counts)
    report["implementation_sha256"] = implementation_digest()
    evidence = sorted((root / "inventory").glob("*.json")) + sorted(
        (root / "candles").glob("*.json")
    )
    write_market_report(
        report,
        market_fingerprint(determinism, config_test_hooks.get_env),
        output,
        evidence,
        "sports-" + model + "-v1",
        "jan-feb-fit_mar-apr15-test_2026",
        (Observation(name="adjusted_brier", value=require_float(report, "adjusted_brier")),),
    )
    sys.stdout.write(f"Report: {output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
