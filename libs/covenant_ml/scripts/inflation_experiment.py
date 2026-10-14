"""Compare frozen inflation probabilities with actual historical decision candles."""

from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    require_int,
    require_list,
    require_str,
)
from platform_core.run_record import Observation, run_record_sidecar


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Capture decision quotes or reproduce a complete experiment offline.

    Args:
        argv: Explicit command arguments or the current process arguments.
        pin: Arithmetic initializer executed before numerical imports.

    Returns:
        Zero after persisting the report and source-bound RunRecord.

    Raises:
        FileExistsError: INFLATION-016 if the report or sidecar already exists.
        ValueError: Native validation, model or insufficient evidence failures propagate.
        HTTPStatusError: Quote acquisition failed.
        OSError: Input or artifact persistence failed.
    """
    determinism = pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.archive import ResponseArchive
    from covenant_ml.markets.decision_quotes import decision_quote
    from covenant_ml.markets.inflation_evaluation import evaluate_inflation
    from covenant_ml.markets.inflation_history import decode_inflation_history
    from covenant_ml.markets.inflation_selection import inflation_inventory, inflation_selection
    from covenant_ml.markets.kalshi import MarketSpec
    from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
    from covenant_ml.markets.reporting import write_market_report
    from covenant_ml.markets.types import Quote
    from covenant_ml.markets.validation import read_document

    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "history",
        "historical-inventory",
        "current-inventory",
        "archive",
        "protocol",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--kalshi-url", required=True)
    parser.add_argument("--mode", choices=("capture", "replay"), required=True)
    args = parser.parse_args(argv)
    history_path: Path = args.history
    historical_path: Path = args.historical_inventory
    current_path: Path = args.current_inventory
    archive_path: Path = args.archive
    protocol_path: Path = args.protocol
    output: Path = args.out
    url: str = args.kalshi_url
    mode: str = args.mode
    if output.exists() or run_record_sidecar(output).exists():
        raise FileExistsError("[INFLATION-016] report or RunRecord already exists")
    history = decode_inflation_history(load_json_str(history_path.read_text("utf-8")))
    historical_document = read_document(historical_path)
    records = inflation_inventory((historical_document, read_document(current_path)))
    archived_tickers = {
        require_str(narrow_json_to_dict(value), "ticker")
        for value in require_list(historical_document, "markets")
    }
    selected, coverage = inflation_selection(records, history)
    coverage_by_ticker = {
        require_str(narrow_json_to_dict(value), "ticker"): narrow_json_to_dict(value)
        for value in coverage
    }
    rows: list[tuple[MarketSpec, Quote, float]] = []
    with closing(_test_hooks.client_factory(60.0, None)) as client:
        archive = ResponseArchive(archive_path, url, client, _test_hooks.clock, _test_hooks.pause)
        read = archive.capture if mode == "capture" else archive.replay
        for spec, probability in selected:
            historical = spec["ticker"] in archived_tickers
            quote, reason = decision_quote(spec, read, historical=historical)
            record = coverage_by_ticker[spec["ticker"]]
            record["quote_partition"] = "historical" if historical else "current"
            record["quote_status"] = reason
            record["probability"] = probability
            record["paired_eligible"] = quote is not None
            if quote is not None:
                rows.append((spec, quote, probability))
    report = evaluate_inflation(tuple(rows))
    report["coverage"] = coverage
    report["native_contracts"] = len(records)
    report["selected_contracts"] = len(selected)
    report["implementation_sha256"] = implementation_digest()
    sources = [
        history_path,
        historical_path,
        current_path,
        protocol_path,
        *archive_path.glob("*.json"),
    ]
    observations: tuple[Observation, ...] = (
        {"name": "eligible_contracts", "value": float(require_int(report, "eligible_contracts"))},
    )
    write_market_report(
        report,
        market_fingerprint(determinism, config_test_hooks.get_env),
        output,
        sources,
        "inflation-nowcast-comparison",
        "earlier-sixty-errors-midnight",
        observations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
