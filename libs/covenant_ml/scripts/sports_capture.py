"""Capture repeated declared sports pairs, or reproduce them with no network access."""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_int,
    require_list,
)
from platform_core.run_record import Observation, run_record_sidecar


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Write a provenance-bound repeated sports snapshot report.

    Args:
        argv: Explicit arguments or process arguments.
        pin: Existing arithmetic environment initializer.

    Returns:
        Zero after report and RunRecord persistence.

    Raises:
        ValueError: SPORTS-007 for invalid selection or repetition configuration;
            native identity, rule or receipt validation failures propagate.
        OSError: Required evidence cannot be read or exclusively written.
        HTTPStatusError: A public request failed.
    """
    determinism = pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.archive import ResponseArchive
    from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
    from covenant_ml.markets.reporting import write_market_report
    from covenant_ml.markets.sports_collection import collect_sports_pair
    from covenant_ml.markets.sports_records import decode_sports_pair
    from covenant_ml.markets.validation import parse_timestamp, read_document

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "archive", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("kalshi-url", "us-url", "window-start", "window-end"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=("capture", "replay"), required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--interval-seconds", type=int, required=True)
    parser.add_argument("--max-skew-ms", type=int, required=True)
    arguments = parser.parse_args(argv)
    selection: Path = arguments.selection
    root: Path = arguments.archive
    output: Path = arguments.out
    k_url: str = arguments.kalshi_url
    u_url: str = arguments.us_url
    start_text: str = arguments.window_start
    end_text: str = arguments.window_end
    mode: str = arguments.mode
    capture = mode == "capture"
    rounds: int = arguments.rounds
    interval: int = arguments.interval_seconds
    max_skew: int = arguments.max_skew_ms
    if output.exists() or run_record_sidecar(output).exists():
        raise FileExistsError("[SPORTS-009] report or RunRecord exists; choose a new output")
    start, end = parse_timestamp(start_text), parse_timestamp(end_text)
    pairs = tuple(
        decode_sports_pair(narrow_json_to_dict(row))
        for row in require_list(read_document(selection), "pairs")
    )
    if (
        not pairs
        or len({pair["kalshi_ticker"] for pair in pairs}) != len(pairs)
        or len({pair["us_slug"] for pair in pairs}) != len(pairs)
        or not 1 <= rounds <= 1000
        or not 0 <= interval <= 60
        or not 0 <= start < end
        or max_skew <= 0
    ):
        raise ValueError("[SPORTS-007] invalid unique selection, window or repetition settings")
    snapshots: list[JSONValue] = []
    evidence = [selection]
    previous = start
    with closing(_test_hooks.client_factory(30.0, None)) as client:
        for index in range(rounds):
            if capture and index > 0:
                _test_hooks.pause(interval)
            stores = tuple(
                ResponseArchive(
                    root / str(index) / venue, url, client, _test_hooks.clock, _test_hooks.pause
                )
                for venue, url in (("kalshi", k_url), ("us", u_url))
            )
            for pair in pairs:
                snapshot = collect_sports_pair(
                    pair, stores[0], stores[1], capture, previous, end, max_skew
                )
                previous = require_int(snapshot, "last_received_ms")
                snapshot["round"] = index
                snapshots.append(snapshot)
            for store in stores:
                evidence.extend(sorted(store.root.glob("*.json")))
    report: JSONObject = {
        "schema_version": 1,
        "window_start": start_text,
        "window_end": end_text,
        "max_skew_ms": max_skew,
        "rounds": rounds,
        "interval_seconds": interval,
        "snapshots": snapshots,
        "implementation_sha256": implementation_digest(),
        "orders_submitted": 0,
        "interpretation": "Gross quote screens only; fees and settlement equivalence unverified.",
    }
    write_market_report(
        report,
        market_fingerprint(determinism, config_test_hooks.get_env),
        output,
        evidence,
        "sports-cross-venue-v1",
        str(selection),
        (Observation(name="snapshots", value=float(len(snapshots))),),
    )
    sys.stdout.write(f"Report: {output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
