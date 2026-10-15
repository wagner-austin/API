"""Acquire or replay a fixed Kalshi family inventory without trade-based selection."""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import narrow_json_to_str, require_int, require_list
from platform_core.run_record import run_record_sidecar


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Write a complete source-bound cohort report with its shared RunRecord.

    Args:
        argv: Command arguments, or process arguments when omitted.
        pin: Arithmetic initializer run before importing numeric dependencies.

    Returns:
        Zero after writing verified acquisition evidence.

    Raises:
        ValueError: COHORT-005 for a future end; plan and native errors propagate.
        FileExistsError: MARKET-027 if report or sidecar already exists.
        OSError: Required evidence cannot be read or written.
        HTTPStatusError: Acquisition failed without producing a partial report.
    """
    determinism = pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.event_cohort import collect_event_cohort, require_cohort_window
    from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
    from covenant_ml.markets.reporting import write_market_report
    from covenant_ml.markets.runner import AcquisitionMode, build_collector
    from covenant_ml.markets.validation import parse_timestamp, read_document

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "archive", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    modes: list[str] = [mode.value for mode in AcquisitionMode]
    parser.add_argument("--mode", required=True, choices=modes)
    parser.add_argument("--base-url", default="https://external-api.kalshi.com/trade-api/v2")
    args = parser.parse_args(argv)
    plan_path: Path = args.plan
    root: Path = args.archive
    output: Path = args.out
    mode: str = args.mode
    base_url: str = args.base_url
    if output.exists() or run_record_sidecar(output).exists():
        raise FileExistsError("[MARKET-027] report or RunRecord exists; choose a new output")
    plan = read_document(plan_path)
    series = tuple(narrow_json_to_str(value) for value in require_list(plan, "series"))
    start, end = require_int(plan, "start_ts"), require_int(plan, "end_ts")
    require_cohort_window(series, start, end)
    if end * 1000 > parse_timestamp(_test_hooks.clock()):
        raise ValueError("[COHORT-005] cohort window must end before acquisition starts")
    digest = implementation_digest()
    collector, client = build_collector(root, base_url, AcquisitionMode(mode))
    with closing(client):
        report = collect_event_cohort(collector, series, start, end)
    report["implementation_sha256"] = digest
    record = write_market_report(
        report,
        market_fingerprint(determinism, config_test_hooks.get_env),
        output,
        (*root.glob("*.json"), plan_path),
        "kalshi-event-cohort-v1",
        f"{start}-{end}",
        ({"name": "family_count", "value": float(len(series))},),
    )
    sys.stdout.write(f"Report: {output}\nRun record: {record}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
