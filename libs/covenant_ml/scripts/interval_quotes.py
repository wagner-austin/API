"""Acquire source-bound crypto quote attachments or replay their exact archives."""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import require_int
from platform_core.run_record import run_record_sidecar


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Verify the input cohort and persist all quote evidence with a RunRecord.

    Args:
        argv: Command arguments or process arguments when omitted.
        pin: Arithmetic initializer executed before numeric imports.

    Returns:
        Zero after a complete report and sidecar are written.

    Raises:
        ValueError: IQUOTE-006 for changed provenance; other validation errors propagate.
        FileExistsError: MARKET-027 if either output already exists.
        HTTPStatusError: Acquisition failed; no partial report is emitted.
    """
    determinism = pin()
    from covenant_ml.markets.interval_quotes import collect_interval_quotes
    from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
    from covenant_ml.markets.reporting import read_verified_market_report, write_market_report
    from covenant_ml.markets.runner import AcquisitionMode, build_collector
    from covenant_ml.markets.validation import read_document

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "plan", "archive", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    modes: list[str] = [mode.value for mode in AcquisitionMode]
    parser.add_argument("--mode", required=True, choices=modes)
    parser.add_argument("--base-url", default="https://external-api.kalshi.com/trade-api/v2")
    args = parser.parse_args(argv)
    source: Path = args.cohort
    plan_path: Path = args.plan
    root: Path = args.archive
    output: Path = args.out
    mode: str = args.mode
    base_url: str = args.base_url
    if output.exists() or run_record_sidecar(output).exists():
        raise FileExistsError("[MARKET-027] report or RunRecord exists; choose a new output")
    cohort, paths, cohort_digest = read_verified_market_report(source, "IQUOTE-006")
    plan = read_document(plan_path)
    digest = implementation_digest()
    collector, client = build_collector(root, base_url, AcquisitionMode(mode))
    with closing(client):
        report = collect_interval_quotes(collector, cohort, plan)
    report.update(implementation_sha256=digest, cohort_sha256=cohort_digest)
    record = write_market_report(
        report,
        market_fingerprint(determinism, config_test_hooks.get_env),
        output,
        (*root.glob("*.json"), plan_path, *paths),
        "kalshi-interval-quotes-v1",
        f"{require_int(plan, 'start_ts')}-{require_int(plan, 'end_ts')}",
        ({"name": "contract_count", "value": float(require_int(report, "contract_count"))},),
    )
    sys.stdout.write(f"Report: {output}\nRun record: {record}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
