"""Run the source-bound fixed crypto development comparison without network or orders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import require_int, require_str
from platform_core.run_record import run_record_sidecar


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Verify both complete source chains and write a reproducible comparison report.

    Args:
        argv: Explicit command arguments or process arguments.
        pin: Arithmetic initialization before learner imports.

    Returns:
        Zero after writing the complete report and shared RunRecord.

    Raises:
        ValueError: IFORE-005 for changed provenance or input linkage.
        FileExistsError: MARKET-027 if either output already exists.
        OSError: Required evidence cannot be read or written.
    """
    determinism = pin()
    from covenant_ml.markets.interval_comparison import compare_intervals
    from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
    from covenant_ml.markets.reporting import read_verified_market_report, write_market_report
    from covenant_ml.markets.validation import read_document

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "quotes", "plan", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    cohort_path: Path = args.cohort
    quotes_path: Path = args.quotes
    plan_path: Path = args.plan
    output: Path = args.out
    if output.exists() or run_record_sidecar(output).exists():
        raise FileExistsError("[MARKET-027] report or RunRecord exists; choose a new output")
    cohort, cohort_sources, cohort_hash = read_verified_market_report(cohort_path, "IFORE-005")
    quotes, quote_sources, quote_hash = read_verified_market_report(quotes_path, "IFORE-005")
    plan = read_document(plan_path)
    if (
        require_str(quotes, "cohort_sha256") != cohort_hash
        or require_str(plan, "cohort_sha256") != cohort_hash
        or require_str(plan, "quotes_sha256") != quote_hash
    ):
        raise ValueError("[IFORE-005] declared cohort and quote hashes must match their inputs")
    report = compare_intervals(cohort, quotes, plan)
    report["implementation_sha256"] = implementation_digest()
    record = write_market_report(
        report,
        market_fingerprint(determinism, config_test_hooks.get_env),
        output,
        tuple(sorted({*cohort_sources, *quote_sources, plan_path})),
        "kalshi-interval-comparison-v1",
        "fixed-development",
        ({"name": "evaluation_count", "value": float(require_int(report, "evaluation_count"))},),
    )
    sys.stdout.write(f"Report: {output}\nRun record: {record}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
