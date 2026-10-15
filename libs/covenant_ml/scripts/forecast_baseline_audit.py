"""Verify saved baseline forecasts and publish an offline paired evidence audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import require_dict, require_float
from platform_core.run_record import Observation


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Audit a pinned historical report without acquiring data or fitting models.

    Args:
        argv: Explicit command arguments or process arguments.
        pin: Numeric initializer before numerical imports.

    Returns:
        Zero after the report and shared RunRecord are persisted.

    Raises:
        ValueError: BASELINE-001 for mismatched provenance, or audited evidence errors.
        OSError: Required evidence cannot be read or written.
        FileExistsError: The requested output already exists.
    """
    determinism = pin()
    from covenant_ml.markets.baseline_audit import audit_baseline
    from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
    from covenant_ml.markets.reporting import read_verified_market_report, write_market_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    source: Path = args.report
    expected: str = args.source_sha256
    output: Path = args.out
    document, paths, digest = read_verified_market_report(source.resolve(), "BASELINE-001")
    if digest != expected:
        raise ValueError("[BASELINE-001] source differs from the explicitly pinned report")
    report = audit_baseline(document)
    report["source_report_sha256"] = digest
    report["implementation_sha256"] = implementation_digest()
    contrast = require_dict(report, "logistic_minus_raw_market")
    sidecar = write_market_report(
        report,
        market_fingerprint(determinism, config_test_hooks.get_env),
        output.resolve(),
        paths,
        "historical-baseline-audit-v1",
        digest,
        (
            Observation(
                name="logistic_minus_raw_market_brier", value=require_float(contrast, "mean")
            ),
        ),
    )
    sys.stdout.write(f"Report: {output.resolve()}\nRun record: {sidecar}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
