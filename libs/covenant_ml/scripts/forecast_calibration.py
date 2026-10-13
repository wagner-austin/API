"""Fit and evaluate city weather calibration from verified offline historical evidence."""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import require_dict, require_float
from platform_core.run_record import Observation


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Persist the complete exploratory experiment with all original source hashes.

    Args:
        argv: Explicit command arguments or process arguments.
        pin: CPU determinism initializer called before numerical imports.

    Returns:
        Zero after report and RunRecord have been written.

    Raises:
        ValueError: Invalid historical evidence, dates or calibration inputs.
        OSError: Evidence cannot be read or output already exists.
    """
    determinism = pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.archive import ResponseArchive
    from covenant_ml.markets.calibration_evaluation import evaluate_city_calibration
    from covenant_ml.markets.calibration_evidence import load_calibration_evidence
    from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
    from covenant_ml.markets.reporting import write_market_report

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "markets-archive", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--market-url", required=True)
    arguments = parser.parse_args(argv)
    source: Path = arguments.report
    markets: Path = arguments.markets_archive
    output: Path = arguments.out
    origin: str = arguments.market_url
    with closing(_test_hooks.client_factory(60.0, None)) as client:
        archive = ResponseArchive(markets, origin, client, _test_hooks.clock, _test_hooks.pause)
        plan, rows, bounds, sources, digest = load_calibration_evidence(source, archive)
    report = evaluate_city_calibration(rows, plan, bounds)
    report["source_report_sha256"] = digest
    report["implementation_sha256"] = implementation_digest()
    arms = require_dict(require_dict(report, "city_calibrated"), "arms")
    observations = tuple(
        Observation(name=name + "_brier", value=require_float(require_dict(arms, name), "brier"))
        for name in sorted(arms)
    )
    record = write_market_report(
        report,
        market_fingerprint(determinism, config_test_hooks.get_env),
        output,
        sources,
        "city-weather-calibration-v1",
        digest,
        observations,
    )
    sys.stdout.write(f"Report: {output}\nRun record: {record}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
