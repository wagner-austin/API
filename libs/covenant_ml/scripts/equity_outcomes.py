"""Capture or replay complete price evidence for a frozen stock paper portfolio."""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.run_record import Observation, run_record_sidecar


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Verify the frozen portfolio and persist an auditable price observation.

    Args:
        argv: Explicit CLI arguments or process arguments.
        pin: Arithmetic initializer run before numeric imports.

    Returns:
        Zero after report and RunRecord persistence, including explicit pending.

    Raises:
        ValueError: Invalid snapshot, calendar, receipt or native price evidence.
        FileExistsError: EQUITY-020 for an existing output; capture also refuses
            an existing observation directory.
        HTTPStatusError: Required public GET failed.
        OSError: Required evidence cannot be read or written.
    """
    determinism = pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.archive import ResponseArchive
    from covenant_ml.markets.equity_collection import collect_equity_outcomes
    from covenant_ml.markets.equity_study import load_equity_study
    from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
    from covenant_ml.markets.reporting import write_market_report
    from covenant_ml.markets.validation import read_document

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "calendar", "archive", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("capture", "replay"))
    parser.add_argument("--price-url", default="https://query1.finance.yahoo.com")
    arguments = parser.parse_args(argv)
    snapshot: Path = arguments.snapshot
    calendar: Path = arguments.calendar
    root: Path = arguments.archive
    output: Path = arguments.out
    mode: str = arguments.mode
    url: str = arguments.price_url
    if output.exists() or run_record_sidecar(output).exists():
        raise FileExistsError("[EQUITY-020] report or RunRecord already exists")
    study, sources = load_equity_study(snapshot, read_document(calendar))
    with closing(_test_hooks.client_factory(60.0, None)) as client:
        archive = ResponseArchive(root, url, client, _test_hooks.clock, _test_hooks.pause)
        report, captures = collect_equity_outcomes(study, archive, mode == "capture")
    report["implementation_sha256"] = implementation_digest()
    sidecar = write_market_report(
        report,
        market_fingerprint(determinism, config_test_hooks.get_env),
        output,
        (*sources, calendar, *captures),
        "equity-paper-outcomes-v1",
        study["snapshot_sha256"],
        (Observation(name="cohort", value=float(len(study["cohort"]))),),
    )
    sys.stdout.write(f"Equity outcomes: {output}\nRun record: {sidecar}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
