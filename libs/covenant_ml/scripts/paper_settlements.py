"""Reconcile immutable prospective predictions against actual public market settlements."""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from hashlib import sha256
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import require_int
from platform_core.run_record import Observation


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Verify original prediction evidence and persist an explicit settlement snapshot.

    Args:
        argv: Explicit command arguments or process arguments.
        pin: Numeric determinism initializer, called before numeric imports.

    Returns:
        Zero after the settlement report and RunRecord are persisted, including
        valid pending reports with no performance score.

    Raises:
        ValueError: Original evidence, native identity or chronology is invalid.
        FileExistsError: SETTLEMENT-007 for an existing capture directory, or output collision.
        HTTPStatusError: A required public request failed without retry.
        OSError: Evidence cannot be read or persisted.
    """
    determinism = pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.archive import ResponseArchive
    from covenant_ml.markets.paper_scoring import load_verified_predictions, reconcile_predictions
    from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
    from covenant_ml.markets.reporting import write_market_report

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("predictions", "archive", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("capture", "replay"))
    parser.add_argument("--market-url", default="https://external-api.kalshi.com/trade-api/v2")
    args = parser.parse_args(argv)
    prediction_path: Path = args.predictions
    archive_path: Path = args.archive
    output: Path = args.out
    mode: str = args.mode
    url: str = args.market_url
    predictions, sources = load_verified_predictions(prediction_path)
    capture = mode == "capture"
    if capture and archive_path.exists():
        raise FileExistsError("[SETTLEMENT-007] capture needs a new settlement archive")
    fingerprint = market_fingerprint(determinism, config_test_hooks.get_env)
    with closing(_test_hooks.client_factory(60.0, None)) as client:
        archive = ResponseArchive(archive_path, url, client, _test_hooks.clock, _test_hooks.pause)
        report = reconcile_predictions(predictions, archive, capture)
    digest = sha256(prediction_path.read_bytes()).hexdigest()
    report["prediction_report_sha256"] = digest
    report["implementation_sha256"] = implementation_digest()
    observations = tuple(
        Observation(name=key, value=float(require_int(report, key)))
        for key in ("selected", "pending", "scalar", "rules_changed", "eligible_binary")
    )
    sidecar = write_market_report(
        report,
        fingerprint,
        output,
        (*sources, *archive_path.glob("*.json")),
        "prospective-market-settlements",
        digest,
        observations,
    )
    sys.stdout.write(f"Settlements: {output}\nRun record: {sidecar}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
