"""Run the fixed chronological boosting comparison from verified offline evidence."""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from hashlib import sha256
from pathlib import Path

from platform_core.comparability import RunFingerprint
from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.environment_record import capture_package_versions
from platform_core.json_utils import require_dict, require_float
from platform_core.run_record import Observation, run_record_sidecar


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Verify frozen identities, fit earlier labels and persist all later predictions.

    Args:
        argv: Explicit command arguments or process arguments.
        pin: CPU initializer, called before numerical imports.

    Returns:
        Zero after the report, native models and RunRecord are saved.

    Raises:
        ValueError: BOOST-005 for mismatched source or protocol identity.
        FileExistsError: BOOST-006 if output evidence already exists.
        OSError: An input or output cannot be accessed.
    """
    determinism = pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.archive import ResponseArchive
    from covenant_ml.markets.boosting_evaluation import evaluate_boosting
    from covenant_ml.markets.calibration_evidence import load_calibration_evidence
    from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
    from covenant_ml.markets.reporting import write_market_report

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "markets-archive", "protocol", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("market-url", "source-sha256", "protocol-sha256", "first-fold-end"):
        parser.add_argument("--" + name, required=True)
    arguments = parser.parse_args(argv)
    source: Path = arguments.report
    markets: Path = arguments.markets_archive
    protocol: Path = arguments.protocol
    output_arg: Path = arguments.out
    output = output_arg.resolve()
    origin: str = arguments.market_url
    expected_source: str = arguments.source_sha256
    expected_protocol: str = arguments.protocol_sha256
    first_fold: str = arguments.first_fold_end
    if output.exists() or run_record_sidecar(output).exists():
        raise FileExistsError("[BOOST-006] output evidence already exists")
    protocol_bytes = protocol.read_bytes()
    if sha256(protocol_bytes).hexdigest() != expected_protocol:
        raise ValueError("[BOOST-005] protocol differs from frozen digest")
    with closing(_test_hooks.client_factory(60.0, None)) as client:
        archive = ResponseArchive(markets, origin, client, _test_hooks.clock, _test_hooks.pause)
        plan, rows, _, sources, digest = load_calibration_evidence(source, archive)
    if digest != expected_source:
        raise ValueError("[BOOST-005] source report differs from frozen digest")
    artifacts = output.with_suffix(".models")
    report = evaluate_boosting(rows, plan, (first_fold, plan["end"]), artifacts)
    protocol_copy = artifacts / "protocol.md"
    with protocol_copy.open("xb") as destination:
        destination.write(protocol_bytes)
    report["source_report_sha256"] = digest
    report["protocol_sha256"] = expected_protocol
    report["implementation_sha256"] = implementation_digest()
    base = market_fingerprint(determinism, config_test_hooks.get_env)
    fingerprint: RunFingerprint = {
        **base,
        "packages": (
            *base["packages"],
            *capture_package_versions(("lightgbm",), _test_hooks.installed_version),
        ),
    }
    arms = require_dict(require_dict(report, "scores"), "arms")
    observations = tuple(
        Observation(
            name=name + "_date_brier",
            value=require_float(require_dict(arms, name), "date_mean_brier"),
        )
        for name in sorted(arms)
    )
    paths = (*sources, *artifacts.glob("*.txt"), protocol_copy)
    record = write_market_report(
        report,
        fingerprint,
        output,
        tuple(path.resolve() for path in paths),
        "fixed-forecast-boosting-v1",
        expected_protocol,
        observations,
    )
    sys.stdout.write(f"Report: {output}\nRun record: {record}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
