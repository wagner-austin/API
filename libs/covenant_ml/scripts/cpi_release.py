"""Capture or replay a dated official CPI release with complete provenance."""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.run_record import run_record_sidecar


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Persist a verified publication using a declared implementation.

    Args:
        argv: Explicit CLI arguments or process arguments.
        pin: Arithmetic initializer run before numeric imports.

    Returns:
        Zero after a validated release and its RunRecord are persisted.

    Raises:
        ValueError: CPI-011 for implementation drift; release validation failures
            propagate and produce no completed report.
        FileExistsError: CPI-012 for an existing report or sidecar.
        HTTPStatusError: Required public GET failed.
        OSError: Required evidence cannot be read or written.
    """
    determinism = pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.archive import ResponseArchive
    from covenant_ml.markets.bls_collection import collect_cpi_release
    from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
    from covenant_ml.markets.reporting import write_market_report

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("target-month", "published-at", "implementation-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", required=True, choices=("capture", "replay"))
    parser.add_argument("--source-url", default="https://www.bls.gov")
    arguments = parser.parse_args(argv)
    root: Path = arguments.archive
    output: Path = arguments.out
    target: str = arguments.target_month
    published: str = arguments.published_at
    mode: str = arguments.mode
    url: str = arguments.source_url
    expected: str = arguments.implementation_sha256
    actual = implementation_digest()
    if actual != expected:
        raise ValueError("[CPI-011] implementation differs from the declared deployment")
    if output.exists() or run_record_sidecar(output).exists():
        raise FileExistsError("[CPI-012] report or RunRecord already exists")
    with closing(_test_hooks.client_factory(60.0, None)) as client:
        archive = ResponseArchive(root, url, client, _test_hooks.clock, _test_hooks.pause)
        report, sources = collect_cpi_release(
            archive, target_month=target, published_at=published, capture=mode == "capture"
        )
    report["implementation_sha256"] = actual
    sidecar = write_market_report(
        report,
        market_fingerprint(determinism, config_test_hooks.get_env),
        output,
        sources,
        "official-cpi-release-v1",
        target,
        (),
    )
    sys.stdout.write(f"CPI release: {output}\nRun record: {sidecar}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
