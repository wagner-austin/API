"""Capture, explicitly resume, or replay a frozen Kalshi public execution window."""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import require_int
from platform_core.run_record import Observation, run_record_sidecar


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Persist a complete public tape and a source-bound RunRecord.

    Args:
        argv: CLI arguments or process arguments when omitted.
        pin: Arithmetic initializer called before numeric imports.

    Returns:
        Zero after the report and source hashes have been written.

    Raises:
        ValueError: TAPE-006 for a future window; collection quality errors propagate.
        FileExistsError: MARKET-027 if output or its sidecar already exists.
        OSError: Evidence read or exclusive output write fails.
        HTTPStatusError: Public acquisition fails without emitting a partial report.
    """
    determinism = pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
    from covenant_ml.markets.public_tape import collect_public_tape
    from covenant_ml.markets.reporting import write_market_report
    from covenant_ml.markets.runner import AcquisitionMode, build_collector
    from covenant_ml.markets.validation import parse_timestamp, read_document

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "archive", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    modes: list[str] = [mode.value for mode in AcquisitionMode]
    parser.add_argument("--mode", required=True, choices=modes)
    parser.add_argument("--base-url", default="https://external-api.kalshi.com/trade-api/v2")
    arguments = parser.parse_args(argv)
    plan_path: Path = arguments.plan
    archive: Path = arguments.archive
    output: Path = arguments.out
    mode: str = arguments.mode
    base_url: str = arguments.base_url
    if output.exists() or run_record_sidecar(output).exists():
        raise FileExistsError("[MARKET-027] report or RunRecord exists; choose a new output")
    plan = read_document(plan_path)
    start, end = require_int(plan, "start_ts"), require_int(plan, "end_ts")
    if end * 1000 > parse_timestamp(_test_hooks.clock()):
        raise ValueError("[TAPE-006] window must end before acquisition starts")
    source_digest = implementation_digest()
    collector, client = build_collector(archive, base_url, AcquisitionMode(mode))
    with closing(client):
        report = collect_public_tape(collector, start, end)
    report["implementation_sha256"] = source_digest
    observations: tuple[Observation, ...] = tuple(
        {"name": name, "value": float(require_int(report, name))}
        for name in ("trade_count", "contract_count", "event_count")
    )
    record = write_market_report(
        report,
        market_fingerprint(determinism, config_test_hooks.get_env),
        output,
        (*archive.glob("*.json"), plan_path),
        "kalshi-public-tape-v1",
        f"{start}-{end}",
        observations,
    )
    sys.stdout.write(f"Report: {output}\nRun record: {record}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
