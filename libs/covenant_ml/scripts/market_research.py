"""Capture or replay a frozen market study and emit its auditable RunRecord.

Usage:
    poetry run python -m scripts.market_research --plan plan.json \
        --protocol protocol.json --archive raw --mode capture --out results.json --workers 3
"""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from hashlib import sha256
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import load_json_str, narrow_json_to_dict


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Run the full public-data replication pipeline with explicit acquisition mode.

    Args:
        argv: CLI arguments, or process arguments when omitted.
        pin: CPU determinism initializer, called before importing numeric code.

    Returns:
        Zero only after the full report and run record have been persisted.

    Raises:
        ValueError: Invalid experiment configuration or source evidence.
        HTTPStatusError: Failed acquisition; no final result is emitted.
        OSError: Source read or persistence failure.
    """
    determinism = pin()
    from covenant_ml.markets.experiment import decode_plan, evaluate_panel, write_report
    from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
    from covenant_ml.markets.runner import AcquisitionMode, build_collector, collect_panel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", required=True, type=int)
    modes: list[str] = [mode.value for mode in AcquisitionMode]
    parser.add_argument("--mode", required=True, choices=modes)
    parser.add_argument("--base-url", default="https://external-api.kalshi.com/trade-api/v2")
    arguments = parser.parse_args(argv)
    plan_path: Path = arguments.plan
    protocol_path: Path = arguments.protocol
    archive: Path = arguments.archive
    output: Path = arguments.out
    workers: int = arguments.workers
    mode: str = arguments.mode
    base_url: str = arguments.base_url
    plan = decode_plan(narrow_json_to_dict(load_json_str(plan_path.read_text("utf-8"))))
    if sha256(protocol_path.read_bytes()).hexdigest() != plan["protocol_digest"]:
        raise ValueError("[MARKET-031] protocol file does not match the frozen plan digest")
    fingerprint = market_fingerprint(determinism, config_test_hooks.get_env)
    source_digest = implementation_digest()
    collector, client = build_collector(archive, base_url, AcquisitionMode(mode))
    with closing(client):
        histories = collect_panel(plan, collector, workers, print)
    report = evaluate_panel(histories, plan)
    report["implementation_sha256"] = source_digest
    record = write_report(
        report, plan, fingerprint, output, (*archive.glob("*.json"), plan_path, protocol_path)
    )
    sys.stdout.write(f"Report: {output}\nRun record: {record}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
