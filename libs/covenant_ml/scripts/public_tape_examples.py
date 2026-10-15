"""Produce source-verified, reproducible examples from disjoint public tape windows."""

from __future__ import annotations

import argparse
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import JSONValue, require_int
from platform_core.run_record import Observation


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Recount verified inputs and persist examples with their source evidence.

    Args:
        argv: Explicit arguments or process arguments.
        pin: Arithmetic initializer invoked before numeric imports.

    Returns:
        Zero after the summary and its RunRecord are written.

    Raises:
        ValueError: TAPE-009 for overlapping windows or altered input evidence.
        FileExistsError: MARKET-027 if the output or sidecar already exists.
        OSError: Required evidence cannot be read or written.
        TypeError: Input schema is malformed.
    """
    determinism = pin()
    from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
    from covenant_ml.markets.reporting import read_verified_market_report, write_market_report
    from covenant_ml.markets.tape_examples import summarize_tape

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", required=True, nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args(argv)
    reports: list[Path] = arguments.reports
    output: Path = arguments.out
    windows: list[tuple[int, int]] = []
    summaries: list[JSONValue] = []
    evidence: set[Path] = set()
    trades = 0
    for path in reports:
        report, sources, digest = read_verified_market_report(path, "TAPE-009")
        summary = summarize_tape(report)
        start, end = (
            require_int(summary, "start_ts_inclusive"),
            require_int(summary, "end_ts_exclusive"),
        )
        if any(
            start < previous_end and previous_start < end
            for previous_start, previous_end in windows
        ):
            raise ValueError("[TAPE-009] input windows overlap; totals would duplicate evidence")
        windows.append((start, end))
        summary["input_sha256"] = digest
        summary["input_path"] = str(path)
        summaries.append(summary)
        evidence.update(sources)
        trades += require_int(summary, "trade_count")
    observations: tuple[Observation, ...] = (
        {"name": "trade_count", "value": float(trades)},
        {"name": "window_count", "value": float(len(windows))},
    )
    write_market_report(
        {
            "schema_version": 1,
            "implementation_sha256": implementation_digest(),
            "trade_count": trades,
            "window_count": len(windows),
            "windows": summaries,
        },
        market_fingerprint(determinism, config_test_hooks.get_env),
        output,
        tuple(evidence),
        "kalshi-public-tape-examples-v1",
        "disjoint-windows",
        observations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
