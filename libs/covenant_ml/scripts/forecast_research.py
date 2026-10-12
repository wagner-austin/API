"""Run the frozen chronological forecast comparison against verified market archives."""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from hashlib import sha256
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import (
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_float,
)
from platform_core.run_record import Observation


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Collect explicit-run forecasts, fit earlier dates and report held-out results.

    Args:
        argv: Explicit command arguments or process arguments.
        pin: CPU determinism initializer, called before importing numeric code.

    Returns:
        Zero after report and provenance sidecar are persisted.

    Raises:
        ValueError: FORECAST-010 for protocol mismatch or invalid source evidence.
        HTTPStatusError: Forecast acquisition fails without retry or substitution.
        OSError: Required archive files or output cannot be accessed.
    """
    determinism = pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.archive import ResponseArchive
    from covenant_ml.markets.forecast_collection import collect_forecast_rows
    from covenant_ml.markets.forecast_evaluation import evaluate_forecasts
    from covenant_ml.markets.forecast_records import decode_comparison_plan, encode_comparison_plan
    from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
    from covenant_ml.markets.reporting import write_market_report
    from covenant_ml.markets.runner import AcquisitionMode, build_collector

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "protocol", "markets-archive", "forecast-archive", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    mode_choices: list[str] = [mode.value for mode in AcquisitionMode]
    parser.add_argument("--forecast-mode", choices=mode_choices, required=True)
    parser.add_argument("--market-url", default="https://external-api.kalshi.com/trade-api/v2")
    parser.add_argument("--forecast-url", default="https://single-runs-api.open-meteo.com")
    arguments = parser.parse_args(argv)
    plan_path: Path = arguments.plan
    protocol_path: Path = arguments.protocol
    markets_path: Path = arguments.markets_archive
    forecasts_path: Path = arguments.forecast_archive
    output: Path = arguments.out
    market_url: str = arguments.market_url
    forecast_url: str = arguments.forecast_url
    mode: str = arguments.forecast_mode
    plan = decode_comparison_plan(narrow_json_to_dict(load_json_str(plan_path.read_text("utf-8"))))
    if sha256(protocol_path.read_bytes()).hexdigest() != plan["protocol_digest"]:
        raise ValueError("[FORECAST-010] protocol file differs from frozen plan digest")
    fingerprint = market_fingerprint(determinism, config_test_hooks.get_env)
    source_digest = implementation_digest()
    collector, market_client = build_collector(markets_path, market_url, AcquisitionMode.REPLAY)
    with closing(market_client), closing(_test_hooks.client_factory(60.0, None)) as forecast_client:
        archive = ResponseArchive(
            forecasts_path, forecast_url, forecast_client, _test_hooks.clock, _test_hooks.pause
        )
        readers = {
            AcquisitionMode.CAPTURE: archive.capture,
            AcquisitionMode.RESUME: archive.resume_capture,
            AcquisitionMode.REPLAY: archive.replay,
        }
        rows, counts = collect_forecast_rows(plan, collector, readers[AcquisitionMode(mode)], print)
    report = evaluate_forecasts(rows, plan)
    report["collection_counts"] = dict(counts)
    report["implementation_sha256"] = source_digest
    arms = require_dict(report, "arms")
    observations = tuple(
        Observation(name=name + "_brier", value=require_float(require_dict(arms, name), "brier"))
        for name in sorted(arms)
    )
    sources = (
        *markets_path.glob("*.json"),
        *forecasts_path.glob("*.json"),
        plan_path,
        protocol_path,
    )
    record = write_market_report(
        report,
        fingerprint,
        output,
        sources,
        plan["name"],
        sha256(dump_json_str(encode_comparison_plan(plan)).encode()).hexdigest(),
        observations,
    )
    sys.stdout.write(f"Report: {output}\nRun record: {record}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
