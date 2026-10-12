"""Persist prospective predictions from public APIs, or verify them entirely offline."""

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
    require_int,
    require_list,
)
from platform_core.run_record import Observation


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Bind pre-outcome predictions to receipt evidence, frozen inputs and provenance.

    Args:
        argv: Explicit CLI arguments, or process arguments.
        pin: CPU initializer invoked before numeric imports.

    Returns:
        Zero after verified predictions and a RunRecord are persisted.

    Raises:
        ValueError: PAPER-008 for mismatched frozen files; PAPER-009 for invalid
            prediction timing or changed replay predictions.
        FileExistsError: Capture or output would overwrite evidence.
        HTTPStatusError: A required source request failed, without retry.
        OSError: Input or output access failed.
    """
    determinism = pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.archive import ResponseArchive, decode_capture, encode_capture
    from covenant_ml.markets.paper_collection import PaperReader, collect_paper_predictions
    from covenant_ml.markets.paper_records import decode_paper_plan
    from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
    from covenant_ml.markets.reporting import write_market_report
    from covenant_ml.markets.validation import parse_timestamp

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "protocol", "models", "archive", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("capture", "replay"))
    parser.add_argument("--market-url", default="https://external-api.kalshi.com/trade-api/v2")
    parser.add_argument("--forecast-url", default="https://single-runs-api.open-meteo.com")
    args = parser.parse_args(argv)
    plan_path: Path = args.plan
    protocol_path: Path = args.protocol
    model_path: Path = args.models
    root: Path = args.archive
    output: Path = args.out
    mode: str = args.mode
    market_url: str = args.market_url
    forecast_url: str = args.forecast_url
    plan = decode_paper_plan(narrow_json_to_dict(load_json_str(plan_path.read_text("utf-8"))))
    if (
        sha256(protocol_path.read_bytes()).hexdigest() != plan["protocol_digest"]
        or sha256(model_path.read_bytes()).hexdigest() != plan["model_digest"]
    ):
        raise ValueError("[PAPER-008] protocol or fitted-model report differs from frozen digest")
    model_report = narrow_json_to_dict(load_json_str(model_path.read_text("utf-8")))
    capture = mode == "capture"
    seal_path = root / "prediction.json"
    if capture and root.exists():
        raise FileExistsError("[PAPER-010] capture needs a new archive directory")
    fingerprint = market_fingerprint(determinism, config_test_hooks.get_env)
    with closing(_test_hooks.client_factory(60.0, None)) as client:
        market = PaperReader(
            ResponseArchive(
                root / "markets", market_url, client, _test_hooks.clock, _test_hooks.pause
            ),
            plan,
            capture,
        )
        forecast = PaperReader(
            ResponseArchive(
                root / "forecasts", forecast_url, client, _test_hooks.clock, _test_hooks.pause
            ),
            plan,
            capture,
        )
        report = collect_paper_predictions(plan, model_report, market, forecast)
    report["implementation_sha256"] = implementation_digest()
    payload = dump_json_str(report)
    identity = "paper://" + plan["name"]
    if capture:
        seal = decode_capture(
            {
                "url": identity,
                "captured_at": _test_hooks.clock(),
                "body": payload,
                "digest": sha256(payload.encode()).hexdigest(),
            }
        )
    else:
        seal = decode_capture(narrow_json_to_dict(load_json_str(seal_path.read_text("utf-8"))))
    recorded = parse_timestamp(seal["captured_at"])
    latest = require_int(report, "last_received_ms")
    if (
        seal["url"] != identity
        or seal["body"] != payload
        or not latest <= recorded <= parse_timestamp(plan["window_end"])
        or recorded - latest > 60000
    ):
        raise ValueError(
            "[PAPER-009] changed replay or predictions not recorded promptly in window"
        )
    if capture:
        with seal_path.open("x", encoding="utf-8", newline="\n") as destination:
            destination.write(dump_json_str(encode_capture(seal)))
    report["recorded_at"] = seal["captured_at"]
    sources = (*root.rglob("*.json"), plan_path, protocol_path, model_path)
    observations = (
        Observation(
            name="pre_outcome_predictions", value=float(len(require_list(report, "predictions")))
        ),
    )
    record = write_market_report(
        report,
        fingerprint,
        output,
        sources,
        plan["name"],
        sha256(plan_path.read_bytes()).hexdigest(),
        observations,
    )
    sys.stdout.write(f"Predictions: {output}\nRun record: {record}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
