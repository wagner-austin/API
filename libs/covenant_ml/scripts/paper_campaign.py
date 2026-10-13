"""Freeze a multi-date observation campaign or execute one scheduled acquisition tick."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from uuid import uuid4

from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import JSONObject, dump_json_str


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Prepare fixed daily windows or persist one tick's start and successful result.

    Args:
        argv: Explicit command arguments or process arguments.
        pin: Numeric initializer run before importing the prediction stack.

    Returns:
        Zero for successful planning or a fully completed tick.

    Raises:
        ValueError: Campaign inputs, evidence or timing are invalid.
        RuntimeError: A launched acquisition failed or a prior attempt was interrupted.
        OSError: Evidence persistence or subprocess launch fails.
        OperationalError: Another tick holds the campaign's exclusive lock.
    """
    pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.campaign_execution import tick_campaign
    from covenant_ml.markets.campaign_planning import (
        persist_campaign,
        plan_campaign,
        write_new_json,
    )
    from covenant_ml.markets.provenance import implementation_digest

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    for key in ("root", "protocol", "models-00", "models-12"):
        prepare.add_argument("--" + key, required=True, type=Path)
    for key in ("name", "first", "last"):
        prepare.add_argument("--" + key, required=True)
    tick = commands.add_parser("tick")
    tick.add_argument("--root", required=True, type=Path)
    tick.add_argument("--digest", required=True)
    tick.add_argument("--market-url", default="https://external-api.kalshi.com/trade-api/v2")
    tick.add_argument("--forecast-url", default="https://single-runs-api.open-meteo.com")
    args = parser.parse_args(argv)
    action: str = args.action
    root: Path = args.root
    root = root.resolve()
    if action == "prepare":
        name: str = args.name
        first: str = args.first
        last: str = args.last
        protocol: Path = args.protocol
        model00: Path = args.models_00
        model12: Path = args.models_12
        campaign = plan_campaign(
            name,
            date.fromisoformat(first),
            date.fromisoformat(last),
            _test_hooks.clock(),
            protocol,
            (model00, model12),
            implementation_digest(),
        )
        digest = persist_campaign(campaign, root)
        sys.stdout.write(f"Campaign: {root / 'campaign.json'}\nSHA-256: {digest}\n")
        return 0
    manifest_digest: str = args.digest
    market_url: str = args.market_url
    forecast_url: str = args.forecast_url
    ticks = root / "ticks"
    ticks.mkdir(exist_ok=True)
    identity = uuid4().hex
    write_new_json(
        ticks / (identity + ".started.json"),
        {
            "manifest_digest": manifest_digest,
            "started_at": _test_hooks.clock(),
        },
    )
    counts = tick_campaign(
        root,
        manifest_digest,
        _test_hooks.clock,
        _test_hooks.campaign_runner,
        market_url,
        forecast_url,
    )
    result: JSONObject = {
        "manifest_digest": manifest_digest,
        "finished_at": _test_hooks.clock(),
        "counts": counts,
    }
    write_new_json(ticks / (identity + ".json"), result)
    sys.stdout.write(dump_json_str(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
