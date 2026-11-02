"""Score a frozen forward weather plan offline from its campaign's saved evidence."""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import require_int
from platform_core.run_record import Observation


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Persist a forward plan's report and the RunRecord binding every source it read.

    Args:
        argv: Explicit CLI arguments or process arguments.
        pin: Numeric initializer called before importing the scoring stack.

    Returns:
        Zero after report and RunRecord persistence.

    Raises:
        ValueError: The plan differs from its pinned digest, or saved evidence is
            inconsistent with the plan, the campaign or its archived responses.
        OSError: Required evidence is missing or report persistence fails.
        FileExistsError: Output already exists.
    """
    determinism = pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.archive import ResponseArchive
    from covenant_ml.markets.forward_plan import load_forward_plan
    from covenant_ml.markets.forward_scoring import score_forward_plan
    from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
    from covenant_ml.markets.reporting import write_market_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--market-url", default="https://external-api.kalshi.com/trade-api/v2")
    parser.add_argument("--forecast-url", default="https://single-runs-api.open-meteo.com")
    args = parser.parse_args(argv)
    plan_path: Path = args.plan
    plan_digest: str = args.plan_sha256
    root: Path = args.root
    output: Path = args.out
    market_url: str = args.market_url
    forecast_url: str = args.forecast_url
    plan = load_forward_plan(plan_path, plan_digest)
    with closing(_test_hooks.client_factory(60.0, None)) as client:

        def market_archive(path: Path) -> ResponseArchive:
            return ResponseArchive(path, market_url, client, _test_hooks.clock, _test_hooks.pause)

        def forecast_archive(path: Path) -> ResponseArchive:
            return ResponseArchive(path, forecast_url, client, _test_hooks.clock, _test_hooks.pause)

        report, sources = score_forward_plan(
            plan,
            plan_digest,
            root.resolve(),
            _test_hooks.clock(),
            market_archive,
            forecast_archive,
        )
    report["implementation_sha256"] = implementation_digest()
    sidecar = write_market_report(
        report,
        market_fingerprint(determinism, config_test_hooks.get_env),
        output,
        (*sources, plan_path.resolve()),
        "forward-weather-evaluation",
        plan_digest,
        tuple(
            Observation(name=key, value=float(require_int(report, key)))
            for key in ("declared_dates", "scored_dates", "eligible_contracts")
        ),
    )
    sys.stdout.write(f"Forward evaluation: {output}\nRun record: {sidecar}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
