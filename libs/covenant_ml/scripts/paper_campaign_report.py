"""Evaluate saved frozen-campaign evidence offline, with equal-event-date comparisons."""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import require_dict, require_int
from platform_core.run_record import Observation


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Persist an offline campaign evaluation and complete source provenance.

    Args:
        argv: Explicit CLI arguments or process arguments.
        pin: Numeric initializer called before importing the scoring stack.

    Returns:
        Zero after report and RunRecord persistence.

    Raises:
        ValueError: Frozen inputs or saved acquisition evidence are inconsistent.
        OSError: Required evidence is missing or report persistence fails.
        FileExistsError: Output already exists.
    """
    determinism = pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.archive import ResponseArchive
    from covenant_ml.markets.campaign_evaluation import evaluate_campaign
    from covenant_ml.markets.provenance import implementation_digest, market_fingerprint
    from covenant_ml.markets.reporting import write_market_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--digest", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--market-url", default="https://external-api.kalshi.com/trade-api/v2")
    args = parser.parse_args(argv)
    root: Path = args.root
    digest: str = args.digest
    output: Path = args.out
    url: str = args.market_url
    with closing(_test_hooks.client_factory(60.0, None)) as client:

        def archive_factory(path: Path) -> ResponseArchive:
            return ResponseArchive(path, url, client, _test_hooks.clock, _test_hooks.pause)

        report, sources = evaluate_campaign(
            root.resolve(), digest, _test_hooks.clock(), archive_factory
        )
    report["implementation_sha256"] = implementation_digest()
    sidecar = write_market_report(
        report,
        market_fingerprint(determinism, config_test_hooks.get_env),
        output,
        sources,
        "prospective-campaign-evaluation",
        digest,
        tuple(
            Observation(
                name=key + "_eligible_dates",
                value=float(require_int(require_dict(report, key), "eligible_dates")),
            )
            for key in ("primary_12utc", "secondary_00utc")
        ),
    )
    sys.stdout.write(f"Campaign evaluation: {output}\nRun record: {sidecar}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
