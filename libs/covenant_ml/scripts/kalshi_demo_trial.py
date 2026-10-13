"""Prepare, submit once and reconcile one bounded demo order; no production endpoint."""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import dump_json_str


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Run one explicit demo trial action using private credential files.

    Args:
        argv: Explicit arguments or None to read process arguments.
        pin: Numeric initializer before package imports.

    Returns:
        Zero after the action; all failures propagate as nonzero process exits.

    Raises:
        ValueError: Identity, risk, freshness or account reconciliation failed.
        OSError: Required credentials or trial evidence cannot be accessed.
        HTTPError: Demo HTTP request failed; an existing reservation is never released.
    """
    pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.kalshi_auth import KalshiSigner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-id-file", type=Path, required=True)
    parser.add_argument("--private-key-file", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--ticker", required=True)
    prepare.add_argument("--limit-price", type=int, required=True)
    commands.add_parser("submit")
    commands.add_parser("reconcile")
    commands.add_parser("status")
    args = parser.parse_args(argv)
    key_path: Path = args.key_id_file
    private_path: Path = args.private_key_file
    root: Path = args.root
    action: str = args.action
    signer = KalshiSigner(key_path.read_text("utf-8").strip(), private_path.read_bytes())
    with closing(_test_hooks.client_factory(10, None)) as client:
        service = _test_hooks.demo_trial_factory(client, signer, _test_hooks.clock)
        if action == "prepare":
            ticker: str = args.ticker
            price: int = args.limit_price
            report = service.prepare(root, ticker, price)
        elif action == "submit":
            report = service.submit(root)
        elif action == "reconcile":
            report = service.reconcile(root)
        else:
            report = service.status(root)
    sys.stdout.write(dump_json_str(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
