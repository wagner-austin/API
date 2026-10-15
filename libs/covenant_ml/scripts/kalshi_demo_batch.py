"""Prepare, advance, reconcile or inspect a bounded demo-only execution batch."""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import dump_json_str, require_list


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Run one batch action with private credential files and shared account ownership.

    Args:
        argv: Explicit arguments or None for process arguments.
        pin: Numeric initialization before package imports.

    Returns:
        Zero after successful completion; exceptions propagate to the process.

    Raises:
        ValueError: Account, plan, evidence, freshness or risk validation fails.
        OSError: Required credentials or durable batch evidence cannot be accessed.
        HTTPError: Demo request fails; an existing reservation is never resent.
    """
    pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.kalshi_auth import KalshiSigner
    from covenant_ml.markets.validation import read_document, require_nonempty

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-id-file", type=Path, required=True)
    parser.add_argument("--private-key-file", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--predecessor", type=Path, required=True)
    prepare.add_argument("--tickers-file", type=Path, required=True)
    prepare.add_argument("--limit-price", type=int, required=True)
    commands.add_parser("advance")
    commands.add_parser("reconcile")
    commands.add_parser("status")
    args = parser.parse_args(argv)
    key_path: Path = args.key_id_file
    private_path: Path = args.private_key_file
    root: Path = args.root
    action: str = args.action
    signer = KalshiSigner(key_path.read_text("utf-8").strip(), private_path.read_bytes())
    guard_root = Path.home() / ".local" / "share" / "covenant-ml" / "demo-batches"
    with closing(_test_hooks.client_factory(10, None)) as client:
        service = _test_hooks.demo_batch_factory(client, signer, _test_hooks.clock, guard_root)
        if action == "prepare":
            predecessor: Path = args.predecessor
            ticker_path: Path = args.tickers_file
            price: int = args.limit_price
            tickers = tuple(
                require_nonempty({"ticker": value}, "ticker")
                for value in require_list(read_document(ticker_path), "tickers")
            )
            report = service.prepare(root, predecessor, tickers, price)
        elif action == "advance":
            report = service.advance(root)
        elif action == "reconcile":
            report = service.reconcile(root)
        else:
            report = service.status(root)
    sys.stdout.write(dump_json_str(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
