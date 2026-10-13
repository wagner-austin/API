"""Inspect a demo account with authenticated reads; never submits an order."""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import JSONObject, dump_json_str


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Read local credential files, inspect the demo account and save observations.

    Args:
        argv: Explicit arguments or None for process arguments.
        pin: Numeric initializer run before package imports.

    Returns:
        Zero after writing verified evidence to a new file and standard output.

    Raises:
        OSError: Credentials cannot be read or output cannot be created.
        ValueError: Credentials, account evidence or an HTTP response are invalid.
        HTTPError: A request failed; no retry or order submission occurs.
    """
    pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.kalshi_auth import KalshiSigner
    from covenant_ml.markets.kalshi_demo import DEMO_ORIGIN, encode_preflight

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-id-file", type=Path, required=True)
    parser.add_argument("--private-key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    key_path: Path = args.key_id_file
    private_path: Path = args.private_key_file
    output: Path = args.output
    signer = KalshiSigner(key_path.read_text(encoding="utf-8").strip(), private_path.read_bytes())
    with closing(_test_hooks.client_factory(10, None)) as client:
        evidence = _test_hooks.demo_account_factory(client, signer, _test_hooks.clock).preflight()
    report: JSONObject = {
        "mode": "demo_account_preflight",
        "origin": DEMO_ORIGIN,
        "subaccount": 0,
        "orders_sent": 0,
        "live_ready": False,
        "execution_ready": False,
        "evidence": encode_preflight(evidence),
    }
    payload = dump_json_str(report) + "\n"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
