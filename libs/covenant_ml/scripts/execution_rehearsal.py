"""Rehearse $20 execution controls locally; this command never submits exchange orders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import JSONObject, dump_json_str, require_dict


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Run one durable local rehearsal operation and print its accounting report.

    Args:
        argv: Explicit arguments or None to read process arguments.
        pin: Numeric initializer called before importing the package.

    Returns:
        Zero after a verified journal operation; no exchange order is sent.

    Raises:
        ValueError: Inputs, risk limits or reconciliation evidence are invalid.
        OSError: Evidence files cannot be read or created.
        OperationalError: A journal is missing, inaccessible or held by another writer.
    """
    pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.campaign_execution import read_document
    from covenant_ml.markets.execution_journal import (
        append_execution_event,
        initialize_execution_journal,
        read_execution_journal,
    )
    from covenant_ml.markets.execution_records import (
        decode_intent,
        decode_native_order,
        encode_intent,
        encode_receipt,
        order_payload,
    )
    from covenant_ml.markets.execution_state import execution_report
    from covenant_ml.markets.validation import parse_timestamp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("init")
    commands.add_parser("status")
    reserve = commands.add_parser("reserve")
    reserve.add_argument("--intent", required=True, type=Path)
    receipt = commands.add_parser("receipt")
    receipt.add_argument("--order", required=True, type=Path)
    settlement = commands.add_parser("settlement")
    settlement.add_argument("--client-order-id", required=True)
    settlement.add_argument("--evidence", required=True, type=Path)
    halt = commands.add_parser("halt")
    halt.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    action: str = args.action
    path: Path = args.ledger
    payload: JSONObject | None = None
    if action == "init":
        state = initialize_execution_journal(path)
    elif action == "status":
        state = read_execution_journal(path)
    else:
        event: JSONObject = {"action": action, "observed_ms": parse_timestamp(_test_hooks.clock())}
        if action == "reserve":
            intent_path: Path = args.intent
            intent = decode_intent(read_document(intent_path))
            event["data"] = encode_intent(intent)
            payload = order_payload(intent)
        elif action == "receipt":
            order_path: Path = args.order
            order = decode_native_order(require_dict(read_document(order_path), "order"))
            event["client_order_id"] = order["client_order_id"]
            event["data"] = encode_receipt(order)
        elif action == "settlement":
            identity: str = args.client_order_id
            evidence_path: Path = args.evidence
            event["client_order_id"] = identity
            event["data"] = read_document(evidence_path)
        else:
            reason: str = args.reason
            event["reason"] = reason
        state = append_execution_event(path, event)
    report = execution_report(state)
    report["inspection_only_order_payload"] = payload
    sys.stdout.write(dump_json_str(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
