"""Admit validation proposals under a frozen policy and inspect durable attempts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import JSONObject, dump_json_str


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Initialize or inspect the journal, or record one validation proposal.

    Args:
        argv: Explicit arguments or process arguments.
        pin: Numerical initializer before importing covenant_ml.

    Returns:
        Zero on initialization/status/admission, two for a policy refusal.

    Raises:
        ValueError: Malformed policy/proposal or altered journal evidence.
        OSError: Inputs cannot be read.
        sqlite3.OperationalError: Journal missing or locked.
    """
    pin()
    from covenant_ml.markets import _test_hooks
    from covenant_ml.markets.research_journal import (
        admit_proposal,
        encode_research_attempt,
        research_journal,
        research_policy_digest,
    )
    from covenant_ml.markets.research_records import decode_research_policy
    from covenant_ml.markets.validation import parse_timestamp, read_document

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("init")
    commands.add_parser("status")
    propose = commands.add_parser("propose")
    propose.add_argument("--proposal", required=True, type=Path)
    args = parser.parse_args(argv)
    action: str = args.action
    policy_path: Path = args.policy
    ledger_path: Path = args.ledger
    policy = decode_research_policy(read_document(policy_path))
    journal = research_journal(policy)
    code = 0
    if action == "init":
        attempts = journal.initialize(ledger_path)
    elif action == "status":
        attempts = journal.read(ledger_path)
    else:
        proposal_path: Path = args.proposal
        row = admit_proposal(
            ledger_path,
            policy,
            read_document(proposal_path),
            parse_timestamp(_test_hooks.clock()),
        )
        attempts = (row,)
        if row["status"] == "refused":
            code = 2
    report: JSONObject = {
        "policy_digest": research_policy_digest(policy),
        "attempts": [encode_research_attempt(row) for row in attempts],
    }
    sys.stdout.write(dump_json_str(report) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
