"""Admit validation proposals under a frozen policy and inspect durable attempts.

Every operation here is the evaluator's: a proposer's program runs only inside
the isolated process boundary, whose output reaches ``propose`` as a file. The
``select`` operation seals admission on one completed attempt and binds its
final manifest; ``release`` consumes its one-shot authorization and writes a
private grant for ``research_trial --final-grant`` on the compute node.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from platform_core.determinism_cpu import PinProtocol, pin_single_thread
from platform_core.json_utils import JSONObject, dump_json_str


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Initialize or inspect the journal, record a proposal, run it or select it.

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
    from covenant_ml.markets.research_cluster import decode_research_cluster
    from covenant_ml.markets.research_execution import reconcile_attempt, submit_attempt
    from covenant_ml.markets.research_final import release_final, select_candidate
    from covenant_ml.markets.research_journal import (
        admit_proposal,
        encode_research_attempt,
        research_journal,
        research_policy_digest,
    )
    from covenant_ml.markets.research_keys import EvaluatorKeys, decode_research_access
    from covenant_ml.markets.research_records import decode_research_policy
    from covenant_ml.markets.research_trial_records import decode_research_trial
    from covenant_ml.markets.validation import parse_timestamp, read_document

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("init")
    commands.add_parser("status")
    propose = commands.add_parser("propose")
    propose.add_argument("--proposal", required=True, type=Path)
    for name in ("submit", "reconcile"):
        execution = commands.add_parser(name)
        execution.add_argument("--trial", required=True, type=Path)
        execution.add_argument("--cluster", required=True, type=Path)
        execution.add_argument("--directory", required=True, type=Path)
    select = commands.add_parser("select")
    select.add_argument("--trial", required=True, type=Path)
    select.add_argument("--final-manifest", required=True, type=Path)
    release = commands.add_parser("release")
    release.add_argument("--trial", required=True, type=Path)
    release.add_argument("--final-manifest", required=True, type=Path)
    release.add_argument("--access", required=True, type=Path)
    release.add_argument("--directory", required=True, type=Path)
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
    elif action == "propose":
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
    else:
        trial_path: Path = args.trial
        trial = decode_research_trial(read_document(trial_path))
        if trial["policy"] != policy:
            raise ValueError("RESEARCH_EXECUTION_REQUEST: trial policy differs from journal")
        if action in {"select", "release"}:
            final_manifest: Path = args.final_manifest
            if action == "select":
                attempts = (select_candidate(ledger_path, trial, final_manifest),)
            else:
                access: Path = args.access
                destination: Path = args.directory
                keys = EvaluatorKeys(decode_research_access(read_document(access)))
                attempts = (
                    release_final(ledger_path, trial, final_manifest, destination, keys=keys),
                )
        else:
            cluster_path: Path = args.cluster
            directory: Path = args.directory
            cluster = decode_research_cluster(read_document(cluster_path))
            operation = submit_attempt if action == "submit" else reconcile_attempt
            attempts = (operation(ledger_path, trial, cluster, directory.resolve()),)
    report: JSONObject = {
        "policy_digest": research_policy_digest(policy),
        "attempts": [encode_research_attempt(row) for row in attempts],
    }
    sys.stdout.write(dump_json_str(report) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
