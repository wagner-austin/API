"""Run one proposal program in isolation and pass its output to the maintained admission.

This is the evaluator's operation, the only path from a proposer to the study.
The proposer contributes program text and approved context, both pinned by hash
in the evaluator's private specification; ``run_proposer`` executes the program
on the existing rootless lane with no host files, inherited environment,
credentials or network. The evaluator then writes the returned untrusted object,
content-addressed, into its own specification directory and runs the
maintained ``research_proposals propose`` controller, whose journal records the
raw request before decoding it and keeps every refusal. The policy, journal,
evaluator code, final-period data and transport credentials stay on the
evaluator's side of that boundary, and nothing here can select or release a
candidate for final scoring.
"""

from __future__ import annotations

import hashlib
import pathlib

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_int,
    require_list,
    require_str,
)

from fleet.core import _test_hooks, research_control
from fleet.core.research_proposer import run_proposer

_FIELDS = frozenset(
    {"python", "script", "policy", "ledger", "socket", "image", "program", "context", "timeout"}
)


def propose(document: JSONObject, directory: pathlib.Path, log: pathlib.Path) -> JSONObject:
    """Execute a pinned proposal program and record its output as one admission attempt.

    Args:
        document: Evaluator-owned private specification, already digest-verified.
        directory: That specification's private directory; receives the proposal.
        log: Private admission-controller transcript.

    Returns:
        Metadata only: attempt number, status, refusal reason, candidate digest
        and the SHA-256 of the proposal bytes the controller admitted or refused.

    Raises:
        ValueError: RESEARCH_PROPOSAL_FIELDS for unexpected specification fields;
            RESEARCH_PROPOSAL_REPORT unless the controller reports exactly one
            attempt; input drift, RESEARCH_PROPOSER_* isolation refusals and
            RESEARCH_CONTROLLER_PROCESS failures propagate.
        OSError: A pinned input, the daemon or the controller is unavailable.
    """
    if set(document) != _FIELDS:
        raise ValueError("RESEARCH_PROPOSAL_FIELDS: unexpected private proposer fields")
    python = str(research_control.require_path(document, "python"))
    script = research_control.verified_input(document, "script")
    policy = research_control.verified_input(document, "policy")
    ledger = research_control.require_path(document, "ledger")
    program = _test_hooks.read_text(research_control.verified_input(document, "program"))
    context = narrow_json_to_dict(
        load_json_str(_test_hooks.read_text(research_control.verified_input(document, "context")))
    )
    proposal = run_proposer(
        require_str(document, "socket"),
        require_str(document, "image"),
        program,
        context,
        timeout=require_int(document, "timeout"),
    )
    payload = dump_json_str(proposal)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    path = directory / "proposals" / f"{digest}.json"
    _test_hooks.write_text(path, payload)
    report = research_control.invoke(
        (
            python,
            str(script),
            "--policy",
            str(policy),
            "--ledger",
            str(ledger),
            "propose",
            "--proposal",
            str(path),
        ),
        log,
        60,
        accepted=(0, 2),
    )
    attempts = require_list(report, "attempts")
    if len(attempts) != 1:
        raise ValueError("RESEARCH_PROPOSAL_REPORT: expected exactly one recorded attempt")
    row = narrow_json_to_dict(attempts[0])
    return {
        "number": require_int(row, "number"),
        "status": require_str(row, "status"),
        "reason": require_str(row, "reason"),
        "candidate": require_str(row, "candidate"),
        "proposal_sha256": digest,
    }


__all__ = ["propose"]
