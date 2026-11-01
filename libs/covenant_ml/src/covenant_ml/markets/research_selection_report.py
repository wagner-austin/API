"""Publish the evaluator's own selection measurement through the shared reporter.

The journal remains the only owner of ranking and selection. This process
boundary records its actual outcome, including refusals, without opening data
or reading keys. A failed publication propagates; it cannot undo selection.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from platform_core.cli_boundary import run_cli
from platform_core.json_utils import JSONObject, dump_json_str, load_json_str, narrow_json_to_dict
from platform_core.load_report import post_load_report, report_identity

from . import _test_hooks
from .research_final import select_candidate
from .research_journal import research_journal, research_policy_digest
from .research_records import decode_research_proposal
from .research_trial_records import ResearchTrial, encode_research_trial


def report_selection(
    ledger: Path, trial: ResearchTrial, final_manifest: Path, task: str
) -> tuple[int, JSONObject]:
    """Execute selection and publish measured status and before/after identities.

    Args:
        ledger: Evaluator-owned journal, never copied or reset by this command.
        trial: Exact requested attempt and frozen policy.
        final_manifest: Final ciphertext manifest; no key is consulted.
        task: Full board task id for the existing HPC evidence identity.

    Returns:
        The shared CLI boundary's exit status and the published measurement.

    Raises:
        ValueError: Invalid report task or unreadable journal state.
        OSError: Evidence files cannot be read.
        AppError: Board publication failed; committed selection remains committed.
    """
    report_identity(task, "hpc")
    journal = research_journal(trial["policy"])
    journal.read(ledger)
    before = sha256(ledger.read_bytes()).hexdigest()
    manifest_digest = sha256(final_manifest.read_bytes()).hexdigest()
    started = _test_hooks.clock()
    messages: list[str] = []

    def select(argv: Sequence[str] | None) -> int:
        """Run the existing transaction under the shared refusal boundary."""
        select_candidate(ledger, trial, final_manifest)
        return 0

    code = run_cli(select, emit=messages.append, emit_error=messages.append)
    rows = journal.read(ledger)
    selected: list[JSONObject] = [
        {
            "number": row["number"],
            "candidate": row["candidate"],
            "seed": decode_research_proposal(narrow_json_to_dict(load_json_str(row["request"])))[
                "seed"
            ],
            "selection": row["selection"],
            "final_manifest_digest": row["final_manifest_digest"],
        }
        for row in rows
        if row["selection"]
    ]
    report: JSONObject = {
        "kind": "RESEARCH_SELECTION_EVIDENCE",
        "started_at": started,
        "finished_at": _test_hooks.clock(),
        "controller_python": sys.executable,
        "controller_python_sha256": sha256(Path(sys.executable).read_bytes()).hexdigest(),
        "reporter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "policy_digest": research_policy_digest(trial["policy"]),
        "request_sha256": sha256(dump_json_str(encode_research_trial(trial)).encode()).hexdigest(),
        "ledger": str(ledger.resolve()),
        "journal_sha256_before": before,
        "journal_sha256_after": sha256(ledger.read_bytes()).hexdigest(),
        "final_manifest_digest": manifest_digest,
        "attempt": trial["number"],
        "seed": trial["proposal"]["seed"],
        "exit_status": code,
        "messages": list(messages),
        "selected": list(selected),
    }
    post_load_report(
        post=_test_hooks.http_post,
        read_text=_test_hooks.stack_text,
        task_id=task,
        body="RESEARCH SELECTION LOAD-REPORT\n" + dump_json_str(report),
        family="hpc",
    )
    return code, report


__all__ = ["report_selection"]
