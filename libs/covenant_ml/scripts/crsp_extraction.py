"""Extract CRSP partitions, capture cohort diagnostics, or verify collected evidence.

``extract`` runs inside the scheduler allocation: plaintext stays in its
private scratch and only ciphertext, the receipt and metrics reach the output
directory named by an environment variable. ``refresh`` uses those same
allocation paths and bounds for a pinned validation cohort, writing encrypted
differences and aggregate metrics for the native controller's RunRecord.
``actions`` captures additional bounded corporate-action fields through that
same private lifecycle, without changing or admitting a study dataset.
``verify`` runs on the evaluator
host against the collected directory, posts its metadata on the named board
task under the task's WRDS load-report identity, then writes the same metadata
to standard output; it never prints or posts a source row or key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    require_dict,
    require_int,
    require_list,
)
from platform_core.load_report import post_load_report, report_identity

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.crsp_action_cases import inspect_crsp_actions
from covenant_ml.markets.crsp_cohort_records import decode_crsp_cohort_plan
from covenant_ml.markets.crsp_diagnostic import capture_crsp_diagnostic
from covenant_ml.markets.crsp_extraction_records import decode_crsp_limits
from covenant_ml.markets.crsp_revision import revise_crsp_protocol, verify_crsp_revision
from covenant_ml.markets.crsp_verify import verify_crsp_extraction
from covenant_ml.markets.crsp_writer import extract_crsp
from covenant_ml.markets.research_keys import EvaluatorKeys, ResearchAccess, decode_research_access
from covenant_ml.markets.validation import read_document


def _parser() -> argparse.ArgumentParser:
    """Declare capture and verification commands with every bound and path explicit."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--protocol", required=True, type=Path)
    extract.add_argument("--source", required=True)
    refresh, actions = (commands.add_parser(name) for name in ("refresh", "actions"))
    inspect = commands.add_parser("inspect-actions")
    inspect.add_argument("--actions", required=True, type=Path)
    inspect.add_argument("--actions-sha256", required=True)
    for diagnostic in (refresh, actions, inspect):
        diagnostic.add_argument("--cohort", required=True, type=Path)
        diagnostic.add_argument("--cohort-sha256", required=True)
        diagnostic.add_argument("--plan", required=True, type=Path)
    for capture in (extract, refresh, actions, inspect):
        capture.add_argument("--access", required=True, type=Path)
        capture.add_argument("--scratch-env", required=True)
        capture.add_argument("--output-env", required=True)
    for capture in (extract, refresh, actions):
        capture.add_argument("--setup", type=Path)
        capture.add_argument("--max-rows", required=True, type=int)
        capture.add_argument("--max-relation-bytes", required=True, type=int)
        capture.add_argument("--max-partition-bytes", required=True, type=int)
        capture.add_argument("--seconds", required=True, type=int)
    verify = commands.add_parser("verify")
    verify.add_argument("--collected", required=True, type=Path)
    verify.add_argument("--access", required=True, type=Path)
    verify.add_argument("--scratch", required=True, type=Path)
    mode = verify.add_mutually_exclusive_group()
    mode.add_argument("--open-final", action="store_true")
    mode.add_argument("--ciphertext-only", action="store_true")
    verify.add_argument("--report-task", required=True)
    for name in ("revise-protocol", "verify-revision"):
        revision = commands.add_parser(name)
        revision.add_argument("--outputs", required=True, type=Path)
        revision.add_argument("--receipt-sha256", required=True)
        revision.add_argument("--previous-protocol", required=True, type=Path)
        revision.add_argument("--protocol", required=True, type=Path)
        revision.add_argument("--access", required=True, type=Path)
        revision.add_argument("--destination", required=True, type=Path)
        revision.add_argument("--report-task", required=True)
    return parser


def _inspect_actions(args: argparse.Namespace, settings: ResearchAccess) -> None:
    """Write inspection metadata and native metrics only after authenticated cleanup."""
    environment = config_test_hooks.get_environment()
    cohort: Path = args.cohort
    cohort_digest: str = args.cohort_sha256
    plan: Path = args.plan
    actions: Path = args.actions
    actions_digest: str = args.actions_sha256
    scratch_variable: str = args.scratch_env
    output_variable: str = args.output_env
    output = Path(environment[output_variable])
    targets = (output / "action-cases.json", output / "metrics.json")
    if any(path.exists() or path.is_symlink() for path in targets):
        raise FileExistsError("CRSP_ACTION_OUTPUT: refusing to replace an inspection")
    result = inspect_crsp_actions(
        cohort,
        cohort_digest,
        actions,
        actions_digest,
        decode_crsp_cohort_plan(read_document(plan)),
        EvaluatorKeys(settings),
        scratch=Path(environment[scratch_variable]),
    )
    with targets[0].open("x", encoding="utf-8") as stream:
        stream.write(dump_json_str(result))
    metrics: list[JSONObject] = [
        {"name": "cases", "value": float(len(require_list(result, "cases")))}
    ]
    with targets[1].open("x", encoding="utf-8") as stream:
        stream.write(dump_json_str(metrics))


def main(argv: list[str] | None = None) -> int:
    """Run one command.

    Args:
        argv: Explicit arguments, or the process arguments.

    Returns:
        Zero after a complete extraction, or a verification posted on its task.

    Raises:
        KeyError: A named scheduler environment variable is absent.
        ValueError: Any writer or verifier refusal, or a report task that is
            not a full task id; nothing usable is published.
        OSError: Input, key or storage access fails.
        AppError: The board's credentials are missing or it refuses the
            verification's report; nothing is written to standard output.
    """
    args = _parser().parse_args(argv)
    command: str = args.command
    access: Path = args.access
    settings = decode_research_access(read_document(access))
    if command == "inspect-actions":
        _inspect_actions(args, settings)
        return 0
    if command in {"extract", "refresh", "actions"}:
        environment = config_test_hooks.get_environment()
        scratch_variable: str = args.scratch_env
        output_variable: str = args.output_env
        setup: Path | None = args.setup
        rows: int = args.max_rows
        relation_bytes: int = args.max_relation_bytes
        partition_bytes: int = args.max_partition_bytes
        seconds: int = args.seconds
        bounds = decode_crsp_limits(
            {
                "max_rows": rows,
                "max_relation_bytes": relation_bytes,
                "max_partition_bytes": partition_bytes,
                "seconds": seconds,
            }
        )
        output = Path(environment[output_variable])
        if command in {"refresh", "actions"}:
            cohort: Path = args.cohort
            cohort_digest: str = args.cohort_sha256
            plan: Path = args.plan
            name = "comparison" if command == "refresh" else "actions"
            result = capture_crsp_diagnostic(
                cohort,
                cohort_digest,
                decode_crsp_cohort_plan(read_document(plan)),
                EvaluatorKeys(settings),
                operation="refresh" if command == "refresh" else "actions",
                scratch=Path(environment[scratch_variable]),
                destination=output / name,
                limits=bounds,
                setup=setup,
            )
            with (output / (name + ".json")).open("x", encoding="utf-8") as stream:
                stream.write(dump_json_str(result))
            counts = require_dict(result, "rows")
            metrics: list[JSONObject] = []
            if command == "refresh":
                metrics = [
                    {
                        "name": role + "." + metric,
                        "value": float(require_int(require_dict(counts, role), metric)),
                    }
                    for role in counts
                    for metric in ("old", "fresh", "unchanged", "added", "removed", "changed")
                ]
            else:
                metrics = [
                    {"name": role, "value": float(require_int(counts, role))} for role in counts
                ]
            with (output / "metrics.json").open("x", encoding="utf-8") as stream:
                stream.write(dump_json_str(metrics))
            return 0
        protocol: Path = args.protocol
        source: str = args.source
        extract_crsp(
            protocol,
            settings,
            source=source,
            setup=setup,
            scratch=Path(environment[scratch_variable]),
            output=output,
            limits=bounds,
        )
        return 0
    task: str = args.report_task
    report_identity(task, "wrds")  # refuses a malformed task before a bundle is opened
    if command in {"revise-protocol", "verify-revision"}:
        outputs: Path = args.outputs
        receipt_digest: str = args.receipt_sha256
        previous_protocol: Path = args.previous_protocol
        current_protocol: Path = args.protocol
        destination: Path = args.destination
        if command == "revise-protocol":
            revise_crsp_protocol(
                outputs, receipt_digest, previous_protocol, current_protocol, settings, destination
            )
        result = verify_crsp_revision(
            outputs, receipt_digest, previous_protocol, current_protocol, settings, destination
        )
        heading = "WRDS PROTOCOL REVISION LOAD-REPORT\n"
    else:
        collected: Path = args.collected
        scratch: Path = args.scratch
        open_final: bool = args.open_final
        ciphertext_only: bool = args.ciphertext_only
        result = verify_crsp_extraction(
            collected,
            EvaluatorKeys(settings),
            scratch=scratch,
            open_final=open_final,
            ciphertext_only=ciphertext_only,
        )
        heading = "WRDS EXTRACTION VERIFY LOAD-REPORT\n"
    payload = dump_json_str(result)
    post_load_report(
        post=_test_hooks.http_post,
        read_text=_test_hooks.stack_text,
        task_id=task,
        body=heading + payload,
        family="wrds",
    )
    sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
