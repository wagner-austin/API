"""Re-measure one written validation seed summary and post the reading on its task.

The evaluator runs this after ``research_proposals summarize``: it reads the
journal, the trial and the written summary with its RunRecord, recomputes the
summary through the same owner, and posts the verified reading as ``RESEARCH
SUMMARY LOAD-REPORT`` under the task's ``hpc-report-<task>`` identity. Nothing
is written, staged or submitted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from platform_core.json_utils import dump_json_str

from covenant_ml.markets.research_summary_report import report_summary


def main(argv: list[str] | None = None) -> int:
    """Verify and publish one written seed summary.

    Args:
        argv: Explicit flags or process arguments.

    Returns:
        Zero once the reading was posted and printed.

    Raises:
        ValueError: A malformed task, or a summary its journal does not reproduce.
        OSError: An input cannot be read.
        AppError: The board refused the reading.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--report-task", required=True)
    args = parser.parse_args(argv)
    ledger: Path = args.ledger
    trial: Path = args.trial
    summary: Path = args.summary
    task: str = args.report_task
    report = report_summary(ledger, trial, summary, task)
    sys.stdout.write(dump_json_str(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
