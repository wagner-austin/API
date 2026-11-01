"""Read one privately prepared research attempt and post the reading on its task.

The evaluator runs this beside the controller after ``prepare_bundle``: it
reads the frozen policy, the admission journal and the attempt's ``run.json``
and posts them as ``RESEARCH PREPARATION LOAD-REPORT`` under the task's
``hpc-report-<task>`` identity. Nothing is staged, registered or submitted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from platform_core.json_utils import dump_json_str

from covenant_ml.markets.research_preparation_report import report_preparation


def main(argv: list[str] | None = None) -> int:
    """Measure and publish one prepared attempt.

    Args:
        argv: Explicit flags or process arguments.

    Returns:
        Zero once the reading was posted and printed.

    Raises:
        ValueError: A malformed task, or a recipe foreign to the policy or journal.
        OSError: An input cannot be read.
        AppError: The board refused the reading.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--report-task", required=True)
    args = parser.parse_args(argv)
    policy: Path = args.policy
    ledger: Path = args.ledger
    recipe: Path = args.recipe
    task: str = args.report_task
    report = report_preparation(policy, ledger, recipe, task)
    sys.stdout.write(dump_json_str(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
