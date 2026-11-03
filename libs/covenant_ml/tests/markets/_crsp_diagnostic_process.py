"""Allocation arguments and real subprocess driver shared by CRSP diagnostics."""

from __future__ import annotations

import subprocess
import sys

from platform_core.json_utils import dump_json_str

from covenant_ml.markets.crsp_cohort_records import encode_crsp_cohort_plan

from ._crsp_process_fixtures import client_environment
from ._crsp_refresh_fixtures import RefreshCase


def diagnostic_arguments(
    case: RefreshCase, seconds: int, max_bytes: int, operation: str
) -> list[str]:
    """Render the same explicit native allocation arguments in both test processes."""
    root = case.cohort.extraction.root
    return [
        operation,
        "--cohort",
        str(case.cohort.destination / "bundle.json"),
        "--cohort-sha256",
        case.digest,
        "--plan",
        str(root / "plan.json"),
        "--access",
        str(root / "access.json"),
        "--max-rows",
        "20000",
        "--max-relation-bytes",
        str(max_bytes),
        "--max-partition-bytes",
        "4000000",
        "--seconds",
        str(seconds),
        "--scratch-env",
        "TMPDIR",
        "--output-env",
        "WRDS_OUTPUT_DIR",
    ]


def prepare_allocation(case: RefreshCase) -> dict[str, str]:
    """Create only the allocation's public plan and empty output directory."""
    root = case.cohort.extraction.root
    (root / "plan.json").write_text(dump_json_str(encode_crsp_cohort_plan(case.cohort.plan)))
    (root / "native-output").mkdir()
    return {
        "WRDS_OUTPUT_DIR": str(root / "native-output"),
        "TMPDIR": str(case.cohort.extraction.scratch),
    }


def run_diagnostic_process(
    case: RefreshCase, body: str, seconds: int, max_bytes: int, operation: str
) -> subprocess.CompletedProcess[str]:
    environment = client_environment(case.cohort.extraction.root, body)
    environment.update(prepare_allocation(case))
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.crsp_extraction",
            *diagnostic_arguments(case, seconds, max_bytes, operation),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
